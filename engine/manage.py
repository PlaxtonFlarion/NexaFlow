#
#   __  __
#  |  \/  | __ _ _ __   __ _  __ _  ___
#  | |\/| |/ _` | '_ \ / _` |/ _` |/ _ \
#  | |  | | (_| | | | | (_| | (_| |  __/
#  |_|  |_|\__,_|_| |_|\__,_|\__, |\___|
#                            |___/
#

"""
版权所有 (c) 2024  Framix(画帧秀)
此文件受 Framix(画帧秀) 许可证的保护。您可以在 LICENSE.md 文件中查看详细的许可条款。

Copyright (c) 2024  Framix(画帧秀)
This file is licensed under the Framix(画帧秀) License. See the LICENSE.md file for more details.

# Copyright (c) 2024  Framix(画帧秀)
# このファイルは Framix(画帧秀) ライセンスの下でライセンスされています。詳細は LICENSE.md ファイルを参照してください。
"""

import re
import math
import psutil
import typing
import asyncio
from loguru import logger
from rich.table import Table
from rich.prompt import Prompt
from screeninfo import get_monitors
from engine.device import Device
from engine.terminal import Terminal
from nexacore.design import Design
from nexaflow import const


class ScreenMonitor(object):
    """
    屏幕监视器类，用于获取当前系统屏幕的分辨率信息。
    """

    @staticmethod
    def screen_size(index: int = 0) -> tuple[int, int]:
        """
        获取指定屏幕的分辨率（宽度和高度）。

        Parameters
        ----------
        index : int, optional
            屏幕索引，默认为主屏幕（0）。

        Returns
        -------
        tuple[int, int]
            返回指定屏幕的宽度和高度，格式为 (width, height)。

        Notes
        -----
        - 该方法依赖 `screeninfo` 库，确保环境中已安装该库。
        - 如果系统存在多个显示器，可通过指定 index 获取对应的分辨率。
        """
        screen = get_monitors()[index]
        return screen.width, screen.height


class SourceMonitor(object):
    """
    系统资源监控器类。

    本类用于动态监控系统的 CPU 和内存使用情况，并判断当前资源是否足够稳定以执行视频分析任务。
    包含资源采样、评估和反馈展示等功能。
    """

    history: list[tuple[float, float, float]] = []

    def __init__(self):
        """
        初始化系统资源监控器。

        Notes
        -----
        - 设置 CPU 和内存使用阈值。
        - 计算逻辑根据当前系统的 CPU 核心数量和内存总量动态设定性能标准。
        """
        base_cpu_usage_threshold = 50
        base_mem_usage_threshold = 70

        self.cpu_cores = psutil.cpu_count()
        self.mem_total = psutil.virtual_memory().total / (1024 ** 3)

        self.cpu_usage_threshold = base_cpu_usage_threshold + min(85, int(self.cpu_cores * 0.5))
        self.mem_usage_threshold = max(50, base_mem_usage_threshold - self.mem_total / 10)
        self.mem_spare_threshold = max(1, self.mem_total * 0.2)

    async def monitor(self) -> None:
        """
        异步资源监控任务入口。

        该方法会持续采样系统的 CPU 和内存状态，在采集 5 次数据后对系统资源进行评估。
        若资源达标则终止监控，否则重启采样过程，直到资源稳定为止。

        Notes
        -----
        - 每次评估的周期为 5 次采样，每次采样间隔约为 2 秒。
        - 每次评估后会显示一个资源概览表格，并根据结果决定是否继续监控。
        - 使用 `rich.progress` 显示进度条。

        Workflow
        --------
        1. 初始化进度条并开始资源采样；
        2. 每次采样后记录 CPU 使用率、内存使用率和可用内存；
        3. 采样达到 5 次后调用 `evaluate_resources` 进行评估；
        4. 如果资源评估结果为稳定，终止进度条并展示结果；
        5. 若不稳定，则清空采样记录并重启评估过程；
        6. 每轮采样之间等待 2 秒。
        """
        first_examine = True
        progress = Design.show_progress()
        task = progress.add_task(description=f"Analyzer", total=5)
        progress.start()

        while True:
            current_cpu_usage = psutil.cpu_percent(1)
            memory_info = psutil.virtual_memory()
            current_mem_usage = memory_info.percent
            current_mem_spare = memory_info.available / (1024 ** 3)

            self.history.append((current_cpu_usage, current_mem_usage, current_mem_spare))
            progress.update(task, advance=1)

            if first_examine:
                table, explain = await self.evaluate_resources(True)
                if explain == "stable":
                    progress.update(task, completed=5)
                    progress.stop()
                    return Design.console.print(table)
                first_examine = False

            elif len(self.history) >= 5:
                table, explain = await self.evaluate_resources(False)
                if explain == "stable":
                    progress.update(task)
                    progress.stop()
                    return Design.console.print(table)

                progress.stop()
                Design.console.print(table)
                progress = Design.show_progress()
                task = progress.add_task(description=f"Analyzer", total=5)
                progress.start()

            await asyncio.sleep(2)

    async def evaluate_resources(self, first_examine: bool) -> typing.Coroutine | tuple["Table", str]:
        """
        评估当前采样数据所反映的系统资源状况。

        Parameters
        ----------
        first_examine : bool
            是否为首次采样，用于判断是否清空历史记录。

        Returns
        -------
        tuple
            返回两个元素：
            - Table: 格式化后的资源使用情况表格（rich.Table）。
            - str: 资源稳定状态，值为 "stable" 或 "unstable"。

        Notes
        -----
        - 若平均 CPU 使用率低于阈值，内存使用率低于阈值，且可用内存高于阈值，视为资源稳定；
        - 否则视为资源不稳定，需重新采样；
        - 当检测为稳定后会清空历史记录；
        - 每次结果将以表格形式展示在控制台。
        """
        avg_cpu_usage = sum(i[0] for i in self.history) / len(self.history)
        avg_mem_usage = sum(i[1] for i in self.history) / len(self.history)
        avg_mem_spare = sum(i[2] for i in self.history) / len(self.history)

        table = Table(
            header_style="bold #F5F5DC",
            title_justify="center",
            show_header=True,
            show_lines=True
        )
        table.add_column("CPU Usage", justify="left", width=18)
        table.add_column("Memory Usage", justify="left", width=18)
        table.add_column("Memory Available", justify="left", width=18)
        information = [
            f"[bold #D2B48C]{avg_cpu_usage:.2f} %",
            f"[bold #D2B48C]{avg_mem_usage:.2f} %",
            f"[bold #D2B48C]{avg_mem_spare:.2f} G",
        ]
        table.add_row(*information)

        if (avg_cpu_usage <= self.cpu_usage_threshold and avg_mem_usage <= self.mem_usage_threshold
                and avg_mem_spare >= self.mem_spare_threshold):
            self.history.clear()
            table.title = f"[bold #54FF9F]**<* {const.DESC} Performance Success *>**"
            return table, "stable"

        if not first_examine:
            self.history.clear()
        table.title = f"[bold #FFEC8B]**<* {const.DESC} Performance Warning *>**"
        return table, "unstable"


class AsyncAnimationManager(object):
    """
    一个异步动画任务管理器，用于统一控制 CLI 动画的启动与停止。

    该类支持仅运行一个动画任务，若有新任务启动则自动取消当前任务。
    使用 asyncio.Event 控制动画函数的终止时机，使其适应非阻塞的异步 CLI 环境。

    Attributes
    ----------
    __task : asyncio.Task | None
        当前正在运行的动画任务（协程）。

    __animation_event : asyncio.Event
        控制动画任务停止的事件对象，供动画函数内部监听。
    """

    def __init__(self):
        self.__task: asyncio.Task | None = None
        self.__animation_event: asyncio.Event = asyncio.Event()

    async def start(self, function: typing.Callable) -> typing.Coroutine | None:
        """
        启动一个异步动画函数（必须是 async def），若已有动画在运行会先取消。

        Parameters
        ----------
        function : Callable[[asyncio.Event], Awaitable]
            异步动画函数，需接受一个 asyncio.Event 参数以控制动画终止。

        Returns
        -------
        Coroutine | None
            启动新的动画任务。若有已有动画在运行，则先取消。

        Notes
        -----
        - 调用该方法前，现有动画任务将通过 stop() 停止。
        - 传入的 function 需在内部周期性检测 `event.is_set()` 以判断是否终止。
        """
        await self.stop()  # 若已有动画在运行，先取消

        self.__animation_event.clear()

        self.__task = asyncio.create_task(
            function(self.__animation_event)
        )

    async def stop(self) -> typing.Coroutine | None:
        """
        停止当前动画任务（如存在），通过设置事件并 cancel 协程任务。

        Returns
        -------
        Coroutine | None
            协程任务被取消或无任务可取消时返回 None。

        Notes
        -----
        - 若任务未完成，将尝试 cancel 并等待其正常退出。
        - 动画函数应在检测到事件触发后自行退出，以避免阻塞。
        """
        if self.__task and not self.__task.done():
            self.__animation_event.set()

            self.__task.cancel()
            try:
                await self.__task
            except asyncio.CancelledError:
                logger.debug(f"Animation cancelled")

        self.__task = None


class Manage(object):
    """
    设备管理器类。

    该类负责通过 ADB 接口发现并管理 Android 设备，支持获取设备的硬件信息（如品牌、版本、CPU、内存、分辨率等），
    并提供用户选择和切换设备的交互方式。适用于多设备调试和控制场景，支持异步批量设备信息收集与展示。
    """

    device_dict: dict[str, "Device"] = {}

    def __init__(self, adb: str):
        self.adb = adb

    async def current_device(self) -> dict[str, "Device"]:

        async def _device_cpu(sn: str) -> typing.Any:
            cmd = [
                self.adb, "-s", sn, "wait-for-device", "shell", "cat", "/proc/cpuinfo", "|", "grep", "processor"
            ]
            return len(re.findall(r"processor", cpu, re.S)) if (cpu := await Terminal.cmd_line(cmd)) else None

        async def _device_ram(sn: str) -> typing.Any:
            cmd = [
                self.adb, "-s", sn, "wait-for-device", "shell", "free"
            ]
            if ram := await Terminal.cmd_line(cmd):
                for line in ram.splitlines()[1:2]:
                    if match := re.search(r"\d+", line.split()[1]):
                        total_ram = int(match.group()) / 1024 / 1024 / 1024
                        return math.ceil(total_ram)
            return None

        async def _device_tag(sn: str) -> typing.Any:
            cmd = [
                self.adb, "-s", sn, "wait-for-device", "shell", "getprop", "ro.product.brand"
            ]
            return tag if (tag := await Terminal.cmd_line(cmd)) else None

        async def _device_ver(sn: str) -> typing.Any:
            cmd = [
                self.adb, "-s", sn, "wait-for-device", "shell", "getprop", "ro.build.version.release"
            ]
            return ver if (ver := await Terminal.cmd_line(cmd)) else None

        async def _device_display(sn: str) -> dict:
            cmd = [
                self.adb, "-s", sn, "wait-for-device", "shell", "dumpsys", "display", "|", "grep", "mViewports="
            ]
            screen_dict = {}
            if information_list := await Terminal.cmd_line(cmd):
                if display_list := re.findall(r"DisplayViewport\{.*?}", information_list):
                    fit: typing.Any = lambda x: re.search(x, display)
                    for display in display_list:
                        if all((
                                i := fit(r"(?<=displayId=)\d+"),
                                w := fit(r"(?<=deviceWidth=)\d+"),
                                h := fit(r"(?<=deviceHeight=)\d+"))):
                            screen_dict.update({int(i.group()): (int(w.group()), int(h.group()))})
            return screen_dict

        async def _device_information(sn: str) -> "Device":
            information_list = await asyncio.gather(
                _device_tag(sn), _device_ver(sn), _device_cpu(sn), _device_ram(sn), _device_display(sn)
            )
            return Device(self.adb, sn, *information_list)

        device_dict = {}
        if device_list := await Terminal.cmd_line([self.adb, "devices"]):
            if serial_list := [line.split()[0] for line in device_list.split("\n")[1:]]:
                device_instance_list = await asyncio.gather(
                    *(_device_information(serial) for serial in serial_list), return_exceptions=True
                )
                for device_instance in device_instance_list:
                    if isinstance(device_instance, Exception):
                        return device_dict
                device_dict = {device.sn: device for device in device_instance_list}
        return device_dict

    async def operate_device(self) -> typing.Optional[list["Device"]]:
        while True:
            if len(current_device_dict := await self.current_device()) == 0:
                Design.simulation_progress(f"Wait for device to connect ...")
                continue

            self.device_dict = current_device_dict
            return list(self.device_dict.values())

    async def another_device(self) -> typing.Optional[list["Device"]]:
        while True:
            if len(current_device_dict := await self.current_device()) == 0:
                Design.simulation_progress(f"Wait for device to connect ...")
                continue

            self.device_dict = current_device_dict

            if len(self.device_dict) == 1:
                return list(self.device_dict.values())

            for index, device in enumerate(self.device_dict.values()):
                Design.notes(f"[bold][bold #FFFACD]Connect:[/] [{index + 1:02}] {device}[/]")

            if (action := Prompt.ask(
                    "[bold #FFEC8B]请输入序列号选择一台设备[/]", console=Design.console, default="00")) == "00":
                return list(self.device_dict.values())

            try:
                choose_device = self.device_dict[action]
            except KeyError as e:
                Design.notes(f"{const.ERR}序列号不存在 -> {e}[/]\n")
                await asyncio.sleep(1)
                continue

            self.device_dict = {action: choose_device}
            return list(self.device_dict.values())

    async def display_device(self) -> None:
        Design.console.print(f"[bold]<Link> <{'单设备模式' if len(self.device_dict) == 1 else '多设备模式'}>[/]")
        for device in self.device_dict.values():
            Design.console.print(f"[bold #00FFAF]Connect:[/] [bold]{device}[/]")

    @staticmethod
    async def display_select(device_list: list["Device"]) -> None:
        select_dict = {
            device.sn: device for device in device_list if len(device.display) > 1
        }

        if len(select_dict) == 0:
            return Design.console.print(f"{const.WRN}没有多屏幕的设备[/]\n")

        table = Table(
            title=f"[bold #FF851B]{const.ITEM} {const.DESC} Select Command Line",
            header_style="bold #F5F5DC",
            title_justify="center",
            show_header=True,
            show_lines=True
        )
        table.add_column("Serial", justify="left", width=22)
        table.add_column("Screen", justify="left", width=8)
        table.add_column("Density", justify="left", width=14)
        table.add_column("Options", justify="left", width=24)

        choices = []
        for device in select_dict.values():
            for index, display in device.display.items():
                choices.append(f"{device.sn};{index}")
                info = [
                    f"[bold #FFAFAF]{device.sn}[/]",
                    f"[bold #AFD7FF]{index}[/]",
                    f"[bold #AFD7FF]{list(display)}[/]",
                    f"[bold #FFF68F]{device.sn}[bold #FF3030];[/]{index}[/]"
                ]
                table.add_row(*info)
        Design.console.print(table)

        action = Prompt.ask("[bold #FFEC8B]Select Display[/]", console=Design.console, choices=choices)
        sn, display_id = re.split(r";", action, re.S)
        select_dict[sn].id, screen = int(display_id), select_dict[sn].display[int(display_id)]
        Design.notes(f"{const.SUC}{sn} -> ID=[{display_id}] DISPLAY={list(screen)}")


if __name__ == '__main__':
    pass
