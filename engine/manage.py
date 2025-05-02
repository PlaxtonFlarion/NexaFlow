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

import os
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
    def screen_size(index: int) -> tuple[int, int, int, int]:
        """
        Screen Size 获取指定索引的屏幕尺寸信息。

        Parameters
        ----------
        index : typing.Union[str, int]
            屏幕索引号，用于在多显示器环境中指定目标屏幕。

        Returns
        -------
        tuple[int, int, int, int]
            返回屏幕的位置信息和尺寸：(x坐标, y坐标, 宽度, 高度)。

        Notes
        -----
        - 调用 `screeninfo` 库获取本地屏幕列表。
        - 如果索引超出可用范围，自动回退到第一个屏幕，保证函数稳定性。

        Workflow
        --------
        1. 根据传入的索引查找对应屏幕。
        2. 捕获索引错误并回退到默认屏幕。
        3. 返回屏幕的位置信息和尺寸。
        """
        try:
            screen = get_monitors()[index]
        except IndexError:
            screen = get_monitors()[0]

        return screen.x, screen.y, screen.width, screen.height


class SourceMonitor(object):
    """
    系统资源监测器。

    用于实时采样CPU、内存和磁盘使用率，判断系统是否处于稳定状态，从而决定是否可以安全地进行后续高负载操作。

    Class Attributes
    ----------------
    __message : dict
        内部共享的信息字典，包含默认提示文本、状态记录和基础资源阈值设置。

    Instance Attributes
    -------------------
    cpu_threshold : int
        CPU使用率的稳定阈值，超过此值判定为系统繁忙。

    mem_threshold : int
        内存使用率的稳定阈值，超过此值判定为系统繁忙。

    usages : dict
        当前系统资源使用情况，包括"cpu"、"mem"和"disk"占用率。

    afresh : int
        当系统不稳定时，重新检测前等待的秒数。

    __stable : bool
        系统是否稳定的标志位。

    Notes
    -----
    - 自动根据CPU核心数和总内存调整合理的阈值，适配不同硬件环境。
    - 检测流程采用两阶段：快速预检+稳定性深度采样，确保准确可靠。

    Workflow
    --------
    1. 快速采样初步检测系统负载状态；
    2. 若初步检测不通过，则进行深度稳定性采样；
    3. 在多轮采样后确认系统是否满足运行条件。
    """
    __message: dict[typing.Any, typing.Any] = {
        "msg": ["...", 0],
        "status": [],
        "base_cpu_threshold": 60,
        "base_mem_threshold": 80,
    }

    def __init__(self):
        """
        初始化SourceMonitor对象，基于系统硬件资源智能调整阈值。

        - CPU阈值：基础值 + 核心数补偿（最多增加40）
        - 内存阈值：基础值 - 总内存量补偿（每20GB减少1%）
        """
        cpu_cores = psutil.cpu_count()
        mem_total = psutil.virtual_memory().total / (1024 ** 3)

        self.cpu_threshold = self.message["base_cpu_threshold"] + min(40, int(cpu_cores * 1.5))
        self.mem_threshold = max(50, self.message["base_mem_threshold"] - mem_total / 20)

        self.usages = {}  # 当前资源使用情况
        self.afresh = 10  # 失败后重试等待秒数

        self.__stable = False  # 初始默认为不稳定状态

    @property
    def message(self) -> dict[typing.Any, typing.Any]:
        """
        获取当前内部状态信息。

        Returns
        -------
        dict
            包含当前提示信息、状态列表、基础CPU和内存阈值等。
        """
        return self.__message

    @message.setter
    def message(self, value: typing.Any):
        """
        设置内部状态信息。

        Parameters
        ----------
        value : typing.Any
            新的状态信息字典。
        """
        self.__message = value

    @property
    def stable(self) -> bool:
        """
        获取当前系统稳定性标志。

        Returns
        -------
        bool
            系统是否稳定。
        """
        return self.__stable

    @stable.setter
    def stable(self, value: bool):
        """
        设置系统稳定性标志。

        Parameters
        ----------
        value : bool
            设定系统是否稳定，必须为布尔类型。

        Raises
        -------
        AssertionError
            当传入值不是布尔类型时抛出。
        """
        assert type(value) is bool, f"Stable must be a boolean value."
        self.__stable = value

    async def sample_average(self, duration: float, interval: float = 0.5) -> dict:
        """
        持续采样指定时长内的CPU、内存和磁盘使用率，并计算平均值。

        Parameters
        ----------
        duration : float
            采样总时长（秒）。

        interval : float, optional
            每次采样的间隔（秒），默认0.5秒。

        Returns
        -------
        dict
            包括"cpu"、"mem"、"dsk"的平均使用率百分比。
        """
        cpu_samples, mem_samples, dsk_samples = [], [], []

        elapsed = 0.0

        while elapsed < duration:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            dsk = psutil.disk_usage(os.sep).percent

            self.usages = {
                **{"cpu": cpu, "mem": mem, "dsk": dsk}
            }

            cpu_samples.append(cpu)
            mem_samples.append(mem)
            dsk_samples.append(dsk)

            await asyncio.sleep(interval)
            elapsed += interval
            self.message["msg"] = [f"[bold #00D7FF]Checking {float(duration - elapsed):.1f} s[/]", 1]

        return {
            "cpu": sum(cpu_samples) / len(cpu_samples),
            "mem": sum(mem_samples) / len(mem_samples),
            "dsk": sum(dsk_samples) / len(dsk_samples),
        }

    async def system_stable(self, avg_dict: dict) -> bool:
        """
        判断当前资源使用率是否低于阈值，认为系统是否稳定。

        Parameters
        ----------
        avg_dict : dict
            包含当前平均CPU、内存和磁盘使用率。

        Returns
        -------
        bool
            如果所有资源均低于各自阈值则返回True，否则返回False。
        """
        return all((avg_dict["cpu"] < self.cpu_threshold, avg_dict["mem"] < self.mem_threshold))

    async def monitor_stable(self) -> None:
        """
        持续监控系统状态，直到系统资源使用率低于阈值。

        Workflow
        --------
        - 首先快速采样一次初步判断。
        - 若不稳定，则循环采样，并在每轮采样失败后等待一段时间。
        - 成功采样并符合稳定条件后，更新稳定状态标志。
        """
        begin_avg = await self.sample_average(duration=2.0, interval=0.2)
        if await self.system_stable(begin_avg):
            self.message["msg"] = ["[bold #87FFAF]Stable[/]", 1]
            await asyncio.sleep(0.5)
            self.stable = True
            return

        while not self.stable:
            self.message["msg"] = ["[bold #FFAFAF]Unstable[/]", 1]
            again_avg = await self.sample_average(duration=10.0, interval=1.0)
            if await self.system_stable(again_avg):
                self.stable = True
                break

            for i in range(self.afresh):
                self.message["msg"] = [f"[bold 	#FFD700]Retry after {float(self.afresh - i):.1f} s[/]", 0]
                await asyncio.sleep(1)
            self.message["msg"] = [f"[bold #A8A8A8]...[/]", 1]
            await asyncio.sleep(0.5)


class AsyncAnimationManager(object):
    """
    管理异步动画任务的上下文工具类。

    该类用于封装动画生命周期的启动与终止逻辑，支持通过 `async with` 语句自动管理动画执行流程。

    Parameters
    ----------
    function : Optional[Callable]
        接收一个 asyncio.Event 的异步动画函数，必须为 async def。

    Attributes
    ----------
    __task : Optional[asyncio.Task]
        当前运行的动画任务对象。

    __animation_event : Optional[asyncio.Event]
        控制动画终止的事件信号，用于优雅中断动画。

    __function : Optional[Callable]
        初始化传入的动画函数。
    """

    def __init__(self, function: typing.Optional["typing.Callable"] = None):
        self.__task: typing.Optional["asyncio.Task"] = None
        self.__animation_event: typing.Optional["asyncio.Event"] = asyncio.Event()
        self.__function = function

    async def __aenter__(self):
        """
        异步上下文进入方法，自动调用 start() 启动动画。
        """
        await self.start(self.__function)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        异步上下文退出方法，自动调用 stop() 停止动画。
        """
        await self.stop()

    async def start(self, function: "typing.Callable") -> typing.Optional["typing.Coroutine"]:
        """
         启动指定动画函数任务，若已有动画在运行将优雅取消。

         Parameters
         ----------
         function : Callable
             接收 asyncio.Event 参数的异步动画函数。
         """
        await self.stop()  # 若已有动画在运行，先取消

        self.__animation_event.clear()

        self.__task = asyncio.create_task(
            function(self.__animation_event)
        )

    async def stop(self) -> typing.Optional["typing.Coroutine"]:
        """
        停止当前动画任务（如存在），设置终止事件并取消任务对象。

        Notes
        -----
        - 若无任务正在运行则跳过处理。
        - 使用 asyncio.CancelledError 捕获取消异常。
        """
        if self.__task and not self.__task.done():
            self.__animation_event.set()

            self.__task.cancel()
            try:
                await self.__task
            except asyncio.CancelledError:
                logger.debug(f"Animation cancelled ...")

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
                Design.console.print(f"[bold][bold #FFFACD]Connect:[/] [{index + 1:02}] {device}[/]")

            if (action := Prompt.ask(
                    "[bold #FFEC8B]请输入序列号选择一台设备[/]", console=Design.console, default="00")) == "00":
                return list(self.device_dict.values())

            try:
                choose_device = self.device_dict[action]
            except KeyError as e:
                Design.console.print(f"{const.ERR}序列号不存在 -> {e}[/]\n")
                await asyncio.sleep(1)
                continue

            self.device_dict = {action: choose_device}
            return list(self.device_dict.values())

    async def display_device(self, ctrl: str) -> None:
        mode = "单设备模式" if len(self.device_dict) == 1 else "多设备模式"
        Design.console.print(
            f"[bold]<Link> <{mode}> **<*{ctrl}*>**[/]"
        )
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
        Design.console.print(f"{const.SUC}{sn} -> ID=[{display_id}] DISPLAY={list(screen)}")


if __name__ == '__main__':
    pass
