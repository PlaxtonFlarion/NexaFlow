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
from frameflow.skills.show import Show
from nexaflow import const


class ScreenMonitor(object):

    @staticmethod
    def screen_size():
        screen = get_monitors()[0]
        return screen.width, screen.height


class SourceMonitor(object):

    history = []

    def __init__(self):
        base_cpu_usage_threshold = 50
        base_mem_usage_threshold = 70

        self.cpu_cores = psutil.cpu_count()
        self.mem_total = psutil.virtual_memory().total / (1024 ** 3)

        self.cpu_usage_threshold = base_cpu_usage_threshold + min(85, int(self.cpu_cores * 0.5))
        self.mem_usage_threshold = max(50, base_mem_usage_threshold - self.mem_total / 10)
        self.mem_spare_threshold = max(1, self.mem_total * 0.2)

    async def monitor(self):
        first_examine = True
        progress = Show.show_progress()
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
                    return Show.console.print(table)
                first_examine = False

            elif len(self.history) >= 5:
                table, explain = await self.evaluate_resources(False)
                if explain == "stable":
                    progress.update(task)
                    progress.stop()
                    return Show.console.print(table)

                progress.stop()
                Show.console.print(table)
                progress = Show.show_progress()
                task = progress.add_task(description=f"Analyzer", total=5)
                progress.start()

            await asyncio.sleep(2)

    async def evaluate_resources(self, first_examine: bool):

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
            table.title = f"[bold #54FF9F]〇 {const.DESC} Performance Success 〇"
            return table, "stable"

        if not first_examine:
            self.history.clear()
        table.title = f"[bold #FFEC8B]〇 {const.DESC} Performance Warning 〇"
        return table, "unstable"


class Manage(object):

    device_list: list["Device"] = []

    def __init__(self, adb: str):
        self.adb = adb

    async def current_device(self) -> dict[str, "Device"]:

        async def _device_cpu(sn):
            cmd = [
                self.adb, "-s", sn, "wait-for-usb-device", "shell",
                "cat", "/proc/cpuinfo", "|", "grep", "processor"
            ]
            if cpu := await Terminal.cmd_line(*cmd):
                return len(re.findall(r"processor", cpu, re.S))

        async def _device_ram(sn):
            cmd = ["adb", "-s", sn, "wait-for-usb-device", "shell", "free"]
            if ram := await Terminal.cmd_line(*cmd):
                for line in ram.splitlines()[1:2]:
                    if match := re.search(r"\d+", line.split()[1]):
                        total_ram = int(match.group()) / 1024 / 1024 / 1024
                        return math.ceil(total_ram)

        async def _device_tag(sn):
            cmd = [
                self.adb, "-s", sn, "wait-for-usb-device", "shell",
                "getprop", "ro.product.brand"
            ]
            return await Terminal.cmd_line(*cmd)

        async def _device_ver(sn):
            cmd = [
                self.adb, "-s", sn, "wait-for-usb-device", "shell",
                "getprop", "ro.build.version.release"
            ]
            return await Terminal.cmd_line(*cmd)

        async def _device_display(sn):
            cmd = [
                self.adb, "-s", sn, "wait-for-usb-device", "shell",
                "dumpsys", "display", "|", "grep", "mViewports="
            ]
            screen_dict = {}
            if information_list := await Terminal.cmd_line(*cmd):
                if display_list := re.findall(r"DisplayViewport\{.*?}", information_list):
                    fit: typing.Any = lambda x: re.search(x, display)
                    for display in display_list:
                        if all((
                                i := fit(r"(?<=displayId=)\d+"),
                                w := fit(r"(?<=deviceWidth=)\d+"),
                                h := fit(r"(?<=deviceHeight=)\d+"))):
                            screen_dict.update({int(i.group()): (int(w.group()), int(h.group()))})
            return screen_dict

        async def _device_information(sn):
            information_list = await asyncio.gather(
                _device_tag(sn), _device_ver(sn), _device_cpu(sn), _device_ram(sn),
                _device_display(sn), return_exceptions=True
            )
            for device_info in information_list:
                if isinstance(device_info, Exception):
                    return device_info
            return Device(self.adb, sn, *information_list)

        device_dict = {}
        device_list = await Terminal.cmd_line(self.adb, "devices")
        if sn_list := [line.split()[0] for line in device_list.split("\n")[1:]]:
            device_instance_list = await asyncio.gather(
                *(_device_information(sn) for sn in sn_list), return_exceptions=True
            )
            for device_instance in device_instance_list:
                if isinstance(device_instance, Exception):
                    return device_dict
            device_dict = {device.sn: device for device in device_instance_list}
        return device_dict

    async def operate_device(self) -> list["Device"]:
        while True:
            self.device_list = []
            if len(device_dict := await self.current_device()) == 0:
                Show.simulation_progress(f"Wait for device to connect ...", 1, 0.05)
                continue

            for device in device_dict.values():
                self.device_list.append(device)

            if len(self.device_list) == 1:
                return self.device_list

            for index, device in enumerate(device_dict.values()):
                Show.console.print(f"[bold][bold #FFFACD]Connect:[/] [{index + 1:02}] {device}[/]")

            return self.device_list

    async def another_device(self) -> list["Device"]:
        while True:
            self.device_list = []
            if len(device_dict := await self.current_device()) == 0:
                Show.simulation_progress(f"Wait for device to connect ...", 1, 0.05)
                continue

            for device in device_dict.values():
                self.device_list.append(device)

            if len(self.device_list) == 1:
                return self.device_list

            for index, device in enumerate(device_dict.values()):
                Show.console.print(f"[bold][bold #FFFACD]Connect:[/] [{index + 1:02}] {device}[/]")

            try:
                if (action := Prompt.ask(
                        "[bold #FFEC8B]请输入序列号选择一台设备[/]", console=Show.console, default="00")) == "00":
                    return self.device_list
                return [device_dict[action]]
            except KeyError as e:
                Show.console.print(f"{const.ERR}序列号不存在 -> {e}[/]\n")
                await asyncio.sleep(1)

    async def display_device(self) -> None:
        logger.info(f"<Link> <{'单设备模式' if len(self.device_list) == 1 else '多设备模式'}>")
        for device in self.device_list:
            logger.info(f"[bold #00FFAF]Connect:[/] {device}")

    async def display_select(self):
        while True:
            self.device_list = []
            if len(device_dict := await self.current_device()) == 0:
                Show.simulation_progress(f"Wait for device to connect ...", 1, 0.05)
                continue

            for device in device_dict.values():
                self.device_list.append(device)

            select_dict = {
                device.sn: device for device in self.device_list if len(device.display) > 1
            }

            if len(select_dict) == 0:
                for index, device in enumerate(device_dict.values()):
                    Show.console.print(f"[bold][bold #FFFACD]Connect:[/] [{index + 1:02}] {device}[/]")
                return Show.console.print(f"{const.WRN}没有多屏幕的设备[/]\n")

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
            Show.console.print(table)

            action = Prompt.ask("[bold #FFEC8B]Select Display[/]", console=Show.console, choices=choices)
            sn, display_id = re.split(r";", action, re.S)
            select_dict[sn].id, screen = int(display_id), select_dict[sn].display[int(display_id)]
            return logger.success(f"{const.SUC} {sn} -> ID=[{display_id}] DISPLAY={list(screen)}")


if __name__ == '__main__':
    pass
