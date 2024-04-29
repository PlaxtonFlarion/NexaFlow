import re
import math
import typing
import asyncio
from loguru import logger
from rich.prompt import Prompt
from engine.device import Device
from engine.terminal import Terminal
from frameflow.skills.show import Show


class Manage(object):

    device_list: list["Device"] = []

    def __init__(self, adb: str):
        self.adb = adb

    async def current_device(self) -> dict[str, "Device"]:

        async def _device_cpu(serial):
            cmd = [self.adb, "-s", serial, "wait-for-usb-device", "shell", "cat", "/proc/cpuinfo", "|", "grep", "processor"]
            if cpu := await Terminal.cmd_line(*cmd):
                return len(re.findall(r"processor", cpu, re.S))

        async def _device_ram(serial):
            cmd = ["adb", "-s", serial, "wait-for-usb-device", "shell", "free"]
            if ram := await Terminal.cmd_line(*cmd):
                for line in ram.splitlines()[1:2]:
                    if match := re.search(r"\d+", line.split()[1]):
                        total_ram = int(match.group()) / 1024 / 1024 / 1024
                        return math.ceil(total_ram)

        async def _device_species(serial):
            cmd = [self.adb, "-s", serial, "wait-for-usb-device", "shell", "getprop", "ro.product.brand"]
            return await Terminal.cmd_line(*cmd)

        async def _device_version(serial):
            cmd = [self.adb, "-s", serial, "wait-for-usb-device", "shell", "getprop", "ro.build.version.release"]
            return await Terminal.cmd_line(*cmd)

        async def _device_display(serial):
            cmd = [self.adb, "-s", serial, "wait-for-usb-device", "shell", "dumpsys", "display", "|", "grep", "mViewports="]
            screen_dict = {}
            if information_list := await Terminal.cmd_line(*cmd):
                if display_list := re.findall(r"DisplayViewport\{.*?}", information_list):
                    fit: typing.Any = lambda x: re.search(x, display)
                    for display in display_list:
                        if all((i := fit(r"(?<=displayId=)\d+"), w := fit(r"(?<=deviceWidth=)\d+"), h := fit(r"(?<=deviceHeight=)\d+"))):
                            screen_dict.update({int(i.group()): (int(w.group()), int(h.group()))})
            return screen_dict

        async def _device_information(serial):
            information = await asyncio.gather(
                _device_species(serial), _device_version(serial), _device_cpu(serial), _device_ram(serial), _device_display(serial),
                return_exceptions=True
            )
            for info in information:
                if isinstance(info, Exception):
                    return info
            return Device(self.adb, serial, *information)

        device_dict = {}
        devices = await Terminal.cmd_line(self.adb, "devices")
        if serial_list := [i.split()[0] for i in devices.split("\n")[1:]]:
            result_list = await asyncio.gather(
                *(_device_information(serial) for serial in serial_list), return_exceptions=True
            )

            for result in result_list:
                if isinstance(result, Exception):
                    return device_dict

            device_dict = {str(i + 1): device for i, device in enumerate(result_list)}
        return device_dict

    async def operate_device(self) -> list["Device"]:
        while True:
            self.device_list = []
            if len(device_dict := await self.current_device()) == 0:
                Show.console.print(f"[bold #FFFACD]设备未连接,等待设备连接[/] ...")
                await asyncio.sleep(5)
                continue

            for k, v in device_dict.items():
                self.device_list.append(v)

            if len(self.device_list) == 1:
                return self.device_list

            for k, v in device_dict.items():
                Show.console.print(f"[bold][bold #FFFACD]Connect:[/] [{k}] {v}[/]")

            return self.device_list

    async def another_device(self) -> list["Device"]:
        while True:
            self.device_list = []
            if len(device_dict := await self.current_device()) == 0:
                Show.console.print(f"[bold #FFFACD]设备未连接,等待设备连接[/] ...")
                await asyncio.sleep(5)
                continue

            for k, v in device_dict.items():
                self.device_list.append(v)

            if len(self.device_list) == 1:
                return self.device_list

            for k, v in device_dict.items():
                Show.console.print(f"[bold][bold #FFFACD]Connect:[/] [{k}] {v}[/]")

            try:
                if (action := Prompt.ask(
                        "[bold #FFEC8B]请输入编号选择一台设备[/]", console=Show.console, default="all")) == "all":
                    return self.device_list
                return [device_dict[action]]
            except KeyError:
                Show.console.print(f"[bold #FFC0CB]没有该序号,请重新选择[/] ...\n")
                await asyncio.sleep(1)

    async def display_device(self) -> None:
        logger.info(f"<Link> <{'单设备模式' if len(self.device_list) == 1 else '多设备模式'}>")
        for device in self.device_list:
            logger.info(f"[bold #00FFAF]Connect:[/] {device}")

    async def display_select(self):
        while True:
            self.device_list = []
            if len(device_dict := await self.current_device()) == 0:
                Show.console.print(f"[bold #FFFACD]设备未连接,等待设备连接[/] ...")
                await asyncio.sleep(5)
                continue

            for k, v in device_dict.items():
                self.device_list.append(v)

            select_dict = {
                device.serial: device
                for device in self.device_list if len(device.display) > 1
            }

            if len(select_dict) == 0:
                for k, v in device_dict.items():
                    Show.console.print(f"[bold][bold #FFFACD]Connect:[/] [{k}] {v}[/]")
                return Show.console.print(f"[bold #FFC0CB]没有多屏幕的设备[/] ...")

            for k, v in select_dict.items():
                Show.console.print(f"{k} {v}")

            choices = [
                f"{sn};{idx}" for sn, device in select_dict.items()
                for idx in device.display.keys()
            ]
            try:
                if action := Prompt.ask("[bold #FFEC8B]Select[/]", console=Show.console, choices=choices):
                    if len(select := re.split(r";", action, re.S)) == 2:
                        select_dict[select[0]].id = int(select[1])
            except KeyError:
                Show.console.print(f"[bold #FFC0CB]没有该序号,请重新选择[/] ...\n")
                await asyncio.sleep(1)


if __name__ == '__main__':
    pass
