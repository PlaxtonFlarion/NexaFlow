import re
import typing
import asyncio
from rich.prompt import Prompt
from engine.device import Device
from engine.terminal import Terminal
from frameflow.skills.show import Show


class Manage(object):

    def __init__(self, adb: str):
        self.adb = adb

    async def current_device(self) -> dict[str, "Device"]:

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
                            screen_dict.update({i.group(): (w.group(), h.group())})
            return screen_dict

        async def _device_information(serial):
            information = await asyncio.gather(
                _device_species(serial), _device_version(serial), _device_display(serial), return_exceptions=True
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
            device_list = []
            if len(device_dict := await self.current_device()) == 0:
                Show.console.print(f"[bold yellow]设备未连接,等待设备连接[/bold yellow] ...")
                await asyncio.sleep(5)
                continue

            for k, v in device_dict.items():
                device_list.append(v)

            if len(device_list) == 1:
                return device_list

            for k, v in device_dict.items():
                Show.console.print(f"[bold][bold yellow]Connect:[/bold yellow] [{k}] {v}")

            try:
                if (action := Prompt.ask("[bold #5FD7FF]请输入编号选择一台设备", console=Show.console)) == "000":
                    return device_list
                return [device_dict[action]]
            except KeyError:
                Show.console.print(f"[bold red]没有该序号,请重新选择[/bold red] ...\n")
                await asyncio.sleep(1)


if __name__ == '__main__':
    pass