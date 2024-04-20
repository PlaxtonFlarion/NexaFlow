import re
import asyncio
from rich.prompt import Prompt
from engine.device import Device
from engine.terminal import Terminal
from frameflow.skills.show import Show


class Manage(object):

    def __init__(self, adb: str):
        self.adb = adb

    async def current_device(self) -> dict[str, "Device"]:

        async def device_info(serial):
            cmd_initial = [self.adb, "-s", serial, "wait-for-usb-device", "shell"]
            information_list = await asyncio.gather(
                Terminal.cmd_line(*(cmd_initial + ["getprop", "ro.product.brand"])),
                Terminal.cmd_line(*(cmd_initial + ["getprop", "ro.build.version.release"])),
                Terminal.cmd_line(*(cmd_initial + ["wm", "size"])),
                return_exceptions=True
            )

            for information in information_list:
                if isinstance(information, Exception):
                    return information
            species, version, display = information_list

            mate = re.search(r"(?<=Physical size:\s)(\d+)x(\d+)", display)
            size = tuple(mate.group().split("x")) if mate else ()
            return Device(self.adb, serial, species, version, size)

        device_dict = {}
        devices = await Terminal.cmd_line(self.adb, "devices")
        if serial_list := [i.split()[0] for i in devices.split("\n")[1:]]:
            result_list = await asyncio.gather(
                *(device_info(serial) for serial in serial_list), return_exceptions=True
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
                Show.console.print(f"[bold yellow]设备未连接,等待设备连接 ...")
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
                Show.console.print(f"[bold red]没有该序号,请重新选择 ...[/bold red]\n")
                await asyncio.sleep(1)


if __name__ == '__main__':
    pass
