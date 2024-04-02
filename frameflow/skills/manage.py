import re
import asyncio
from rich.prompt import Prompt
from nexaflow.terminal import Terminal
from frameflow.skills.show import Show
from frameflow.skills.device import Device


class Manage(object):

    def __init__(self, adb: str):
        self.adb = adb

    async def current_device(self) -> dict[str, "Device"]:

        async def device_info(serial):
            cmd_initial = [self.adb, "-s", serial, "wait-for-usb-device", "shell"]
            brand, version, screen = await asyncio.gather(
                Terminal.cmd_line(*(cmd_initial + ["getprop", "ro.product.brand"])),
                Terminal.cmd_line(*(cmd_initial + ["getprop", "ro.build.version.release"])),
                Terminal.cmd_line(*(cmd_initial + ["wm", "size"]), )
            )
            mate = re.search(r"(?<=Physical size:\s)(\d+)x(\d+)", screen)
            size = tuple(mate.group().split("x")) if mate else ()
            return Device(self.adb, serial, brand, version, size)

        device_dict = {}
        devices = await Terminal.cmd_line(self.adb, "devices")
        if len(serial_list := [i.split()[0] for i in devices.split("\n")[1:]]) > 0:
            result = await asyncio.gather(
                *(device_info(serial) for serial in serial_list)
            )
            device_dict = {str(i + 1): device for i, device in enumerate(result)}
        return device_dict

    async def operate_device(self) -> list["Device"]:
        final = []
        while True:
            if len(device_dict := await self.current_device()) == 0:
                Show.console.print(f"[bold yellow]设备未连接,等待设备连接 ...")
                await asyncio.sleep(5)
                continue

            for k, v in device_dict.items():
                final.append(v)
                Show.console.print(f"[bold][bold yellow]Connect:[/bold yellow] [{k}] {v}")

            try:
                action = Prompt.ask("[bold #5FD7FF]请输入编号选择一台设备", console=Show.console)
                final = final if action == "000" else [device_dict[action]]
            except KeyError:
                final.clear()
                Show.console.print(f"[bold red]没有该序号,请重新选择 ...[/bold red]\n")
                continue

            return final


if __name__ == '__main__':
    pass
