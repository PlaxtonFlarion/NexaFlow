import re
import asyncio
from rich.prompt import Prompt
from frameflow.show import Show
from nexaflow.terminal import Terminal


class Manage(object):

    def __init__(self, adb: str):
        self.__adb = adb

    async def current_device(self) -> dict[str, "Device"]:

        async def check(serial: str) -> "Device":
            brand, version = await asyncio.gather(
                Terminal.cmd_line(self.__adb, "-s", serial, "wait-for-usb-device", "shell", "getprop", "ro.product.brand"),
                Terminal.cmd_line(self.__adb, "-s", serial, "wait-for-usb-device", "shell", "getprop", "ro.build.version.release")
            )
            return Device(self.__adb, serial, brand, version)

        device_dict = {}
        devices = await Terminal.cmd_line(self.__adb, "devices")
        serial_list = [i.split()[0] for i in devices.split("\n")[1:]]
        if len(serial_list) > 0:
            tasks = [check(serial) for serial in serial_list]
            result = await asyncio.gather(*tasks)
            device_dict = {str(idx + 1): c for idx, c in enumerate(result)}
        return device_dict

    async def operate_device(self) -> list["Device"]:
        final = []
        while True:
            device_dict: dict[str, "Device"] = await self.current_device()
            if len(device_dict) > 0:
                for k, v in device_dict.items():
                    final.append(v)
                    Show.console.print(f"[bold][bold yellow]已连接设备[/bold yellow] [{k}] {v}")

                if len(device_dict) > 1:
                    try:
                        action = Prompt.ask("[bold #5FD7FF]请输入编号选择一台设备")
                        final = final if action == "000" else [device_dict[action]]
                    except KeyError:
                        Show.console.print(f"[bold red]没有该序号,请重新选择 ...[/bold red]\n")
                        continue

                if len(final) == 1:
                    Show.console.print(f"[bold]<Link> <单设备模式>")
                else:
                    Show.console.print(f"[bold]<Link> <多设备模式>")

                return final

            else:
                Show.console.print(f"[bold yellow]设备未连接,等待设备连接 ...")
                await asyncio.sleep(5)

    # async def operate_device(self):
    #     while True:
    #         device_dict: dict[str, "Device"] = await self.current_device()
    #         if len(device_dict) > 0:
    #             for k, v in device_dict.items():
    #                 Show.console.print(f"[bold][bold yellow]已连接设备[/bold yellow] [{k}] {v}")
    #             if len(device_dict) == 1:
    #                 return device_dict["1"]
    #
    #             try:
    #                 action = Prompt.ask("[bold #5FD7FF]请输入编号选择一台设备")
    #                 return device_dict[action]
    #             except KeyError:
    #                 Show.console.print(f"[bold red]没有该序号,请重新选择 ...")
    #
    #         else:
    #             Show.console.print(f"[bold yellow]设备未连接,等待设备连接 ...")
    #             await asyncio.sleep(5)


class Phones(object):

    def __init__(self, serial: str, brand: str, version: str):
        self.serial, self.brand, self.version = serial, brand, version

    def __str__(self):
        return f"<Phone brand={self.brand} version=OS{self.version} serial={self.serial}>"

    __repr__ = __str__


class Device(Phones):

    def __init__(self, adb: str, serial: str, brand: str, version: str):
        super().__init__(serial, brand, version)
        self.__initial = [adb, "-s", serial, "wait-for-usb-device"]

    @staticmethod
    async def ask_sleep(secs: float) -> None:
        await asyncio.sleep(secs)

    async def ask_tap(self, x: int, y: int) -> None:
        cmd = self.__initial + ["shell", "input", "tap", str(x), str(y)]
        await Terminal.cmd_line(*cmd)

    async def ask_is_screen_on(self) -> str:
        cmd = self.__initial + ["shell", "dumpsys", "deviceidle", "|", "grep", "mScreenOn"]
        result = await Terminal.cmd_line(*cmd)
        return result.split("=")[-1].strip()

    async def ask_swipe_unlock(self) -> None:
        screen = await self.ask_is_screen_on()
        if screen == "false":
            await self.ask_key_event(26)
            await asyncio.sleep(1)
            cmd = self.__initial + ["shell", "input", "touchscreen", "swipe", "250", "650", "250", "50"]
            await Terminal.cmd_line(*cmd)
            await asyncio.sleep(1)

    async def ask_key_event(self, key_code: int) -> None:
        cmd = self.__initial + ["shell", "input", "keyevent", str(key_code)]
        await Terminal.cmd_line(*cmd)

    async def ask_force_stop(self, package: str) -> None:
        cmd = self.__initial + ["shell", "am", "force-stop", package]
        await Terminal.cmd_line(*cmd)

    async def ask_wifi(self, switch: str = "enable") -> None:
        cmd = self.__initial + ["shell", "svc", "wifi", switch]
        await Terminal.cmd_line(*cmd)

    async def ask_all_package(self, level: int = 10000) -> list[str]:
        cmd = self.__initial + ["shell", "ps"]
        result = await Terminal.cmd_line(*cmd)
        package_list = []
        for line in result.splitlines():
            parts = line.split()
            uid = parts[1] if len(parts) > 1 else None
            if uid and uid.isdigit() and int(uid) >= level:
                pkg_name = parts[-1]
                package_list.append(pkg_name)
        return package_list

    async def ask_current_activity(self) -> str:
        cmd = self.__initial + ["shell", "dumpsys", "window", "|", "grep", "mCurrentFocus"]
        result = await Terminal.cmd_line(*cmd)
        match = re.search(r"(?<=Window\{).*?(?=})", result)
        return match.group().split()[-1].strip() if match else ""

    async def ask_start_app(self, package: str = "com.android.settings/com.android.settings.Settings") -> str:
        cmd = self.__initial + ["shell", "am", "start", "-n", package]
        result = await Terminal.cmd_line(*cmd)
        return result.strip()

    async def ask_screen_size(self) -> str:
        cmd = self.__initial + ["shell", "wm", "size"]
        result = await Terminal.cmd_line(*cmd)
        return result.strip()

    async def ask_screen_density(self) -> str:
        cmd = self.__initial + ["shell", "wm", "density"]
        result = await Terminal.cmd_line(*cmd)
        return result.strip()

    async def ask_force_filter(self, package: str) -> None:
        current_screen = await self.ask_current_activity()
        if package not in current_screen:
            screen = current_screen.split("/")[0] if "/" in current_screen else current_screen
            await self.ask_force_stop(screen)


if __name__ == '__main__':
    pass
