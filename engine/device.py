import re
import asyncio
from engine.terminal import Terminal


class _Phone(object):

    def __init__(self, serial: str, species: str, version: str, display: dict):
        self.serial, self.species, self.version, self.display = serial, species, version, display

    def __str__(self):
        return f"<Device {self.species} os={self.version} sn={self.serial} display={self.display}>"

    __repr__ = __str__


class Device(_Phone):

    def __init__(self, adb: str, serial: str, species: str, version: str, display: dict):
        super().__init__(serial, species, version, display)
        self.initial = [adb, "-s", serial, "wait-for-usb-device"]

    @staticmethod
    async def ask_sleep(secs: float) -> None:
        await asyncio.sleep(secs)

    async def ask_tap(self, x: int, y: int) -> None:
        cmd = self.initial + ["shell", "input", "tap", str(x), str(y)]
        await Terminal.cmd_line(*cmd)

    async def ask_is_screen_on(self) -> str:
        cmd = self.initial + ["shell", "dumpsys", "deviceidle", "|", "grep", "mScreenOn"]
        result = await Terminal.cmd_line(*cmd)
        return result.split("=")[-1].strip()

    async def ask_swipe_unlock(self) -> None:
        if await self.ask_is_screen_on() == "false":
            await self.ask_key_event(26)
            await asyncio.sleep(1)
            cmd = self.initial + ["shell", "input", "touchscreen", "swipe", "250", "650", "250", "50"]
            await Terminal.cmd_line(*cmd)
            await asyncio.sleep(1)

    async def ask_key_event(self, key_code: int) -> None:
        cmd = self.initial + ["shell", "input", "keyevent", str(key_code)]
        await Terminal.cmd_line(*cmd)

    async def ask_force_stop(self, package: str) -> None:
        cmd = self.initial + ["shell", "am", "force-stop", package]
        await Terminal.cmd_line(*cmd)

    async def ask_wifi(self, power: str) -> None:
        cmd = self.initial + ["shell", "svc", "wifi", power]
        await Terminal.cmd_line(*cmd)

    async def ask_all_package(self, level: int) -> list[str]:
        cmd = self.initial + ["shell", "ps"]
        result = await Terminal.cmd_line(*cmd)
        package_list = []
        for line in result.splitlines():
            uid = parts[1] if len(parts := line.split()) > 1 else None
            if uid and uid.isdigit() and int(uid) >= level:
                package_list.append(parts[-1])
        return package_list

    async def ask_current_activity(self) -> str:
        cmd = self.initial + ["shell", "dumpsys", "window", "|", "grep", "mCurrentFocus"]
        result = await Terminal.cmd_line(*cmd)
        match = re.search(r"(?<=Window\{).*?(?=})", result)
        return match.group().split()[-1].strip() if match else ""

    async def ask_start_app(self, package: str) -> str:
        cmd = self.initial + ["shell", "am", "start", "-n", package]
        result = await Terminal.cmd_line(*cmd)
        return result.strip()

    async def ask_screen_size(self) -> str:
        cmd = self.initial + ["shell", "wm", "size"]
        result = await Terminal.cmd_line(*cmd)
        return result.strip()

    async def ask_screen_density(self) -> str:
        cmd = self.initial + ["shell", "wm", "density"]
        result = await Terminal.cmd_line(*cmd)
        return result.strip()

    async def ask_force_filter(self, package: str) -> None:
        current_screen = await self.ask_current_activity()
        if package not in current_screen:
            screen = current_screen.split("/")[0] if "/" in current_screen else current_screen
            await self.ask_force_stop(screen)


if __name__ == '__main__':
    pass