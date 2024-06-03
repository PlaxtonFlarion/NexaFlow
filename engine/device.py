import re
import asyncio
from urllib.parse import quote
from engine.terminal import Terminal


class _Phone(object):

    def __init__(self, sn: str, *args):
        self.sn = sn
        self.tag, self.ver, self.cpu, self.ram, self.display, *_ = args
        self.id = list(self.display.keys())[0]

    def __str__(self):
        head = f"<Device {self.tag} SN={self.sn} OS=[{self.ver}]"
        tail = f"CPU=[{self.cpu}] RAM=[{self.ram}] ID=[{self.id}] DISPLAY={self.display}>"
        return head + " " + tail

    __repr__ = __str__


class Device(_Phone):

    def __init__(self, adb: str, sn: str, *args):
        super().__init__(sn, *args)
        self.initial = [adb, "-s", sn, "wait-for-device"]

    @staticmethod
    async def sleep(delay: float) -> None:
        await asyncio.sleep(delay)

    async def deep_link(self, url: str, service: str):
        compose = f"{url}?{service}"
        cmd = f"{' '.join(self.initial)} shell am start -W -a android.intent.action.VIEW -d \"{compose}\""
        pattern = "(?<=input_text=).*?(?=\\&)"
        if input_text := re.search(fr"{pattern}", cmd):
            if len(text := input_text.group()) > 2:
                cmd = re.sub(fr"{pattern}", quote(text), cmd)
        await Terminal.cmd_line_shell(cmd)

    async def tap(self, x: int, y: int) -> None:
        cmd = self.initial + ["shell", "input", "tap", f"{x}", f"{y}"]
        await Terminal.cmd_line(*cmd)

    async def screen_status(self) -> str:
        cmd = self.initial + ["shell", "dumpsys", "deviceidle", "|", "grep", "mScreenOn"]
        result = await Terminal.cmd_line(*cmd)
        return result.split("=")[-1].strip()

    async def swipe_unlock(self) -> None:
        if await self.screen_status() == "false":
            await self.key_event(26)
            await self.sleep(1)
            cmd = self.initial + ["shell", "input", "touchscreen", "swipe", "250", "650", "250", "50"]
            await Terminal.cmd_line(*cmd)
            await self.sleep(1)

    async def key_event(self, key_code: int) -> None:
        cmd = self.initial + ["shell", "input", "keyevent", f"{key_code}"]
        await Terminal.cmd_line(*cmd)

    async def force_stop(self, pkg: str) -> None:
        cmd = self.initial + ["shell", "am", "force-stop", pkg]
        await Terminal.cmd_line(*cmd)

    async def wifi(self, power: str) -> None:
        cmd = self.initial + ["shell", "svc", "wifi", power]
        await Terminal.cmd_line(*cmd)

    async def package_list(self, level: int) -> list[str]:
        cmd = self.initial + ["shell", "ps"]
        result = await Terminal.cmd_line(*cmd)
        package_list = []
        for line in result.splitlines():
            uid = parts[1] if len(parts := line.split()) > 1 else None
            if uid and uid.isdigit() and int(uid) >= level:
                package_list.append(parts[-1])
        return package_list

    async def current_activity(self) -> str:
        cmd = self.initial + ["shell", "dumpsys", "window", "|", "grep", "mCurrentFocus"]
        result = await Terminal.cmd_line(*cmd)
        match = re.search(r"(?<=Window\{).*?(?=})", result)
        return match.group().split()[-1].strip() if match else ""

    async def start_application(self, pkg: str) -> str:
        cmd = self.initial + ["shell", "am", "start", "-n", pkg]
        result = await Terminal.cmd_line(*cmd)
        return result.strip()

    async def screen_size(self) -> str:
        cmd = self.initial + ["shell", "wm", "size"]
        result = await Terminal.cmd_line(*cmd)
        return result.strip()

    async def screen_density(self) -> str:
        cmd = self.initial + ["shell", "wm", "density"]
        result = await Terminal.cmd_line(*cmd)
        return result.strip()


if __name__ == '__main__':
    pass
