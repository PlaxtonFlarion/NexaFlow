import re
import time
from typing import List
from engine.terminal import Terminal


class Device(Terminal):

    def __init__(self, serial: str, models: str):
        self.serial, self.models = serial, models
        self.__initial = ["adb", "-s", self.serial, "wait-for-usb-device"]

    def __str__(self):
        return f"<Device [{self.serial}] - [{self.models}]>"

    __repr__ = __str__

    @staticmethod
    def sleep(secs: float):
        time.sleep(secs)

    def tap(self, x: int, y: int) -> None:
        cmd = self.__initial + ["shell", "input", "tap", str(x), str(y)]
        self.cmd_oneshot(cmd)

    def is_screen_on(self) -> str:
        cmd = self.__initial + ["shell", "dumpsys", "deviceidle", "|", "grep", "mScreenOn"]
        result = self.cmd_oneshot(cmd)
        return result.split("=")[-1].strip()

    def swipe_unlock(self) -> None:
        if self.is_screen_on() == "false":
            self.key_event(26)
            time.sleep(0.5)
            cmd = self.__initial + ["shell", "input", "touchscreen", "swipe", "250", "650", "250", "50"]
            self.cmd_oneshot(cmd)
            time.sleep(0.5)

    def key_event(self, key_code: int) -> None:
        cmd = self.__initial + ["shell", "input", "keyevent", str(key_code)]
        self.cmd_oneshot(cmd)

    def force_stop(self, package: str) -> None:
        cmd = self.__initial + ["shell", "am", "force-stop", package]
        self.cmd_oneshot(cmd)

    def wifi(self, switch: str = "enable") -> None:
        cmd = self.__initial + ["shell", "svc", "wifi", switch]
        self.cmd_oneshot(cmd)

    def all_package(self, level: int = 10000) -> List[str]:
        cmd = self.__initial + ["shell", "ps"]
        result = self.cmd_oneshot(cmd)
        package_list = []
        for line in result.splitlines():
            parts = line.split()
            uid = parts[1] if len(parts) > 1 else None
            if uid and uid.isdigit() and int(uid) >= level:
                pkg_name = parts[-1]
                package_list.append(pkg_name)
        return package_list

    def current_activity(self) -> str:
        cmd = self.__initial + ["shell", "dumpsys", "window", "|", "grep", "mCurrentFocus"]
        result = self.cmd_oneshot(cmd)
        match = re.search(r"(?<=Window\{).*?(?=})", result)
        return match.group().split()[-1].strip() if match else ""

    def start_app(self, package: str = "com.android.settings/com.android.settings.Settings") -> str:
        cmd = self.__initial + ["shell", "am", "start", "-n", package]
        result = self.cmd_oneshot(cmd)
        return result.strip()

    def screen_size(self) -> str:
        cmd = self.__initial + ["shell", "wm", "size"]
        result = self.cmd_oneshot(cmd)
        return result.strip()

    def screen_density(self) -> str:
        cmd = self.__initial + ["shell", "wm", "density"]
        result = self.cmd_oneshot(cmd)
        return result.strip()

    def force_filter(self, package: str) -> None:
        current_screen = self.current_activity()
        if package not in current_screen:
            screen = current_screen.split("/")[0] if "/" in current_screen else current_screen
            self.force_stop(screen)


if __name__ == '__main__':
    pass
