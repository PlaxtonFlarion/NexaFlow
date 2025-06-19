#   ____             _
#  |  _ \  _____   _(_) ___ ___
#  | | | |/ _ \ \ / / |/ __/ _ \
#  | |_| |  __/\ V /| | (_|  __/
#  |____/ \___| \_/ |_|\___\___|
#
# ==== Notes: License ====
# Copyright (c) 2024  Framix :: 画帧秀
# This file is licensed under the Framix :: 画帧秀 License. See the LICENSE.md file for more details.

import re
import typing
import asyncio
import uiautomator2 as u2
from urllib.parse import quote
from engine.terminal import Terminal


class _Phone(object):
    """
    设备信息基础模型类。

    用于封装 Android 设备的基本信息，例如序列号、品牌、系统版本、CPU 核心数、内存大小和显示分辨率等，常用于初始化与表示设备对象。
    """

    display: dict = {}

    def __init__(self, sn: str, *args):
        """
        初始化设备对象。
        """
        self.sn = sn

        self.tag, self.ver, self.cpu, self.ram, self.display, *_ = args

        try:
            self.id = list(self.display.keys())[0] if self.display else 0
        except IndexError:
            self.id = 0

        self.screen_w, self.screen_h = self.display[self.id]

    def __str__(self):
        """
        格式化设备对象为字符串描述。
        """
        head = f"<Device {self.tag} SN={self.sn} OS=[{self.ver}]"
        tail = f"CPU=[{self.cpu}] RAM=[{self.ram}] ID=[{self.id}] DISPLAY={self.display}>"
        return head + " " + tail

    __repr__ = __str__


class Device(_Phone):
    """
    Android 设备操作封装类，提供设备级数据采集与控制接口。

    该类用于连接并操作指定序列号的 Android 设备，提供基于 ADB 与 uiautomator2 的高层封装。
    """

    __facilities: typing.Optional[typing.Union["u2.Device", "u2.UiObject"]] = None

    def __init__(self, adb: str, sn: str, *args):
        """
        初始化 Device 类的实例。
        """
        super().__init__(sn, *args)
        self.__initial = [adb, "-s", self.sn, "wait-for-device"]

    @property
    def facilities(self) -> typing.Optional[typing.Union["u2.Device", "u2.UiObject"]]:
        """
        属性方法 `facilities`，用于获取当前实例的设施属性，可能为设备对象或 UI 控件对象。
        """
        return self.__facilities

    @facilities.setter
    def facilities(self, value: typing.Optional[typing.Union["u2.Device", "u2.UiObject"]]) -> None:
        """
        设置方法 `facilities`，用于将设备对象或 UI 控件对象赋值给设施属性。
        """
        self.__facilities = value

    @staticmethod
    async def sleep(delay: float, *_, **__) -> None:
        """
        异步等待指定的时间。
        """
        return await asyncio.sleep(delay)

    # Notes: ======================== ADB ========================

    async def device_online(self, *_, **__) -> typing.Any:
        """
        等待设备上线。
        """
        return await Terminal.cmd_line(self.__initial)

    async def deep_link(self, url: str, service: str, *_, **__) -> typing.Any:
        """
        通过深度链接启动指定的应用服务。
        """
        pattern = r"(?<=input_text=).*?(?=\\$|\"$|\\\\$)"
        compose = f"{url}?{service}"

        initial = [f'"{self.__initial[0]}"'] + (self.__initial[1:])
        cmd = f"{' '.join(initial)} shell am start -W -a android.intent.action.VIEW -d \"{compose}\""

        if input_text := re.search(pattern, cmd):
            if (text := input_text.group()) != "''":
                cmd = re.sub(pattern, "\'" + quote(text) + "\'", cmd)

        return await Terminal.cmd_line_shell(cmd)

    async def activity(self, *_, **__) -> typing.Any:
        """
        获取当前设备活动的应用程序或窗口名称。
        """
        cmd = self.__initial + ["shell", "dumpsys", "window", "|", "grep", "mCurrentFocus"]
        response = await Terminal.cmd_line(cmd)

        return match.group().split()[-1] if (
            match := re.search(r"(?<=Window\{).*?(?=})", response)
        ) else None

    async def tap(self, x: int, y: int, *_, **__) -> typing.Any:
        """
        模拟在设备屏幕上点击指定坐标位置。
        """
        cmd = self.__initial + ["shell", "input", "tap", f"{x}", f"{y}"]
        return await Terminal.cmd_line(cmd)

    async def swipe(
        self, start_x: int, start_y: int, end_x: int, end_y: int, duration: int = 500, *_, **__
    ) -> typing.Any:
        """
        模拟在设备屏幕上从一个位置滑动到另一个位置。
        """
        cmd = self.__initial + [
            "shell", "input", "swipe", f"{start_x}", f"{start_y}", f"{end_x}", f"{end_y}", f"{duration}"
        ]
        return await Terminal.cmd_line(cmd)

    async def key_event(self, key_code: int, *_, **__) -> typing.Any:
        """
        模拟按下设备的硬件按键。
        """
        cmd = self.__initial + ["shell", "input", "keyevent", f"{key_code}"]
        return await Terminal.cmd_line(cmd)

    async def notification(self, *_, **__) -> typing.Any:
        """
        打开设备的通知栏。
        """
        cmd = self.__initial + ["shell", "cmd", "statusbar", "expand-notifications"]
        return await Terminal.cmd_line(cmd)

    async def install(self, src: str, *_, **__) -> typing.Any:
        """
        安装APK文件。
        """
        cmd = self.__initial + ["install", src]
        return await Terminal.cmd_line(cmd)

    async def uninstall(self, package: str, *_, **__) -> typing.Any:
        """
        卸载指定的应用程序。
        """
        cmd = self.__initial + ["uninstall", package]
        return await Terminal.cmd_line(cmd)

    async def wifi(self, mode: str, *_, **__) -> typing.Any:
        """
        打开或关闭设备的 Wi-Fi。
        """
        cmd = self.__initial + ["shell", "svc", "wifi", mode]
        return await Terminal.cmd_line(cmd)

    async def hot_spot(self, mode: str, status: str, *_, **__) -> typing.Any:
        """
        控制设备的Wi-Fi或热点开关状态。
        """
        cmd = self.__initial + ["shell", "svc", "wifi", mode, status]
        return await Terminal.cmd_line(cmd)

    async def start_app(self, package: str, *_, **__) -> typing.Any:
        """
        启动指定包名的应用。
        """
        cmd = self.__initial + ["shell", "am", "start", "-n", package]
        return await Terminal.cmd_line(cmd)

    async def force_stop(self, package: str, *_, **__) -> typing.Any:
        """
        强制停止指定包名的应用。
        """
        cmd = self.__initial + ["shell", "am", "force-stop", package]
        return await Terminal.cmd_line(cmd)

    async def screenshot(self, dst: str, *_, **__) -> typing.Any:
        """
        截取设备屏幕并保存到指定路径。
        """
        cmd = self.__initial + ["shell", "screencap", "-p", dst]
        return await Terminal.cmd_line(cmd)

    async def screen_status(self, *_, **__) -> typing.Any:
        """
        检查设备屏幕是否处于打开状态。
        """
        cmd = self.__initial + ["shell", "dumpsys", "deviceidle", "|", "grep", "mScreenOn"]
        return await Terminal.cmd_line(cmd)

    async def screen_size(self, *_, **__) -> typing.Any:
        """
        获取设备屏幕的分辨率。
        """
        cmd = self.__initial + ["shell", "wm", "size"]
        response = await Terminal.cmd_line(cmd)

        return (int(match.group(1)), int(match.group(2))) if (
            match := re.search(r"Physical size:\s(\d+)x(\d+)", response)
        ) else None

    async def screen_density(self, *_, **__) -> typing.Any:
        """
        获取设备屏幕的像素密度。
        """
        cmd = self.__initial + ["shell", "wm", "density"]
        response = await Terminal.cmd_line(cmd)

        return int(match.group(1)) if (
            match := re.search(r"Physical density:\s(\d+)", response)
        ) else None

    async def screen_orientation(self, *_, **__) -> typing.Any:
        """
        获取当前屏幕方向。
        """
        cmd = self.__initial + ["shell", "dumpsys", "display", "|", "grep", "mCurrentOrientation"]
        response = await Terminal.cmd_line(cmd)

        return int(match.group(1)) if (
            match := re.search(r"mCurrentOrientation=(\d+)", response)
        ) else 0

    async def reboot(self, *_, **__) -> typing.Any:
        """
        重启设备。
        """
        cmd = self.__initial + ["reboot"]
        await Terminal.cmd_line(cmd)
        return await self.device_online()

    async def input_text(self, text: str, *_, **__) -> typing.Any:
        """
        向设备输入指定文本。
        """
        cmd = self.__initial + ["shell", "input", "text", text]
        return await Terminal.cmd_line(cmd)

    async def push(self, local: str, remote: str, *_, **__) -> typing.Any:
        """
        将本地文件推送到设备。
        """
        cmd = self.__initial + ["push", local, remote]
        return await Terminal.cmd_line(cmd)

    async def pull(self, remote: str, local: str, *_, **__) -> typing.Any:
        """
        从设备拉取文件到本地。
        """
        cmd = self.__initial + ["pull", remote, local]
        return await Terminal.cmd_line(cmd)

    async def unlock_screen(self, *_, **__) -> typing.Any:
        """
        解锁 Android 设备屏幕。
        """
        if "true" in await self.screen_status():
            return None

        w, h = await self.screen_size()

        start_x = end_x = w // 2
        start_y, end_y = int(h * 0.8), int(h * 0.2)

        await self.key_event(26)
        await asyncio.sleep(1)
        await self.swipe(start_x, start_y, end_x, end_y)

    # Notes: ======================== Automator ========================

    async def automator_activation(self, *_, **__) -> None:
        """
        通过设备的序列号激活 uiautomator2 连接。
        """
        self.facilities = await asyncio.to_thread(u2.connect, self.sn)

    async def automator_unlock_screen(self, *_, **__) -> typing.Any:
        """
        解锁 Android 设备屏幕。
        """
        if not self.facilities.info.get("screenOn"):
            await asyncio.to_thread(self.facilities.screen_on)

        return await asyncio.to_thread(self.facilities.swipe_ext, direction="up", scale=0.8)

    async def automator(
        self, selector: typing.Optional[dict], function: str, *args, **kwargs
    ) -> typing.Any:
        """
        自动化方法的异步调用函数。
        """
        try:
            element = self.facilities(**selector) if selector else self.facilities

            return await asyncio.to_thread(
                current, *args, **kwargs
            ) if callable(current := getattr(element, function)) else current

        except Exception as e:
            return e


if __name__ == '__main__':
    pass
