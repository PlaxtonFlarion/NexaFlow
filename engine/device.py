#   ____             _
#  |  _ \  _____   _(_) ___ ___
#  | | | |/ _ \ \ / / |/ __/ _ \
#  | |_| |  __/\ V /| | (_|  __/
#  |____/ \___| \_/ |_|\___\___|
#

# ==== Notes: 版权申明 ====
# 版权所有 (c) 2024  Framix(画帧秀)
# 此文件受 Framix(画帧秀) 许可证的保护。您可以在 LICENSE.md 文件中查看详细的许可条款。

# ==== Notes: License ====
# Copyright (c) 2024  Framix(画帧秀)
# This file is licensed under the Framix(画帧秀) License. See the LICENSE.md file for more details.

# ==== Notes: ライセンス ====
# Copyright (c) 2024  Framix(画帧秀)
# このファイルは Framix(画帧秀) ライセンスの下でライセンスされています。詳細は LICENSE.md ファイルを参照してください。

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
        await asyncio.sleep(delay)

    async def device_online(self, *_, **__) -> None:
        """
        等待设备上线。
        """
        await Terminal.cmd_line(self.__initial)

    async def deep_link(self, url: str, service: str, *_, **__) -> None:
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
        return resp if (resp := await Terminal.cmd_line_shell(cmd)) else None

    async def activity(self, *_, **__) -> typing.Optional[str]:
        """
        获取当前设备活动的应用程序或窗口名称。
        """
        cmd = self.__initial + [
            "shell", "dumpsys", "window", "|", "findstr", "mCurrentFocus"
        ]
        if resp := await Terminal.cmd_line(cmd):
            if match := re.search(r"(?<=Window\{).*?(?=})", resp):
                return match.group().split()[-1]

    async def screen_status(self, *_, **__) -> typing.Optional[bool]:
        """
        检查设备屏幕是否处于打开状态。
        """
        cmd = self.__initial + [
            "shell", "dumpsys", "deviceidle", "|", "findstr", "mScreenOn"
        ]
        return bool(resp.split("=")[-1]) if (resp := await Terminal.cmd_line(cmd)) else None

    async def tap(self, x: int, y: int, *_, **__) -> None:
        """
        模拟在设备屏幕上点击指定坐标位置。
        """
        cmd = self.__initial + [
            "shell", "input", "tap", f"{x}", f"{y}"
        ]
        await Terminal.cmd_line(cmd)

    async def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: int, *_, **__) -> None:
        """
        模拟在设备屏幕上从一个位置滑动到另一个位置。
        """
        cmd = self.__initial + [
            "shell", "input", "touchscreen", "swipe", f"{start_x}", f"{start_y}", f"{end_x}", f"{end_y}", f"{duration}"
        ]
        await Terminal.cmd_line(cmd)

    async def key_event(self, key_code: int, *_, **__) -> None:
        """
        模拟按下设备的硬件按键。
        """
        cmd = self.__initial + [
            "shell", "input", "keyevent", f"{key_code}"
        ]
        await Terminal.cmd_line(cmd)

    async def force_stop(self, package: str, *_, **__) -> None:
        """
        强制停止指定包名的应用。
        """
        cmd = self.__initial + [
            "shell", "am", "force-stop", package
        ]
        await Terminal.cmd_line(cmd)

    async def notification(self, *_, **__) -> None:
        """
        打开设备的通知栏。
        """
        cmd = self.__initial + [
            "shell", "cmd", "statusbar", "expand-notifications"
        ]
        await Terminal.cmd_line(cmd)

    async def install(self, apk_source: str, *_, **__) -> None:
        """
        安装APK文件。
        """
        cmd = self.__initial + [
            "install", apk_source
        ]
        await Terminal.cmd_line(cmd)

    async def uninstall(self, package: str, *_, **__) -> None:
        """
        卸载指定的应用程序。
        """
        cmd = self.__initial + [
            "uninstall", package
        ]
        await Terminal.cmd_line(cmd)

    async def screenshot(self, destination: str, *_, **__) -> None:
        """
        截取设备屏幕并保存到指定路径。
        """
        cmd = self.__initial + [
            "shell", "screencap", "-p", destination
        ]
        await Terminal.cmd_line(cmd)

    async def wifi(self, mode: str, *_, **__) -> None:
        """
        打开或关闭设备的 Wi-Fi。
        """
        cmd = self.__initial + [
            "shell", "svc", "wifi", mode
        ]
        await Terminal.cmd_line(cmd)

    async def hot_spot(self, mode: str, status: str, *_, **__) -> None:
        """
        控制设备的Wi-Fi或热点开关状态。
        """
        cmd = self.__initial + [
            "shell", "svc", "wifi", mode, status
        ]
        await Terminal.cmd_line(cmd)

    async def start_application(self, package: str, *_, **__) -> typing.Optional[str]:
        """
        启动指定包名的应用。
        """
        cmd = self.__initial + [
            "shell", "am", "start", "-n", package
        ]
        return resp if (resp := await Terminal.cmd_line(cmd)) else None

    async def screen_size(self, *_, **__) -> typing.Optional[str]:
        """
        获取设备屏幕的分辨率。
        """
        cmd = self.__initial + [
            "shell", "wm", "size"
        ]
        return resp if (resp := await Terminal.cmd_line(cmd)) else None

    async def screen_density(self, *_, **__) -> typing.Optional[str]:
        """
        获取设备屏幕的像素密度。
        """
        cmd = self.__initial + [
            "shell", "wm", "density"
        ]
        return resp if (resp := await Terminal.cmd_line(cmd)) else None

    async def automator_activation(self, *_, **__) -> None:
        """
        通过设备的序列号激活 uiautomator2 连接。
        """
        self.facilities = await asyncio.to_thread(u2.connect, self.sn)

    async def automator(
            self,
            choice: typing.Optional[dict],
            method: typing.Optional[str],
            *args,
            **kwargs
    ) -> typing.Any:
        """
        自动化方法的异步调用函数。
        """
        if (element := self.facilities(**choice) if choice else self.facilities).exists():
            if callable(function := getattr(element, method)):
                try:
                    resp = await asyncio.to_thread(function, *args, **kwargs)
                except Exception as e:
                    return e
                return resp
            raise AttributeError(f"{method} Not callable ...")
        raise AttributeError(f"{choice or element.serial} Not found ...")


if __name__ == '__main__':
    pass
