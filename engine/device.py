#
#   ____             _
#  |  _ \  _____   _(_) ___ ___
#  | | | |/ _ \ \ / / |/ __/ _ \
#  | |_| |  __/\ V /| | (_|  __/
#  |____/ \___| \_/ |_|\___\___|
#

import re
import typing
import asyncio
import uiautomator2 as u2
from urllib.parse import quote
from engine.terminal import Terminal


class _Phone(object):

    def __init__(self, sn: str, *args):
        self.sn = sn
        self.tag, self.ver, self.cpu, self.ram, self.display, *_ = args
        self.id = list(self.display.keys())[0] if self.display else 0

    def __str__(self):
        head = f"<Device {self.tag} SN={self.sn} OS=[{self.ver}]"
        tail = f"CPU=[{self.cpu}] RAM=[{self.ram}] ID=[{self.id}] DISPLAY={self.display}>"
        return head + " " + tail

    __repr__ = __str__


class Device(_Phone):

    __facilities: typing.Optional["u2.Device"] = None

    def __init__(self, adb: str, sn: str, *args):
        """
        初始化 Device 类的实例。
        """
        super().__init__(sn, *args)
        self.__initial = [adb, "-s", self.sn, "wait-for-device"]

    @property
    def facilities(self) -> typing.Optional["u2.Device"]:
        """
        返回与设备的连接实例 (uiautomator2)。
        """
        return self.__facilities

    @facilities.setter
    def facilities(self, value: typing.Optional["u2.Device"]):
        """
        设置设备的连接实例。
        """
        self.__facilities = value

    @staticmethod
    async def sleep(delay: float, *_, **__) -> None:
        """
        异步等待指定的时间。
        """
        await asyncio.sleep(delay)

# platform-tools #######################################################################################################

    async def deep_link(self, url: str, service: str, *_, **__) -> None:
        """
        通过深度链接启动指定的应用服务。
        """
        compose = f"{url}?{service}"
        cmd = f"{' '.join(self.__initial)} shell am start -W -a android.intent.action.VIEW -d \"{compose}\""
        if input_text := re.search(r"(?<=input_text=).*?(?=\\&)", cmd):
            if (text := input_text.group()) != "''":
                cmd = re.sub(r"(?<=input_text=).*?(?=\\&)", quote(text), cmd)
        return resp if (resp := await Terminal.cmd_line_shell(cmd)) else None

    async def activity(self, *_, **__) -> typing.Optional[str]:
        """
        获取当前设备活动的应用程序或窗口名称。
        """
        cmd = self.__initial + [
            "shell", "dumpsys", "window", "|", "findstr", "mCurrentFocus"
        ]
        if resp := await Terminal.cmd_line(*cmd):
            if match := re.search(r"(?<=Window\{).*?(?=})", resp):
                return match.group().split()[-1]

    async def screen_status(self, *_, **__) -> typing.Optional[bool]:
        """
        检查设备屏幕是否处于打开状态。
        """
        cmd = self.__initial + [
            "shell", "dumpsys", "deviceidle", "|", "findstr", "mScreenOn"
        ]
        return bool(resp.split("=")[-1]) if (resp := await Terminal.cmd_line(*cmd)) else None

    async def tap(self, x: int, y: int, *_, **__) -> None:
        """
        模拟在设备屏幕上点击指定坐标位置。
        """
        cmd = self.__initial + [
            "shell", "input", "tap", f"{x}", f"{y}"
        ]
        await Terminal.cmd_line(*cmd)

    async def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: int, *_, **__) -> None:
        """
        模拟在设备屏幕上从一个位置滑动到另一个位置。
        """
        cmd = self.__initial + [
            "shell", "input", "touchscreen", "swipe", f"{start_x}", f"{start_y}", f"{end_x}", f"{end_y}", f"{duration}"
        ]
        await Terminal.cmd_line(*cmd)

    async def key_event(self, key_code: int, *_, **__) -> None:
        """
        模拟按下设备的硬件按键。
        """
        cmd = self.__initial + [
            "shell", "input", "keyevent", f"{key_code}"
        ]
        await Terminal.cmd_line(*cmd)

    async def force_stop(self, package: str, *_, **__) -> None:
        """
        强制停止指定包名的应用。
        """
        cmd = self.__initial + [
            "shell", "am", "force-stop", package
        ]
        await Terminal.cmd_line(*cmd)

    async def notification(self, *_, **__) -> None:
        """
        打开设备的通知栏。
        """
        cmd = self.__initial + [
            "shell", "cmd", "statusbar", "expand-notifications"
        ]
        await Terminal.cmd_line(*cmd)

    async def install(self, apk_source: str, *_, **__) -> None:
        """
        安装APK文件。
        """
        cmd = self.__initial + [
            "install", apk_source
        ]
        await Terminal.cmd_line(*cmd)

    async def uninstall(self, package: str, *_, **__) -> None:
        """
        卸载指定的应用程序。
        """
        cmd = self.__initial + [
            "uninstall", package
        ]
        await Terminal.cmd_line(*cmd)

    async def screenshot(self, destination: str, *_, **__) -> None:
        """
        截取设备屏幕并保存到指定路径。
        """
        cmd = self.__initial + [
            "shell", "screencap", "-p", destination
        ]
        await Terminal.cmd_line(*cmd)

    async def wifi(self, mode: str, *_, **__) -> None:
        """
        打开或关闭设备的 Wi-Fi。
        """
        cmd = self.__initial + [
            "shell", "svc", "wifi", mode
        ]
        await Terminal.cmd_line(*cmd)

    async def hot_spot(self, mode: str, status: str, *_, **__) -> None:
        """
        控制设备的Wi-Fi或热点开关状态。
        """
        cmd = self.__initial + [
            "shell", "svc", "wifi", mode, status
        ]
        await Terminal.cmd_line(*cmd)

    async def start_application(self, package: str, *_, **__) -> typing.Optional[str]:
        """
        启动指定包名的应用。
        """
        cmd = self.__initial + [
            "shell", "am", "start", "-n", package
        ]
        return resp if (resp := await Terminal.cmd_line(*cmd)) else None

    async def screen_size(self, *_, **__) -> typing.Optional[str]:
        """
        获取设备屏幕的分辨率。
        """
        cmd = self.__initial + [
            "shell", "wm", "size"
        ]
        return resp if (resp := await Terminal.cmd_line(*cmd)) else None

    async def screen_density(self, *_, **__) -> typing.Optional[str]:
        """
        获取设备屏幕的像素密度。
        """
        cmd = self.__initial + [
            "shell", "wm", "density"
        ]
        return resp if (resp := await Terminal.cmd_line(*cmd)) else None

# uiautomator2 #########################################################################################################

    async def automator_activation(self) -> None:
        """
        通过设备的序列号激活 uiautomator2 连接。
        """
        self.facilities = await asyncio.to_thread(u2.connect, self.sn)

    async def automator(
            self,
            method: str,
            selector: typing.Optional[dict] = None,
            *args,
            **kwargs
    ) -> typing.Union[None, str, Exception]:
        """
        自动化方法的异步调用函数。

        参数:
            - method (str): 要调用的目标方法名称，作为字符串传递。
            - selector (dict, optional): 一个可选的字典，用于定位元素的选择器。默认值为 None。
            - *args: 可变位置参数，传递给目标方法。
            - **kwargs: 可变关键字参数，传递给目标方法。

        功能说明:
            1. 根据提供的 `selector` 查找元素。如果未提供选择器，则使用默认的设施对象。
            2. 通过 `getattr` 动态获取 `element` 对象上对应的方法。
            3. 如果该方法可调用，则将其与提供的参数一起异步执行。
            4. 使用 `asyncio.to_thread` 将同步函数调用转换为异步调用，以避免阻塞事件循环。
            5. 返回方法的执行结果。如果结果为空，则返回 None。

        返回值:
            - 如果执行成功，返回方法的结果。
            - 如果没有返回值或方法调用结果为空，则返回 None。
            - 如果发生异常，返回异常对象。

        注意:
            - 该方法是异步方法，适用于需要在事件循环中调用的方法。
            - 确保 `method` 是 `element` 对象上的有效方法名称，否则可能会引发 AttributeError。
        """
        element = self.facilities(**selector) if selector else self.facilities
        if callable(function := getattr(element, method)):
            return resp if (resp := await asyncio.to_thread(function, *args, **kwargs)) else None


if __name__ == '__main__':
    pass
