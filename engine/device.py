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

        参数:
        adb (str): ADB 路径。
        sn (str): 设备的序列号。
        *args: 其他可选参数。

        该方法调用了父类 _Phone 的构造函数，并初始化用于等待设备连接的 ADB 命令。
        """
        super().__init__(sn, *args)
        self.__initial = [adb, "-s", self.sn, "wait-for-device"]

    @property
    def facilities(self) -> typing.Optional["u2.Device"]:
        """
        返回与设备的连接实例 (uiautomator2)。

        Returns:
        typing.Optional[u2.Device]: 返回设备连接实例，或如果未连接则返回 None。
        """
        return self.__facilities

    @facilities.setter
    def facilities(self, value: typing.Optional["u2.Device"]):
        """
        设置设备的连接实例。

        参数:
        value (typing.Optional[u2.Device]): 设备连接实例，或 None。
        """
        self.__facilities = value

    @staticmethod
    async def sleep(delay: float) -> None:
        """
        异步等待指定的时间。

        参数:
        delay (float): 需要等待的秒数。
        """
        await asyncio.sleep(delay)

# platform-tools #######################################################################################################

    async def deep_link(self, url: str, service: str) -> None:
        """
        通过深度链接启动指定的应用服务。

        参数:
        url (str): 应用的 URL。
        service (str): 服务的参数。

        该方法构建一个 Android Intent 命令来启动指定的服务，并通过 ADB 执行该命令。
        如果命令中包含需要 URL 编码的文本，则进行编码处理。
        """
        compose = f"{url}?{service}"
        cmd = f"{' '.join(self.__initial)} shell am start -W -a android.intent.action.VIEW -d \"{compose}\""
        if input_text := re.search(r"(?<=input_text=).*?(?=\\&)", cmd):
            if (text := input_text.group()) != "''":
                cmd = re.sub(r"(?<=input_text=).*?(?=\\&)", quote(text), cmd)
        return resp if (resp := await Terminal.cmd_line_shell(cmd)) else None

    async def activity(self) -> typing.Optional[str]:
        """
        获取当前设备活动的应用程序或窗口名称。

        Returns:
        typing.Optional[str]: 返回当前活动的应用或窗口名称，如果无法获取则返回 None。

        该方法使用 ADB 命令 'dumpsys window' 来获取当前活动窗口的信息，并解析出活动窗口的名称。
        """
        cmd = self.__initial + [
            "shell", "dumpsys", "window", "|", "findstr", "mCurrentFocus"
        ]
        if resp := await Terminal.cmd_line(*cmd):
            if match := re.search(r"(?<=Window\{).*?(?=})", resp):
                return match.group().split()[-1]

    async def screen_status(self) -> typing.Optional[bool]:
        """
        检查设备屏幕是否处于打开状态。

        Returns:
        typing.Optional[bool]: 如果屏幕是打开的，返回 True；如果屏幕是关闭的，返回 False；如果无法确定，则返回 None。

        该方法通过 ADB 命令 'dumpsys deviceidle' 来检查设备屏幕的状态。
        """
        cmd = self.__initial + [
            "shell", "dumpsys", "deviceidle", "|", "findstr", "mScreenOn"
        ]
        return bool(resp.split("=")[-1]) if (resp := await Terminal.cmd_line(*cmd)) else None

    async def tap(self, x: int, y: int) -> None:
        """
        模拟在设备屏幕上点击指定坐标位置。

        参数:
        x (int): 点击的 x 坐标。
        y (int): 点击的 y 坐标。

        该方法使用 ADB 命令 'input tap' 在指定坐标模拟点击操作。
        """
        cmd = self.__initial + [
            "shell", "input", "tap", f"{x}", f"{y}"
        ]
        await Terminal.cmd_line(*cmd)

    async def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: int) -> None:
        """
        模拟在设备屏幕上从一个位置滑动到另一个位置。

        参数:
        start_x (int): 滑动起点的 x 坐标。
        start_y (int): 滑动起点的 y 坐标。
        end_x (int): 滑动终点的 x 坐标。
        end_y (int): 滑动终点的 y 坐标。
        duration (int): 滑动的持续时间，以毫秒为单位。

        该方法使用 ADB 命令 'input touchscreen swipe' 在屏幕上执行滑动操作。
        """
        cmd = self.__initial + [
            "shell", "input", "touchscreen", "swipe", f"{start_x}", f"{start_y}", f"{end_x}", f"{end_y}", f"{duration}"
        ]
        await Terminal.cmd_line(*cmd)

    async def key_event(self, key_code: int) -> None:
        """
        模拟按下设备的硬件按键。

        参数:
        key_code (int): 按键的代码，例如返回键的代码为 4。

        该方法使用 ADB 命令 'input keyevent' 来模拟按键操作。
        """
        cmd = self.__initial + [
            "shell", "input", "keyevent", f"{key_code}"
        ]
        await Terminal.cmd_line(*cmd)

    async def force_stop(self, package: str) -> None:
        """
        强制停止指定包名的应用。

        参数:
        package (str): 需要强制停止的应用包名。

        该方法使用 ADB 命令 'am force-stop' 来停止应用。
        """
        cmd = self.__initial + [
            "shell", "am", "force-stop", package
        ]
        await Terminal.cmd_line(*cmd)

    async def notification(self) -> None:
        """
        打开设备的通知栏。
        """
        cmd = self.__initial + [
            "shell", "cmd", "statusbar", "expand-notifications"
        ]
        await Terminal.cmd_line(*cmd)

    async def install(self, apk_source: str) -> None:
        """
        安装APK文件。

        参数:
        apk_source (str): APK文件的路径。
        """
        cmd = self.__initial + [
            "install", apk_source
        ]
        await Terminal.cmd_line(*cmd)

    async def uninstall(self, package: str) -> None:
        """
        卸载指定的应用程序。

        参数:
        package (str): 应用程序的包名。
        """
        cmd = self.__initial + [
            "uninstall", package
        ]
        await Terminal.cmd_line(*cmd)

    async def screenshot(self, destination: str) -> None:
        """
        截取设备屏幕并保存到指定路径。

        参数:
        destination (str): 保存截屏文件的路径。
        """
        cmd = self.__initial + [
            "shell", "screencap", "-p", destination
        ]
        await Terminal.cmd_line(*cmd)

    async def wifi(self, mode: str) -> None:
        """
        打开或关闭设备的 Wi-Fi。

        参数:
        mode (str): 'enable' 表示打开 Wi-Fi，'disable' 表示关闭 Wi-Fi。

        该方法使用 ADB 命令 'svc wifi' 来控制 Wi-Fi 的状态。
        """
        cmd = self.__initial + [
            "shell", "svc", "wifi", mode
        ]
        await Terminal.cmd_line(*cmd)

    async def hot_spot(self, mode: str, status: str) -> None:
        """
        控制设备的Wi-Fi或热点开关状态。

        参数:
        mode (str): 表示要控制的模式类型。例如 "wifi" 用于控制 Wi-Fi 模块。
        status (str): 表示要执行的操作状态。例如 "enable" 或 "disable" 用于打开或关闭 Wi-Fi，"setapenabled true" 用于启用热点，"setapenabled false" 用于禁用热点。

        该方法通过 ADB 命令 `svc wifi` 来控制 Android 设备上的 Wi-Fi 或热点的启用和禁用。
        """
        cmd = self.__initial + [
            "shell", "svc", "wifi", mode, status
        ]
        await Terminal.cmd_line(*cmd)

    async def start_application(self, package: str) -> typing.Optional[str]:
        """
        启动指定包名的应用。

        参数:
        package (str): 应用的包名及活动名称，格式为 'com.package/.MainActivity'。

        Returns:
        typing.Optional[str]: 返回启动应用的响应结果，如果失败则返回 None。

        该方法使用 ADB 命令 'am start' 来启动应用。
        """
        cmd = self.__initial + [
            "shell", "am", "start", "-n", package
        ]
        return resp if (resp := await Terminal.cmd_line(*cmd)) else None

    async def screen_size(self) -> typing.Optional[str]:
        """
        获取设备屏幕的分辨率。

        Returns:
        typing.Optional[str]: 返回屏幕的分辨率，例如 '1080x1920'，如果失败则返回 None。

        该方法使用 ADB 命令 'wm size' 来获取设备屏幕的分辨率。
        """
        cmd = self.__initial + [
            "shell", "wm", "size"
        ]
        return resp if (resp := await Terminal.cmd_line(*cmd)) else None

    async def screen_density(self) -> typing.Optional[str]:
        """
        获取设备屏幕的像素密度。

        Returns:
        typing.Optional[str]: 返回屏幕的像素密度，例如 '480'，如果失败则返回 None。

        该方法使用 ADB 命令 'wm density' 来获取设备屏幕的像素密度。
        """
        cmd = self.__initial + [
            "shell", "wm", "density"
        ]
        return resp if (resp := await Terminal.cmd_line(*cmd)) else None

# uiautomator2 #########################################################################################################

    async def automator_activation(self) -> None:
        """
        通过设备的序列号激活 uiautomator2 连接。

        该方法通过调用 asyncio.to_thread 方法在后台线程中连接到设备。
        """
        self.facilities = await asyncio.to_thread(u2.connect, self.sn)

    async def automator(self, method: str, selector: dict = None, *args) -> typing.Union[None, str, Exception]:
        """
        通过 uiautomator2 调用设备上的自动化方法。

        参数:
        method (str): 要调用的方法名。
        selector (dict, optional): 用于选择 UI 元素的选择器。
        *args: 传递给方法的额外参数，可以是位置参数或字典参数。

        Returns:
        typing.Union[None, str, Exception]: 返回方法执行的结果，如果方法返回了响应则返回响应字符串，如果发生异常则返回异常。

        该方法使用 `getattr` 动态获取 `uiautomator2` 元素上的方法，并使用传递的参数调用它。
        它首先尝试从 `args` 中分离出字典参数和非字典参数，最后将这些参数传递给目标方法执行。
        """
        element = self.facilities(**selector) if selector else self.facilities
        if callable(function := getattr(element, method)):
            arg_list, arg_dict = [], {}
            for arg in args:
                if isinstance(arg, dict):
                    arg_dict.update(arg)
                else:
                    arg_list.append(arg)
            return resp if (resp := await asyncio.to_thread(function, *arg_list, **arg_dict)) else None


if __name__ == '__main__':
    pass
