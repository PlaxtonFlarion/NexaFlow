import re
from typing import Any, Optional
from engine.activate import Active
from engine.terminal import Terminal
from plan.skills.device import Device


class Manage(object):

    __mobile: Optional["Device"] = None

    def __init__(self, log_level: str):
        self.__device_dict = {}
        self.__current_device()
        Active.active(log_level)

    @property
    def mobile(self):
        assert self.__mobile, "未连接设备 ..."
        return self.__mobile

    @mobile.setter
    def mobile(self, value):
        self.__mobile = value or None

    def __current_device(self) -> None:
        cmd = ["adb", "devices", "-l"]
        result = Terminal.cmd_oneshot(cmd)

        fit: Any = lambda x: re.search(r"(?<=:).*", x).group()
        for line in result.splitlines()[1:]:
            if line:
                serial, _, models, *_ = line.split()
                self.__device_dict.update({serial: Device(serial, fit(models))})

        if len(self.__device_dict) == 1:
            for _, device in self.__device_dict.items():
                self.mobile = device

    def operate_device(self, serial: str) -> Optional["Device"]:
        return self.__device_dict.get(serial, None)


if __name__ == '__main__':
    pass
