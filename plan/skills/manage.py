import re
from typing import Any, Optional
from engine.terminal import Terminal
from plan.skills.device import Device


class Manage(object):

    Phone: Optional["Device"] = None

    def __init__(self):
        self.__device_dict: dict[str, "Device"] = {}
        self.__serial_list: list[str] = []
        self.current_device()

    @property
    def serials(self) -> list[str]:
        return self.__serial_list

    def current_device(self) -> None:
        cmd = ["adb", "devices", "-l"]
        result = Terminal.cmd_oneshot(cmd)

        fit: Any = lambda x: re.search(r"(?<=:).*", x).group()
        for line in result.splitlines()[1:]:
            if line:
                serial, _, models, *_ = line.split()
                self.__device_dict.update({serial: Device(serial, fit(models))})
                self.__serial_list.append(serial)

        if len(self.__device_dict) == 1:
            for _, device in self.__device_dict.items():
                self.Phone = device

    def operate_device(self, serial: str) -> Optional["Device"]:
        return self.__device_dict.get(serial, None)


if __name__ == '__main__':
    pass
