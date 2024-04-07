import re
from typing import Any, Optional
from engine.terminal import Terminal
from plan.skills.device import Device


class Manage(object):

    Phone: Optional["Device"] = None

    def __init__(self):
        self.device_dict = {}
        self.current_device()

    def current_device(self) -> None:
        cmd = ["adb", "devices", "-l"]
        result = Terminal.cmd_oneshot(cmd)

        fit: Any = lambda x: re.search(r"(?<=:).*", x).group()
        for line in result.splitlines()[1:]:
            if line:
                serial, _, models, *_ = line.split()
                self.device_dict.update({serial: Device(serial, fit(models))})

        if len(self.device_dict) == 1:
            for _, device in self.device_dict.items():
                self.Phone = device

    def operate_device(self, serial: str) -> Optional["Device"]:
        return self.device_dict.get(serial, None)


if __name__ == '__main__':
    pass
