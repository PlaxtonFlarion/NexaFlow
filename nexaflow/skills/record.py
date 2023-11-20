import os
import time
import signal
import threading
from loguru import logger
from subprocess import Popen
from typing import Union, IO, Optional
from nexaflow.terminal import Terminal


class Record(object):

    def __init__(self):
        self.__connection: Optional[Popen] = None
        self.__record_event: threading.Event = threading.Event()
        self.__initial: str = "scrcpy"

    def start_record(self, video_path: str, serial: str = None) -> None:
        cmd = [
            self.__initial, "--no-audio", "--video-bit-rate", "8M", "--max-fps", "60", "-Nr",
            f"{os.path.join(video_path, 'screen')}.mkv"
        ]
        if serial:
            cmd.insert(1, "-s")
            cmd.insert(2, serial)
        self.__connection = Terminal.cmd_connect(cmd)

        def stream(flow: Union[int, IO[str]]) -> None:
            for line in iter(flow.readline, ""):
                logger.info(" ".join(line.strip().split()))
            flow.close()

        if self.__connection:
            self.__record_event.set()
            threading.Thread(target=stream, args=(self.__connection.stdout, )).start()
            threading.Thread(target=stream, args=(self.__connection.stderr, )).start()
            time.sleep(1)

    def stop_record(self) -> None:
        self.__connection.send_signal(signal.CTRL_C_EVENT)
        self.__record_event.clear()
        self.__connection = None

        try:
            Terminal.cmd_oneshot(["taskkill", "/im", "scrcpy.exe"])
        except KeyboardInterrupt:
            logger.info("Stop with Ctrl_C_Event ...")


if __name__ == '__main__':
    pass
