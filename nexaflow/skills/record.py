import os
import sys
import signal
import asyncio
import threading
from asyncio.subprocess import Process
from subprocess import Popen
from loguru import logger
from typing import Union, IO, Optional
from nexaflow.terminal import Terminal


class Record(object):

    def __init__(self):
        self.__connection: Optional[Popen] = None
        self.__transports: Optional[Process] = None
        self.__input_task: Optional[asyncio.Task] = None
        self.__error_task: Optional[asyncio.Task] = None
        self.__record_event: threading.Event = threading.Event()
        self.__initial: str = "scrcpy"

    @property
    def record_event(self) -> threading.Event:
        return self.__record_event

    @record_event.setter
    def record_event(self, value):
        self.__record_event = value

    @property
    def connection(self):
        return self.__connection

    @connection.setter
    def connection(self, value):
        self.__connection = value

    @property
    def transports(self) -> Optional[Process]:
        return self.__transports

    @transports.setter
    def transports(self, value):
        self.__transports = value

    @property
    def input_task(self):
        return self.__input_task

    @input_task.setter
    def input_task(self, value):
        self.__input_task = value

    @property
    def error_task(self):
        return self.__error_task

    @error_task.setter
    def error_task(self, value):
        self.__error_task = value

    async def start_record_display(self, video_path: str, serial: str = None) -> None:
        cmd = [
            self.__initial, "--no-audio", "--video-bit-rate", "8M", "--max-fps", "60", "--record",
            f"{os.path.join(video_path, 'screen')}.mkv"
        ]
        if serial:
            cmd.insert(1, "-s")
            cmd.insert(2, serial)
        self.connection = Terminal.cmd_connect(cmd)

        def stream(flow: Union[int, IO[str]]) -> None:
            for line in iter(flow.readline, ""):
                logger.info(" ".join(line.strip().split()))
            flow.close()

        if self.connection:
            self.__record_event.set()
            threading.Thread(target=stream, args=(self.connection.stdout, )).start()
            threading.Thread(target=stream, args=(self.connection.stderr, )).start()
            await asyncio.sleep(1)

    async def start_record_silence(self, video_path: str, serial: str = None) -> None:
        cmd = [
            self.__initial, "--no-audio", "--video-bit-rate", "8M", "--max-fps", "60", "-Nr",
            f"{os.path.join(video_path, 'screen')}.mkv"
        ]
        if serial:
            cmd.insert(1, "-s")
            cmd.insert(2, serial)
        self.transports, self.input_task, self.error_task = Terminal.cmd_link(*cmd)
        if self.transports:
            self.__record_event.set()
            await asyncio.sleep(1)

    async def stop_record(self) -> None:
        if sys.platform == "win32":
            self.input_task.cancel()
            self.error_task.cancel()
            self.transports.send_signal(signal.CTRL_C_EVENT)
        else:
            self.transports.terminate()

        self.record_event.clear()
        self.connection = None
        self.transports = None


if __name__ == '__main__':
    pass
