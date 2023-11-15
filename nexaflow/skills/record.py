import os
import sys
import signal
import asyncio
import threading
import subprocess
from loguru import logger
from typing import Union, IO
from nexaflow.terminal import Terminal


class Record(object):

    def __init__(self):
        self.__process = None
        self.__initial = "scrcpy"

    async def start_record_display(self, video_path: str, serial: str = None) -> None:
        cmd = [
            self.__initial, "--no-audio", "--video-bit-rate", "8M", "--max-fps", "60", "--record",
            f"{os.path.join(video_path, 'screen')}.mkv"
        ]
        if serial:
            cmd.insert(0, "-s")
            cmd.insert(1, serial)
        self.__process = Terminal.cmd_connect(cmd)

        def stream(flow: Union[int, IO[str]] = subprocess.PIPE) -> None:
            for line in iter(flow.readline, ""):
                logger.info(" ".join(line.strip().split()))
            flow.close()

        if self.__process:
            threading.Thread(target=stream, args=(self.__process.stdout, )).start()
            threading.Thread(target=stream, args=(self.__process.stderr, )).start()
            await asyncio.sleep(1)

    async def start_record_silence(self, video_path: str, serial: str = None) -> None:
        cmd = [
            self.__initial, "--no-audio", "--video-bit-rate", "8M", "--max-fps", "60", "-Nr",
            f"{os.path.join(video_path, 'screen')}.mkv"
        ]
        if serial:
            cmd.insert(0, "-s")
            cmd.insert(1, serial)
        self.__process, input_task, error_task = Terminal.cmd_link(*cmd)
        if self.__process:
            await asyncio.sleep(1)

    async def stop_record(self) -> None:
        if sys.platform == "win32":
            self.__process.send_signal(signal.CTRL_C_EVENT)
        else:
            self.__process.terminate()
        self.__process = None


if __name__ == '__main__':
    pass
