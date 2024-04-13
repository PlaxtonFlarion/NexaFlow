import os
import time
import signal
import pygame
import threading
from loguru import logger
from subprocess import Popen
from typing import Union, IO, Optional
from engine.terminal import Terminal


class Medias(object):

    def __init__(self):
        self.__volume: float = 1.0
        self.__connection: Optional[Popen] = None
        self.__record_event: threading.Event = threading.Event()

    def audio_player(self, audio_file: str):
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.set_volume(self.__volume)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    def start_record(self, video_path: str, serial: str = None) -> None:
        cmd = [
            "scrcpy", "--no-audio", "--video-bit-rate", "8M", "--max-fps", "60", "-Nr",
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

    def close_record(self) -> None:
        self.__connection.send_signal(signal.CTRL_C_EVENT)
        self.__record_event.clear()
        self.__connection = None

        try:
            Terminal.cmd_oneshot(["taskkill", "/im", "scrcpy.exe"])
        except KeyboardInterrupt:
            logger.info("Stop with Ctrl_C_Event ...")


if __name__ == '__main__':
    pass
