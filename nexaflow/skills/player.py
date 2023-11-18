import os
import re
import pygame
from loguru import logger
from typing import Tuple, List
from nexaflow.constants import Constants

AUDIO_DIRS: str = os.path.join(Constants.WORK, "audio")


class Player(object):

    def __init__(self):
        pygame.mixer.init()
        if not os.path.exists(AUDIO_DIRS):
            os.makedirs(AUDIO_DIRS, exist_ok=True)

    @staticmethod
    def load_all_audio() -> List[Tuple[str, str]]:
        audio_list = []
        for audio_file in os.listdir(AUDIO_DIRS):
            if ".mp3" in audio_file or ".wav" in audio_file:
                if match := re.search(r".*?(?=\.)", audio_file):
                    audio_list.append(
                        (match.group(), os.path.join(AUDIO_DIRS, audio_file))
                    )
        return audio_list

    @staticmethod
    def load_audio(audio_name: str) -> Tuple[str, str]:
        query, audio = "", ""
        for audio_file in os.listdir(AUDIO_DIRS):
            if audio_name in audio_file:
                if match := re.search(r".*?(?=\.)", audio_file):
                    query = match.group()
                    audio = os.path.join(AUDIO_DIRS, audio_file)
        return query, audio

    @staticmethod
    def play_audio(audio: str, volume: float = 1.0):
        if os.path.isfile(audio):
            pygame.mixer.music.load(audio)
            pygame.mixer.music.set_volume(volume)
            pygame.mixer.music.play()
            logger.info(f"INFO: Playing audio {audio}")
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        else:
            logger.error(f"{audio} 不是一个音频文件 ...")


if __name__ == '__main__':
    pass
