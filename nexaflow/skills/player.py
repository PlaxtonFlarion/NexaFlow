import os
import re
import pygame
from loguru import logger
from typing import Tuple, List


class Player(object):

    def __init__(self):
        pygame.mixer.init()

    @staticmethod
    def load_all_audio(audio_dirs: str) -> List[Tuple[str, str]]:
        audio_list = []
        for audio_file in os.listdir(audio_dirs):
            if ".mp3" in audio_file or ".wav" in audio_file:
                if match := re.search(r".*?(?=\.)", audio_file):
                    audio_list.append(
                        (match.group(), os.path.join(audio_dirs, audio_file))
                    )
        return audio_list

    @staticmethod
    def load_audio(audio_dirs: str, audio_name: str) -> Tuple[str, str]:
        query, audio = "", ""
        for audio_file in os.listdir(audio_dirs):
            if audio_name in audio_file:
                if match := re.search(r".*?(?=\.)", audio_file):
                    query = match.group()
                    audio = os.path.join(audio_dirs, audio_file)
        return query, audio

    @staticmethod
    def play_audio(audio_file: str, volume: float = 1.0):
        if os.path.isfile(audio_file):
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.set_volume(volume)
            pygame.mixer.music.play()
            logger.info(f"INFO: Playing audio {audio_file}")
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        else:
            logger.error(f"{audio_file} 不是一个音频文件 ...")


if __name__ == '__main__':
    pass
