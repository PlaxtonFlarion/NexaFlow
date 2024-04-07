import os
import pygame
from loguru import logger


class Player(object):

    @staticmethod
    def play_audio(audio_file: str, volume: float = 1.0):
        if not os.path.isfile(audio_file):
            return logger.error(f"{audio_file} 不是一个音频文件 ...")

        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play()
        logger.info(f"INFO: Playing audio {audio_file}")
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)


if __name__ == '__main__':
    pass
