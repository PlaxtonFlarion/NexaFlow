import sys
from loguru import logger


def initialization(log_level: str):
    logger.remove(0)
    log_format = "| <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sys.stderr, format=log_format, level=log_level.upper())


if __name__ == '__main__':
    pass
