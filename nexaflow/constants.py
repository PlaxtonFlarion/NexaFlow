import os
import sys
from loguru import logger


CHARSET = r"utf-8"

CUT_RESULT_FILE_NAME = r"cut_result.json"
REPORT_FILE_NAME = r"report.html"

BACKGROUND_COLOR = r"#fffaf4"
UNSTABLE_FLAG = r"-1"
UNKNOWN_STAGE_FLAG = r"-2"
IGNORE_FLAG = r"-3"

DEFAULT_THRESHOLD = 0.98


class Constants(object):

    NEXA = os.path.dirname(os.path.abspath(__file__))
    WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FORMAT: str = "| <level>{level: <8}</level> | <level>{message}</level>"

    @classmethod
    def initial_logger(cls, log_level: str = "INFO"):
        logger.remove(0)
        logger.add(sys.stderr, format=cls.FORMAT, level=log_level.upper())


if __name__ == '__main__':
    pass

