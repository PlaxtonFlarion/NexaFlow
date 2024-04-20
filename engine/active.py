import sys
from loguru import logger
from nexaflow import const


class Active(object):

    @staticmethod
    def active(log_level: str):
        logger.remove(0)
        log_format = fr"{const.DESC} | <level>{{level: <8}}</level> | <level>{{message}}</level>"
        logger.add(sys.stderr, format=log_format, level=log_level.upper())


class Review(object):

    data = tuple()

    def __init__(self, start: int, end: int, cost: float, struct=None):
        self.data = start, end, cost, struct

    def __str__(self):
        start, end, cost, struct = self.data
        kc = "KC" if struct else "None"
        return f"<Review start={start} end={end} cost={cost} struct={kc}>"

    __repr__ = __str__


if __name__ == '__main__':
    pass
