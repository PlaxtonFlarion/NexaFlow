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

    def __init__(self, start: int, end: int, cost: float, classifier=None):
        self.data = start, end, cost, classifier

    def __str__(self):
        start, end, cost, classifier = self.data
        kc = "KC" if classifier else "None"
        return f"<Review start={start} end={end} cost={cost} classifier={kc}>"

    __repr__ = __str__


if __name__ == '__main__':
    pass
