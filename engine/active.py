import sys
from loguru import logger
from nexaflow import const


class Active(object):

    @staticmethod
    def active(log_level: str):
        logger.remove(0)
        log_format = f"\033[1m{const.DESC}\033[0m | <level>{{level: <8}}</level> | <level>{{message}}</level>"
        logger.add(sys.stderr, format=log_format, level=log_level.upper())


class Review(object):

    __material: tuple = tuple()

    def __init__(self, start: int, end: int, cost: float, struct=None):
        self.material = start, end, cost, struct

    @property
    def material(self):
        return self.__material

    @material.setter
    def material(self, value):
        self.__material = value

    def __str__(self):
        start, end, cost, struct = self.material
        kc = f"KC" if struct else f"None"
        return f"<Review start={start} end={end} cost={cost} struct={kc}>"

    __repr__ = __str__


if __name__ == '__main__':
    pass
