from rich.console import Console
from rich.logging import RichHandler
from loguru import logger
from nexaflow import const


class RichSink(RichHandler):

    def __init__(self, console):
        super().__init__(console=console, rich_tracebacks=True, show_path=False, show_time=False)

    def emit(self, record):
        log_message = self.format(record)
        self.console.print(log_message)


class Active(object):

    console = Console()

    @staticmethod
    def active(log_level: str):
        logger.remove(0)
        log_format = f"[bold]{const.DESC} | <level>{{level: <8}}</level> | <level>{{message}}</level>"
        logger.add(RichSink(Active.console), format=log_format, level=log_level.upper(), diagnose=False)


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
