from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from nexaflow import const


class RichSink(RichHandler):

    def __init__(self, console: "Console"):
        super().__init__(console=console, rich_tracebacks=True, show_path=False, show_time=False)

    def emit(self, record):
        log_message = self.format(record)
        self.console.print(log_message)


class Active(object):

    @staticmethod
    def active(log_level: str):
        logger.remove(0)
        log_format = f"[bold]{const.DESC} | <level>{{level: <8}}</level> | <level>{{message}}</level>"
        logger.add(RichSink(Console()), format=log_format, level=log_level.upper(), diagnose=False)


class Review(object):

    __material: tuple = tuple()

    def __init__(self, *args):
        self.material = args

    @property
    def material(self):
        return self.__material

    @material.setter
    def material(self, value):
        self.__material = value

    def __str__(self):
        start, end, cost, *_ = self.material
        return f"<Review start={start} end={end} cost={cost}>"

    __repr__ = __str__


class FramixError(Exception):
    pass


class FramixAnalysisError(FramixError):

    def __init__(self, msg):
        self.msg = msg


class FramixAnalyzerError(FramixError):

    def __init__(self, msg):
        self.msg = msg


class FramixReporterError(FramixError):

    def __init__(self, msg):
        self.msg = msg


if __name__ == '__main__':
    pass
