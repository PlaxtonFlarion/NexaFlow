import os


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


if __name__ == '__main__':
    pass

