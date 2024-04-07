import os

NAME = r"framix"
VERSION = r"0.1.0-beta"
URL = r"https://github.com/PlaxtonFlarion/NexaFlow"
AUTHOR = r"AceKeppel"
EMAIL = r"AceKeppel@outlook.com"
DESC = r"Framix"
LICENSE = r"MIT"

CHARSET = r"utf-8"
CUT_RESULT_FILE_NAME = r"cut_result.json"
REPORT_FILE_NAME = r"report.html"
BACKGROUND_COLOR = r"#fffaf4"
UNSTABLE_FLAG = r"-1"
UNKNOWN_STAGE_FLAG = r"-2"
IGNORE_FLAG = r"-3"
DEFAULT_THRESHOLD = 0.98

NEXA = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == '__main__':
    pass
