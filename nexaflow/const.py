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

FRATE = 60
THRES = 0.97
SHIFT = 3
BLOCK = 6

NEXA = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AUDIO = os.path.join(WORK, "audio")
ARRAY = os.path.join(WORK, "data")
CREDO = os.path.join(WORK, "report")
MODEL = os.path.join(WORK, "archivix", "molds", "Keras_Gray_W256_H256_00000.h5")
TEMPLATE = os.path.join(NEXA, "template")
TEMPLATE_ATOM_TOTAL = os.path.join(NEXA, "template", "template_atom_total.html")
TEMPLATE_MAIN_TOTAL = os.path.join(NEXA, "template", "template_main_total.html")
TEMPLATE_MAIN = os.path.join(NEXA, "template", "template_main.html")


if __name__ == '__main__':
    pass
