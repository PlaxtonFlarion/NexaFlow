#
#    ____                _
#   / ___|___  _ __  ___| |_
#  | |   / _ \| '_ \/ __| __|
#  | |__| (_) | | | \__ \ |_
#   \____\___/|_| |_|___/\__|
#

import os

ITEM = r"NexaFlow"
NAME = r"framix"
YEAR = r"2024"
VERSION = r"1.0.0"
URL = r"https://github.com/PlaxtonFlarion/NexaFlow"
AUTHOR = r"AceKeppel"
EMAIL = r"AceKeppel@outlook.com"
DESC = r"Framix"
ALIAS = r"画帧秀"
LICENSE = r"MIT"

DECLARE = f"""[bold]^* {DESC} ({ALIAS}) *^
(C) {YEAR} {DESC}. All rights reserved.
Version [bold #FFD75F]{VERSION}[/] - Licensed under the [bold #FF00FF]{LICENSE}[/] License.[/]
"""

CHARSET = r"UTF-8"
CUT_RESULT_FILE_NAME = r"cut_result.json"
REPORT_FILE_NAME = r"report.html"
BACKGROUND_COLOR = r"#FFFAF4"
UNSTABLE_FLAG = r"-1"
UNKNOWN_STAGE_FLAG = r"-2"
IGNORE_FLAG = r"-3"

SUC = f"[bold #FFFFFF on #32CD32]"
WRN = f"[bold #000000 on #FFFF00]"
ERR = f"[bold #FFFFFF on #FF6347]"

DEFAULT_SCALE = 0.5

SHAPE = None
SCALE = None
START = None
CLOSE = None
LIMIT = None
GAUSS = None
GRIND = None
FRATE = 60

BOOST = False
COLOR = False
BEGIN = (0, 1)
FINAL = (-1, -1)
THRES = 0.98
SHIFT = 3
BLOCK = 3
SCOPE = 1
GRADE = 2
CROPS = []
OMITS = []
HOOKS = [{"x": 0, "y": 0, "x_size": 0, "y_size": 0}]

DB_FILES_NAME = f"{NAME}_data.db"
DB_TABLE_NAME = f"stocks"

R_TOTAL_TAG = f"Nexa"
R_UNION_TAG = f"Union_Nexa"
R_COLLECTION = f"Nexa_Collection"
R_RECOVERY = f"Nexa_Recovery"
R_LOG_NOTE = f"Recovery"
R_LOG_FILE = f"nexaflow.log"
R_TOTAL_HEAD = f"{DESC} Information"
R_TOTAL_FILE = f"Nexa_Flow.html"
R_UNION_FILE = f"Union_Nexa_Flow.html"
R_VIDEO_BASE_NAME = f"video"
R_FRAME_BASE_NAME = f"frame"
R_EXTRA_BASE_NAME = f"extra"

FAINT_MODEL = f"Keras_Gray_W256_H256"
COLOR_MODEL = f"Keras_Hued_W256_H256"

F_SCHEMATIC = f"schematic"
F_SPECIALLY = f"Specially"
F_SRC_OPERA_PLACE = f"{DESC}_Mix"
F_SRC_MODEL_PLACE = f"{DESC}_Model"
F_SRC_TOTAL_PLACE = f"{DESC}_Report"
F_OPTION = f"option.json"
F_DEPLOY = f"deploy.json"
F_SCRIPT = f"script.json"

NOTE_LEVEL = f"DEBUG"
SHOW_LEVEL = f"INFO"

WHILE_FORMAT = f"[bold]{{time:YYYY-MM-DD HH:mm:ss.SSS}} | <level>{{level: <8}}</level> | {{name}}:{{function}}:{{line}} - <level>{{message}}</level>"
PRINT_FORMAT = f"[bold]{DESC} | <level>{{level: <8}}</level> | <level>{{message}}</level>"
WRITE_FORMAT = f"<green>{{time:YYYY-MM-DD HH:mm:ss.SSS}}</green> | <level>{{level: <8}}</level> | <level>{{message}}</level>"

NEXA = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEMPLATE_ATOM_TOTAL = os.path.join(NEXA, f"templates", f"template_atom_total.html")
TEMPLATE_VIEW_TOTAL = os.path.join(NEXA, f"templates", f"template_view_total.html")
TEMPLATE_MAIN_TOTAL = os.path.join(NEXA, f"templates", f"template_main_total.html")
TEMPLATE_VIEW_SHARE = os.path.join(NEXA, f"templates", f"template_view_share.html")
TEMPLATE_MAIN_SHARE = os.path.join(NEXA, f"templates", f"template_main_share.html")


if __name__ == '__main__':
    pass
