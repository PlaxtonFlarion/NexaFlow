#
#    ____                _
#   / ___|___  _ __  ___| |_
#  | |   / _ \| '_ \/ __| __|
#  | |__| (_) | | | \__ \ |_
#   \____\___/|_| |_|___/\__|
#

"""
版权所有 (c) 2024  Framix(画帧秀)
此文件受 Framix(画帧秀) 许可证的保护。您可以在 LICENSE.md 文件中查看详细的许可条款。

Copyright (c) 2024  Framix(画帧秀)
This file is licensed under the Framix(画帧秀) License. See the LICENSE.md file for more details.

# Copyright (c) 2024  Framix(画帧秀)
# このファイルは Framix(画帧秀) ライセンスの下でライセンスされています。詳細は LICENSE.md ファイルを参照してください。
"""


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

DECLARE = f"""\
[bold]^* {DESC} | {ALIAS} *^
(C) {YEAR} {DESC}. All rights reserved.
Version [bold #FFD75F]{VERSION}[/] - Licensed under the [bold #D7AFFF]{LICENSE}[/] License.
-----------------------------------------------
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

DF_SCALE = 0.5
DF_FRATE = 60

SHAPE = None
SCALE = None
START = None
CLOSE = None
LIMIT = None
GAUSS = None
GRIND = None
FRATE = None

BOOST = False
COLOR = False
BEGIN = (0, 1)
FINAL = (-1, -1)
THRES = 0.98
SHIFT = 3
SLIDE = 1
BLOCK = 3
SCOPE = 1
GRADE = 2
CROPS = []
OMITS = []
HOOKS = [{"x": 0, "y": 0, "x_size": 0, "y_size": 0}]

DB_FILES_NAME = f"{NAME}_data.db"
DB_TABLE_NAME = f"stocks"

R_TOTAL_TAG = f"FX"
R_COLLECTION = f"{DESC}_Collection"
R_RECOVERY = f"{DESC}_Recovery"
R_LOG_FILE = f"{NAME}.log"
R_TOTAL_HEAD = f"{DESC} Information"
R_TOTAL_NAME = f"{DESC}_Arkiv"
R_VIDEO_BASE_NAME = f"video"
R_FRAME_BASE_NAME = f"frame"
R_EXTRA_BASE_NAME = f"extra"

FAINT_MODEL = f"Keras_Gray_W256_H256"
COLOR_MODEL = f"Keras_Hued_W256_H256"

F_SCHEMATIC = f"schematic"
F_STRUCTURE = f"Structure"
F_SRC_OPERA_PLACE = f"{DESC}_Mix"
F_SRC_MODEL_PLACE = f"{DESC}_Model"
F_SRC_TOTAL_PLACE = f"{DESC}_Report"
F_OPTION = f"option.json"
F_DEPLOY = f"deploy.json"
F_SCRIPT = f"script.json"

NOTE_LEVEL = f"DEBUG"
SHOW_LEVEL = f"INFO"

WHILE_FORMAT = f"{{time:YYYY-MM-DD HH:mm:ss.SSS}} | <level>{{level: <8}}</level> | {{name}}:{{function}}:{{line}} - <level>{{message}}</level>"
PRINT_FORMAT = f"{DESC} | <level>{{level: <8}}</level> | <level>{{message}}</level>"
WRITE_FORMAT = f"<green>{{time:YYYY-MM-DD HH:mm:ss.SSS}}</green> | <level>{{level: <8}}</level> | <level>{{message}}</level>"


if __name__ == '__main__':
    pass
