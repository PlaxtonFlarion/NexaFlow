#    ____                _
#   / ___|___  _ __  ___| |_
#  | |   / _ \| '_ \/ __| __|
#  | |__| (_) | | | \__ \ |_
#   \____\___/|_| |_|___/\__|
#

# ==== Notes: 版权申明 ====
# 版权所有 (c) 2024  Framix(画帧秀)
# 此文件受 Framix(画帧秀) 许可证的保护。您可以在 LICENSE.md 文件中查看详细的许可条款。

# ==== Notes: License ====
# Copyright (c) 2024  Framix(画帧秀)
# This file is licensed under the Framix(画帧秀) License. See the LICENSE.md file for more details.

# ==== Notes: ライセンス ====
# Copyright (c) 2024  Framix(画帧秀)
# このファイルは Framix(画帧秀) ライセンスの下でライセンスされています。詳細は LICENSE.md ファイルを参照してください。

# ======== 【基础信息配置 | Core Metadata】========
ITEM             = r"NexaFlow"
NAME             = r"framix"
DESC             = r"Framix"
ALIAS            = r"画帧秀"
VERSION          = r"1.0.0"
YEAR             = r"2024"
LICENSE          = r"Proprietary License"
URL              = r"https://github.com/PlaxtonFlarion/NexaFlow"
AUTHOR           = r"AceKeppel"
EMAIL            = r"AceKeppel@outlook.com"

PUBLISHER        = f"{DESC} Technologies Inc."
COPYRIGHT        = f"Copyright (C) {YEAR} {DESC}. All rights reserved."
WIN_FILE_VERSION = r"1.0.0.0"

DECLARE = f"""\
[bold][bold #00D7AF]>>> {DESC} :: {ALIAS} <<<[/]
[bold #FF8787]Copyright (C)[/] {YEAR} {DESC}. All rights reserved.
Version [bold #FFD75F]{VERSION}[/] :: Licensed software. Authorization required.
{'-' * 59}
"""

# ======== 【路径与文件 | File & Path】========
CHARSET              = r"UTF-8"
CUT_RESULT_FILE_NAME = r"cut_result.json"
REPORT_FILE_NAME     = r"report.html"
BACKGROUND_COLOR     = r"#FFFAF4"

F_SCHEMATIC          = r"schematic"
F_STRUCTURE          = r"Structure"
F_SRC_OPERA_PLACE    = f"{DESC}_Mix"
F_SRC_MODEL_PLACE    = f"{DESC}_Model"
F_SRC_TOTAL_PLACE    = f"{DESC}_Report"
F_OPTION             = f"{NAME}_option.json"
F_DEPLOY             = f"{NAME}_deploy.json"
F_SCRIPT             = f"{NAME}_script.json"
LIC_FILE             = f"{NAME}_signature.lic"

DB_FILES_NAME        = f"{NAME}_data.db"
DB_TABLE_NAME        = r"stocks"

# ======== 【模型配置 | MOD】========
FAINT_MODEL = r"Keras_Gray_W256_H256"
COLOR_MODEL = r"Keras_Hued_W256_H256"

# ======== 【图帧先行 | FST】========
DF_SCALE = 0.5
DF_FRATE = 60

SHAPE    = None
SCALE    = None
START    = None
CLOSE    = None
LIMIT    = None
GAUSS    = None
GRIND    = None
FRATE    = None

# ======== 【智析引擎 | ALS】========
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

# ======== 【标记常量 | Flags】========
UNSTABLE_FLAG      = r"-1"
UNKNOWN_STAGE_FLAG = r"-2"
IGNORE_FLAG        = r"-3"

# ======== 【记录与导出 | Output Metadata】========
R_TOTAL_TAG       = r"FX"
R_COLLECTION      = f"{DESC}_Collection"
R_RECOVERY        = f"{DESC}_Recovery"
R_LOG_FILE        = f"{NAME}.log"
R_TOTAL_HEAD      = f"{DESC} Information"
R_TOTAL_NAME      = f"{DESC}_Arkiv"
R_VIDEO_BASE_NAME = r"video"
R_FRAME_BASE_NAME = r"frame"
R_EXTRA_BASE_NAME = r"extra"

# ======== 【日志样式 | Logging Style】========
SUC          = f"[bold #FFFFFF on #32CD32]"
WRN          = f"[bold #000000 on #FFFF00]"
ERR          = f"[bold #FFFFFF on #FF6347]"

NOTE_LEVEL   = r"DEBUG"
SHOW_LEVEL   = r"INFO"

PRINT_HEAD   = f"[bold #EEEEEE]{DESC} ::[/]"
OTHER_HEAD   = f"{DESC} ::"
ADAPT_HEAD   = f"{DESC} :"

PRINT_FORMAT = f"<level>{{level: <8}}</level> | <level>{{message}}</level>"
WRITE_FORMAT = f"{OTHER_HEAD} <green>{{time:YYYY-MM-DD HH:mm:ss.SSS}}</green> | <level>{{level: <8}}</level> | <level>{{message}}</level>"
WHILE_FORMAT = f"{OTHER_HEAD} <green>{{time:YYYY-MM-DD HH:mm:ss.SSS}}</green> | <level>{{level: <8}}</level> | {{name}}:{{function}}:{{line}} - <level>{{message}}</level>"


if __name__ == '__main__':
    pass
