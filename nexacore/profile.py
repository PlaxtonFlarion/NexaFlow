#   ____             __ _ _
#  |  _ \ _ __ ___  / _(_) | ___
#  | |_) | '__/ _ \| |_| | |/ _ \
#  |  __/| | | (_) |  _| | |  __/
#  |_|   |_|  \___/|_| |_|_|\___|
#
# ==== Notes: License ====
# Copyright (c) 2024  Framix :: 画帧秀
# This file is licensed under the Framix :: 画帧秀 License. See the LICENSE.md file for more details.

import os
import copy
import json
import typing
from pathlib import Path
from loguru import logger
from rich.table import Table
from engine.tinker import FileAssist
from nexacore.parser import Parser
from nexacore.design import Design
from nexacore.argument import Args
from nexaflow import const


class Deploy(object):
    """
    参数部署与配置管理器类。

    该类负责解析、加载、设置与展示命令行工具中的参数配置。
    它支持两大参数组：FST（基础设置）与 ALS（分析设置），通过动态解析器
    `Parser` 实现参数合法性验证，并提供序列化与表格化展示功能。

    Attributes
    ----------
    deploys : dict
        包含所有默认部署参数的嵌套字典，分为 FST 与 ALS 两大组。
    """

    deploys = {
        "FST": {
            "shape": const.SHAPE,
            "scale": const.SCALE,
            "start": const.START,
            "close": const.CLOSE,
            "limit": const.LIMIT,
            "gauss": const.GAUSS,
            "grind": const.GRIND,
            "frate": const.FRATE
        },
        "ALS": {
            "boost": const.BOOST,
            "color": const.COLOR,
            "begin": const.BEGIN,
            "final": const.FINAL,
            "thres": const.THRES,
            "shift": const.SHIFT,
            "slide": const.SLIDE,
            "block": const.BLOCK,
            "scope": const.SCOPE,
            "grade": const.GRADE,
            "crops": const.CROPS,
            "omits": const.OMITS
        }
    }

    def __init__(self, deploy_file: str):
        self.deploy_file = deploy_file

    def __getstate__(self):
        return self.deploys

    def __setstate__(self, state):
        self.deploys = state

    # Notes: ======================== FST Getter ========================

    @property
    def shape(self):
        return self.deploys["FST"]["shape"]

    @property
    def scale(self):
        return self.deploys["FST"]["scale"]

    @property
    def start(self):
        return self.deploys["FST"]["start"]

    @property
    def close(self):
        return self.deploys["FST"]["close"]

    @property
    def limit(self):
        return self.deploys["FST"]["limit"]

    @property
    def gauss(self):
        return self.deploys["FST"]["gauss"]

    @property
    def grind(self):
        return self.deploys["FST"]["grind"]

    @property
    def frate(self):
        return self.deploys["FST"]["frate"]

    # Notes: ======================== ALS Getter ========================

    @property
    def boost(self):
        return self.deploys["ALS"]["boost"]

    @property
    def color(self):
        return self.deploys["ALS"]["color"]

    @property
    def begin(self):
        return self.deploys["ALS"]["begin"]

    @property
    def final(self):
        return self.deploys["ALS"]["final"]

    @property
    def thres(self):
        return self.deploys["ALS"]["thres"]

    @property
    def shift(self):
        return self.deploys["ALS"]["shift"]

    @property
    def slide(self):
        return self.deploys["ALS"]["slide"]

    @property
    def block(self):
        return self.deploys["ALS"]["block"]

    @property
    def scope(self):
        return self.deploys["ALS"]["scope"]

    @property
    def grade(self):
        return self.deploys["ALS"]["grade"]

    @property
    def crops(self):
        return self.deploys["ALS"]["crops"]

    @property
    def omits(self):
        return self.deploys["ALS"]["omits"]

    # Notes: ======================== FST Setter ========================

    @shape.setter
    def shape(self, value: typing.Any):
        self.deploys["FST"]["shape"] = Parser.parse_shape(value)

    @scale.setter
    def scale(self, value: typing.Any):
        # Note 取值范围 0.1 ～ 1.0
        self.deploys["FST"]["scale"] = Parser.parse_waves(
            value, min_v=0.1, max_v=1.0, decimal=1
        )

    @start.setter
    def start(self, value: typing.Any):
        self.deploys["FST"]["start"] = Parser.parse_times(value)

    @close.setter
    def close(self, value: typing.Any):
        self.deploys["FST"]["close"] = Parser.parse_times(value)

    @limit.setter
    def limit(self, value: typing.Any):
        self.deploys["FST"]["limit"] = Parser.parse_times(value)

    @gauss.setter
    def gauss(self, value: typing.Any):
        # Note 取值范围 0.0 ～ 2.0
        self.deploys["FST"]["gauss"] = Parser.parse_waves(
            value, min_v=0.0, max_v=2.0, decimal=1
        )

    @grind.setter
    def grind(self, value: typing.Any):
        # Note 取值范围 0.0 ～ 2.0
        self.deploys["FST"]["grind"] = Parser.parse_waves(
            value, min_v=0.0, max_v=2.0, decimal=1
        )

    @frate.setter
    def frate(self, value: typing.Any):
        # Note 取值范围 1 ～ unlimited
        self.deploys["FST"]["frate"] = Parser.parse_waves(
            value, min_v=1, decimal=0
        )

    # Notes: ======================== ALS Setter ========================

    @boost.setter
    def boost(self, value: typing.Any):
        if isinstance(value, bool):
            self.deploys["ALS"]["boost"] = value

    @color.setter
    def color(self, value: typing.Any):
        if isinstance(value, bool):
            self.deploys["ALS"]["color"] = value

    @begin.setter
    def begin(self, value: typing.Any):
        if effective := Parser.parse_stage(value):
            self.deploys["ALS"]["begin"] = effective

    @final.setter
    def final(self, value: typing.Any):
        if effective := Parser.parse_stage(value):
            self.deploys["ALS"]["final"] = effective

    @thres.setter
    def thres(self, value: typing.Any):
        # Note 取值范围 0.85 ～ 1.00
        if effective := Parser.parse_waves(value, min_v=0.85, max_v=1.00, decimal=2):
            self.deploys["ALS"]["thres"] = effective

    @shift.setter
    def shift(self, value: typing.Any):
        # Note 取值范围 0 ～ 15
        if effective := Parser.parse_waves(value, min_v=0, max_v=15, decimal=0):
            self.deploys["ALS"]["shift"] = effective

    @slide.setter
    def slide(self, value: typing.Any):
        # Note 取值范围 1 ～ 10
        if effective := Parser.parse_waves(value, min_v=1, max_v=10, decimal=0):
            self.deploys["ALS"]["slide"] = effective

    @block.setter
    def block(self, value: typing.Any):
        # Note 取值范围 1 ～ 10
        if effective := Parser.parse_waves(value, min_v=1, max_v=10, decimal=0):
            self.deploys["ALS"]["block"] = effective

    @scope.setter
    def scope(self, value: typing.Any):
        # Note 取值范围 1 ～ 20
        if effective := Parser.parse_waves(value, min_v=1, max_v=20, decimal=0):
            self.deploys["ALS"]["scope"] = effective

    @grade.setter
    def grade(self, value: typing.Any):
        # Note 取值范围 1 ～ 5
        if effective := Parser.parse_waves(value, min_v=1, max_v=5, decimal=0):
            self.deploys["ALS"]["grade"] = effective

    @crops.setter
    def crops(self, value: typing.Any):
        self.deploys["ALS"]["crops"] = Parser.parse_hooks(value)

    @omits.setter
    def omits(self, value: typing.Any):
        self.deploys["ALS"]["omits"] = Parser.parse_hooks(value)

    # Notes: ======================== Method ========================

    async def dump_fabric(self) -> None:
        """
        将当前部署参数写入 JSON 文件中，作为配置持久化存储。
        """
        deep_copy_deploys = copy.deepcopy(self.deploys)
        for attr in ["crops", "omits"]:
            if len(deep_copy_deploys["ALS"][attr]) == 0:
                deep_copy_deploys["ALS"][attr] = const.HOOKS

        os.makedirs(os.path.dirname(self.deploy_file), exist_ok=True)
        await FileAssist.dump_parameters(self.deploy_file, deep_copy_deploys)

    async def load_fabric(self) -> None:
        """
        加载部署配置文件并初始化参数值。
        """
        try:
            parameters = await FileAssist.load_parameters(self.deploy_file)
            for key, value in self.deploys.items():
                for k, v in value.items():
                    setattr(self, k, parameters.get(key, {}).get(k, v))
                    logger.debug(f"Load <{k}> = {v} -> {getattr(self, k)}")
        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            logger.debug(f"使用默认参数: {e}")
            logger.debug(f"生成部署文件: {self.deploy_file}")
            await self.dump_fabric()
        except Exception as e:
            logger.debug(f"未知错误 {e}")

    async def view_fabric(self) -> None:
        """
        以表格形式在控制台展示当前部署参数的详细信息。
        """
        deploys_group = {**Args.GROUP_FIRST, **Args.GROUP_EXTRA}

        table_style = {
            "title_justify": "center", "show_header": True, "show_lines": True
        }

        for key, value in self.deploys.items():
            table = Table(
                title=f"[bold #87CEFA]{const.DESC} | {const.ALIAS} Deploys {key}",
                header_style=f"bold #B0C4DE", **table_style
            )
            table.add_column("配置", no_wrap=True, width=8)
            table.add_column("参数", no_wrap=True, width=14)
            table.add_column("范围", no_wrap=True, width=12)
            table.add_column("效果", no_wrap=True, min_width=32)

            information = [
                [f"[bold #D75F87]{v['help']}"] + v["push"](self, Parser)
                for k, v in deploys_group.items() if k.lstrip("--") in value
            ]

            for info in information:
                table.add_row(*info)
            Design.console.print(table)

        for hook_name, hook_list in [("Paint Crop Hook", self.crops), ("Paint Omit Hook", self.omits)]:
            if hook_list:
                table = Table(
                    title=f"[bold #5FAFFF]{const.DESC} | {const.ALIAS} {hook_name}",
                    header_style="bold #AFAFD7", **table_style
                )
                table.add_column("编号", justify="left", width=4)
                table.add_column("区域", justify="left", width=68)

                for index, hook in enumerate(hook_list, start=1):
                    table.add_row(f"[bold #B2B2B2]{index:02}", f"[bold #87AFFF]{hook}")
                Design.console.print(table)


class Option(object):
    """
    用于加载和保存与模型路径和配置信息相关的全局参数设置。

    该类负责从本地配置文件中读取或写入包含模型位置和模型名称的参数。
    它提供了属性访问器（property）来安全管理这些字段，并进行路径有效性校验。

    Attributes
    ----------
    options : dict
        包含配置字段的默认字典，字段包括：
        - "total_place"：总体输出目录；
        - "model_place"：模型目录；
        - "faint_model"：灰度检测模型名称；
        - "color_model"：彩色检测模型名称。
    """

    options = {
        "total_place": "",  # 报告文件夹路径
        "model_place": "",  # 模型文件夹路径
        "faint_model": "",  # 灰度模型名称
        "color_model": "",  # 彩色模型名称
    }

    def __init__(self, option_file: typing.Any):
        self.option_file = option_file

    def __getstate__(self):
        return self.options

    def __setstate__(self, state):
        self.options = state

    # Notes: ======================== Getter ========================

    @property
    def total_place(self):
        return self.options["total_place"]

    @property
    def model_place(self):
        return self.options["model_place"]

    @property
    def faint_model(self):
        return self.options["faint_model"]

    @property
    def color_model(self):
        return self.options["color_model"]

    # Notes: ======================== Setter ========================

    @total_place.setter
    def total_place(self, value: typing.Any):
        if value and type(value) is str and Path(value).is_dir():
            self.options["total_place"] = value

    @model_place.setter
    def model_place(self, value: typing.Any):
        if value and type(value) is str and Path(value).is_dir():
            self.options["model_place"] = value

    @faint_model.setter
    def faint_model(self, value: typing.Any):
        if value and type(value) is str and Path(self.model_place, value).exists():
            self.options["faint_model"] = value

    @color_model.setter
    def color_model(self, value: typing.Any):
        if value and type(value) is str and Path(self.model_place, value).exists():
            self.options["color_model"] = value

    # Notes: ======================== Method ========================

    async def dump_fabric(self) -> None:
        """
        将当前选项配置以 JSON 格式写入指定文件路径。
        """
        os.makedirs(os.path.dirname(self.option_file), exist_ok=True)
        await FileAssist.dump_parameters(self.option_file, self.options)

    async def load_fabric(self) -> None:
        """
        从指定路径加载配置参数，并更新当前选项状态。
        """
        try:
            parameters = await FileAssist.load_parameters(self.option_file)
            for k, v in self.options.items():
                setattr(self, k, parameters.get(k, v))
                logger.debug(f"Load <{k}> = {v} -> {getattr(self, k)}")
        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            logger.debug(f"使用默认参数: {e}")
            logger.debug(f"生成配置文件: {self.option_file}")
            await self.dump_fabric()
        except Exception as e:
            logger.debug(f"未知错误 {e}")


class Script(object):
    """
    Script 类用于管理自动化流程中的脚本文件初始化与结构生成。
    """

    scripts = {
        "command": [
            {
                "ID-X": {
                    "parser": {"FST": {"scale": 0.3}, "ALS": {"boost": True}},
                    "header": ["script"],
                    "change": [],
                    "looper": 1,
                    "prefix": [
                        {"cmds": [], "vals": []}, {"cmds": [], "vals": []}
                    ],
                    "action": [
                        {"cmds": [], "vals": []}, {"cmds": [], "vals": []}
                    ],
                    "suffix": [
                        {"cmds": [], "vals": []}, {"cmds": [], "vals": []}
                    ]
                }
            }
        ]
    }

    def __init__(self, script_file: typing.Any):
        self.script_file = script_file

    async def dump_fabric(self) -> None:
        """
        生成默认脚本结构并写入指定文件路径，用于初始化脚本模板。
        """
        os.makedirs(os.path.dirname(self.script_file), exist_ok=True)
        await FileAssist.dump_parameters(self.script_file, self.scripts)


if __name__ == '__main__':
    pass
