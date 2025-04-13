#
#   ____             __ _ _
#  |  _ \ _ __ ___  / _(_) | ___
#  | |_) | '__/ _ \| |_| | |/ _ \
#  |  __/| | | (_) |  _| | |  __/
#  |_|   |_|  \___/|_| |_|_|\___|
#

"""
版权所有 (c) 2024  Framix(画帧秀)
此文件受 Framix(画帧秀) 许可证的保护。您可以在 LICENSE.md 文件中查看详细的许可条款。

Copyright (c) 2024  Framix(画帧秀)
This file is licensed under the Framix(画帧秀) License. See the LICENSE.md file for more details.

# Copyright (c) 2024  Framix(画帧秀)
# このファイルは Framix(画帧秀) ライセンスの下でライセンスされています。詳細は LICENSE.md ファイルを参照してください。
"""

import os
import copy
import json
import typing
from loguru import logger
from rich.table import Table
from frameflow.skills.parser import Parser
from frameflow.skills.design import Design
from frameflow.argument import Args
from nexaflow import const


def dump_parameters(src: typing.Any, dst: dict) -> None:
    """
    将配置参数以 JSON 格式写入指定路径的文件。

    Parameters
    ----------
    src : typing.Any
        目标文件路径，可为字符串或路径对象，用于存储配置数据。

    dst : dict
        待写入的配置数据字典。

    Returns
    -------
    None
        不返回任何值，写入完成后直接关闭文件。

    Notes
    -----
    - 使用 UTF-8 编码（或由 `const.CHARSET` 指定）进行文件写入；
    - 使用缩进和格式控制以确保 JSON 可读性（indent=4, ensure_ascii=False）；
    - 本函数适用于保存部署参数、应用配置、模型元数据等场景。
    """
    with open(src, "w", encoding=const.CHARSET) as file:
        json.dump(dst, file, indent=4, separators=(",", ":"), ensure_ascii=False)


def load_parameters(src: typing.Any) -> typing.Any:
    """
    从指定路径读取 JSON 文件内容并解析为 Python 字典对象。

    Parameters
    ----------
    src : typing.Any
        JSON 文件路径，支持字符串或路径对象。

    Returns
    -------
    typing.Any
        返回解析后的配置对象（通常为 `dict` 类型）。

    Notes
    -----
    - 使用 UTF-8 编码（或由 `const.CHARSET` 指定）读取 JSON 文件；
    - 若文件内容不符合 JSON 语法，将抛出 `json.decoder.JSONDecodeError`；
    - 可用于恢复用户参数配置、加载模型元信息、初始化系统设置等。
    """
    with open(src, "r", encoding=const.CHARSET) as file:
        return json.load(file)


class Deploy(object):
    """
    参数部署与配置管理器类。

    该类负责解析、加载、设置与展示 Framix 命令行工具中的参数配置。
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
        self.load_deploy(deploy_file)

    def __getstate__(self):
        return self.deploys

    def __setstate__(self, state):
        self.deploys = state

# Getter ###############################################################################################################

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

# Getter ###############################################################################################################

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

# Setter ###############################################################################################################

    @shape.setter
    def shape(self, value: typing.Any):
        if effective := Parser.parse_shape(value):
            self.deploys["FST"]["shape"] = effective

    @scale.setter
    def scale(self, value: typing.Any):
        # Note 取值范围 0.1 ～ 1.0
        if effective := Parser.parse_waves(value, min_v=0.1, max_v=1.0, decimal=1):
            self.deploys["FST"]["scale"] = effective

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
        if effective := Parser.parse_waves(value, min_v=0.0, max_v=2.0, decimal=1):
            self.deploys["FST"]["gauss"] = effective

    @grind.setter
    def grind(self, value: typing.Any):
        # Note 取值范围 0.0 ～ 2.0
        if effective := Parser.parse_waves(value, min_v=0.0, max_v=2.0, decimal=1):
            self.deploys["FST"]["grind"] = effective

    @frate.setter
    def frate(self, value: typing.Any):
        # Note 取值范围 30 ～ 60
        if effective := Parser.parse_waves(value, min_v=30, max_v=60, decimal=0):
            self.deploys["FST"]["frate"] = effective

# Setter ###############################################################################################################

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

    def dump_deploy(self, deploy_file: typing.Any) -> None:
        """
        将当前部署参数写入 JSON 文件中，作为配置持久化存储。

        Parameters
        ----------
        deploy_file : Any
            输出文件路径，可以是字符串路径或类文件对象。
            最终将把 deploy 配置以 JSON 格式保存至该路径。

        Returns
        -------
        None
            不返回任何值，但会在磁盘写入部署配置文件。

        Notes
        -----
        - 在写入前，会对 `deploys` 数据结构做深拷贝，防止原始配置被污染；
        - 如果 `crops` 或 `omits` 为空，则会填充默认的 `const.HOOKS`；
        - 会自动创建目标路径所需的目录结构；
        - 最终通过 `dump_parameters()` 将部署数据写入文件。

        Workflow
        --------
        1. 对当前 `deploys` 结构进行深拷贝；
        2. 检查并修复 `crops` 和 `omits` 字段；
        3. 创建部署文件所在目录（如不存在）；
        4. 调用 `dump_parameters()` 保存为 JSON 文件。
        """
        deep_copy_deploys = copy.deepcopy(self.deploys)
        for attr in ["crops", "omits"]:
            if len(deep_copy_deploys["ALS"][attr]) == 0:
                deep_copy_deploys["ALS"][attr] = const.HOOKS

        os.makedirs(os.path.dirname(deploy_file), exist_ok=True)
        dump_parameters(deploy_file, deep_copy_deploys)

    def load_deploy(self, deploy_file: typing.Any) -> None:
        """
        加载部署配置文件并初始化参数值。

        Parameters
        ----------
        deploy_file : Any
            指定的部署文件路径，可以是字符串路径或类文件对象，包含部署参数的 JSON 配置。

        Returns
        -------
        None
            方法不返回值，但会更新对象内部的参数映射 deploys。

        Notes
        -----
        - 方法首先尝试从指定文件中加载部署参数；
        - 如果文件不存在或内容无效（如格式错误），将自动使用默认值并创建新文件；
        - 所有参数会通过 `setattr()` 动态绑定到对象属性；
        - 会为每个参数输出加载日志，便于调试与回溯；
        - 如果发生不可预知异常，也会被捕获并打印警告日志。

        Workflow
        --------
        1. 调用 `load_parameters()` 加载 JSON 参数；
        2. 遍历默认 deploys 结构并更新每项配置；
        3. 若解析失败或文件缺失，则调用 `dump_deploy()` 写入默认值；
        4. 日志模块记录每个参数的加载结果。
        """
        try:
            parameters = load_parameters(deploy_file)
            for key, value in self.deploys.items():
                for k, v in value.items():
                    setattr(self, k, parameters.get(key, {}).get(k, v))
                    logger.debug(f"Load <{k}> = {v} -> {getattr(self, k)}")
        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            logger.debug(f"使用默认参数: {e}")
            logger.debug(f"生成部署文件: {deploy_file}")
            self.dump_deploy(deploy_file)
        except Exception as e:
            logger.debug(f"未知错误 {e}")

    def view_deploy(self) -> None:
        """
        以表格形式在控制台展示当前部署参数的详细信息。

        Returns
        -------
        None
            该方法不会返回值，但会在终端以表格方式输出参数配置详情。

        Notes
        -----
        - 本方法适用于交互式 CLI 场景，可直观查看当前所有配置项；
        - 使用 `rich.Table` 美化显示输出，包含分组标题、表头、每项参数的名称、值、范围、作用等；
        - 包括两大配置分组：
            - FST：帧级参数（如缩放、滤波器、起止时间等）；
            - ALS：分析参数（如步长、滑动窗口、裁剪钩子等）；
        - 若存在裁剪/忽略钩子 (`crops`、`omits`)，将额外绘制专属区域表格展示其索引与配置内容。

        Workflow
        --------
        1. 合并参数分组元信息（`Args.GROUP_FIRST` 和 `Args.GROUP_EXTRA`）；
        2. 遍历 `FST` 与 `ALS` 配置组，分别构建并打印参数表格；
        3. 若存在 `crops` 或 `omits` 钩子配置，则创建并显示对应的裁剪钩子表格；
        4. 所有输出均通过 `Design.console.print()` 渲染输出至 CLI。
        """
        deploys_group = {**Args.GROUP_FIRST, **Args.GROUP_EXTRA}

        table_style = {
            "title_justify": "center", "show_header": True, "show_lines": True
        }

        for key, value in self.deploys.items():
            table = Table(
                title=f"[bold #87CEFA]{const.DESC}({const.ALIAS}) Deploys {key}",
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

        if crops_list := self.crops:
            table = Table(
                title=f"[bold #5FAFFF]{const.DESC}({const.ALIAS}) Paint Crop Hook",
                header_style="bold #AFAFD7", **table_style
            )
            table.add_column("编号", justify="left", width=4)
            table.add_column("区域", justify="left", width=68)

            for index, hook in enumerate(crops_list, start=1):
                table.add_row(f"[bold #B2B2B2]{index:02}", f"[bold #87AFFF]{hook}")
            Design.console.print(table)

        if omits_list := self.omits:
            table = Table(
                title=f"[bold #5FAFFF]{const.DESC}({const.ALIAS}) Paint Omit Hook",
                header_style="bold #AFAFD7", **table_style
            )
            table.add_column("编号", justify="left", width=4)
            table.add_column("区域", justify="left", width=68)

            for index, hook in enumerate(omits_list, start=1):
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

    Notes
    -----
    - 所有配置项在初始化时通过 `load_option()` 方法自动加载；
    - 若配置文件不存在或格式错误，将自动创建默认配置文件。
    """

    options = {
        "total_place": "",  # 报告文件夹路径
        "model_place": "",  # 模型文件夹路径
        "faint_model": "",  # 灰度模型名称
        "color_model": "",  # 彩色模型名称
    }

    def __init__(self, option_file: typing.Any):
        self.load_option(option_file)

    def __getstate__(self):
        return self.options

    def __setstate__(self, state):
        self.options = state

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

    @total_place.setter
    def total_place(self, value: typing.Any):
        if type(value) is str and os.path.isdir(value):
            self.options["total_place"] = value

    @model_place.setter
    def model_place(self, value: typing.Any):
        if type(value) is str and os.path.isdir(value):
            self.options["model_place"] = value

    @faint_model.setter
    def faint_model(self, value: typing.Any):
        if type(value) is str and value:
            self.options["faint_model"] = value

    @color_model.setter
    def color_model(self, value: typing.Any):
        if type(value) is str and value:
            self.options["color_model"] = value

    def load_option(self, option_file: typing.Any) -> None:
        """
        从指定路径加载配置参数，并更新当前选项状态。

        Parameters
        ----------
        option_file : typing.Any
            配置文件路径，支持字符串或路径对象。

        Returns
        -------
        None

        Notes
        -----
        - 配置文件应为合法的 JSON 格式；
        - 若文件不存在或格式不合法，将创建并写入默认配置；
        - 支持对字段进行类型检查和目录存在性验证；
        - 成功加载后会通过日志记录每一项配置的变更。
        """
        try:
            parameters = load_parameters(option_file)
            for k, v in self.options.items():
                setattr(self, k, parameters.get(k, v))
                logger.debug(f"Load <{k}> = {v} -> {getattr(self, k)}")
        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            logger.debug(f"使用默认参数: {e}")
            logger.debug(f"生成配置文件: {option_file}")
            self.dump_option(option_file)
        except Exception as e:
            logger.debug(f"未知错误 {e}")

    def dump_option(self, option_file: typing.Any) -> None:
        """
        将当前选项配置以 JSON 格式写入指定文件路径。

        Parameters
        ----------
        option_file : typing.Any
            保存配置的目标文件路径。

        Returns
        -------
        None

        Notes
        -----
        - 若目录不存在则自动创建；
        - 文件采用 UTF-8 编码格式，结构清晰（含缩进）；
        - 此操作将覆盖原有文件内容，用于持久化保存当前参数配置。
        """
        os.makedirs(os.path.dirname(option_file), exist_ok=True)
        dump_parameters(option_file, self.options)


class Script(object):

    @staticmethod
    def dump_script(script_file: typing.Any) -> None:
        scripts = {
            "command": [
                {
                    "ID-X": {
                        "parser": {"FST": {"frate": 60}, "ALS": {"boost": True}},
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

        os.makedirs(os.path.dirname(script_file), exist_ok=True)
        dump_parameters(script_file, scripts)


if __name__ == '__main__':
    pass
