#
#   ____
#  |  _ \ __ _ _ __ ___  ___ _ __
#  | |_) / _` | '__/ __|/ _ \ '__|
#  |  __/ (_| | |  \__ \  __/ |
#  |_|   \__,_|_|  |___/\___|_|
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

import re
import typing
import argparse
import datetime
import textwrap
from nexacore import argument
from nexaflow import const


class Parser(object):
    """
    命令行参数解析器。

    封装了基于 argparse 的命令行参数注册与解析逻辑，支持多种参数类型（数值、时间、坐标等）的预处理。
    支持参数分组、互斥逻辑、高亮帮助信息输出等特性，适用于复杂的 CLI 工具配置入口。
    """

    __parse_engine: typing.Optional["argparse.ArgumentParser"] = None

    def __init__(self):
        custom_made_usage = f"""\
        --------------------------------------------
        \033[1;35m{const.NAME}\033[0m Get started quickly
        """
        self.__parse_engine = argparse.ArgumentParser(
            const.NAME,
            usage=f" \033[1;35m{const.NAME}\033[0m [-h] [--help] View help documentation\n" + custom_made_usage,
            description=textwrap.dedent(f'''\
                \033[1;32m{const.DESC} · {const.ALIAS}\033[0m
                \033[1m-----------------------------\033[0m
                \033[1;32mCommand Line Arguments {const.DESC}\033[0m
            '''),
            formatter_class=argparse.RawTextHelpFormatter
        )

        for keys, values in argument.Args.ARGUMENT.items():
            description = "参数兼容"
            if keys in ["核心操控", "辅助利器", "视控精灵"]:
                description = "参数互斥"
                mutually_exclusive = self.__parse_engine.add_argument_group(
                    title=f"\033[1m^* {keys} *^\033[0m",
                    description=textwrap.dedent(f'''\
                        \033[1;33m{description}\033[0m
                    '''),
                )
                cmds = mutually_exclusive.add_mutually_exclusive_group()
            else:
                cmds = self.__parse_engine.add_argument_group(
                    title=f"\033[1m^* {keys} *^\033[0m",
                    description=textwrap.dedent(f'''\
                        \033[1;32m{description}\033[0m
                    '''),
                )

            for key, value in values.items():
                *_, kind = value["view"]
                cmds.add_argument(
                    key, **value["args"], help=textwrap.dedent(f'''\
                \033[1;34m^*{value["help"]}*^\033[0m
                ----------------------
                {value.get("func", "")}
                ----------------------
                \033[35m{const.NAME}\033[0m {key}\033[36m{kind}\033[0m

                    ''')
                )

    @property
    def parse_cmd(self) -> typing.Optional["argparse.Namespace"]:
        return self.__parse_engine.parse_args()

    @property
    def parse_engine(self) -> typing.Optional["argparse.ArgumentParser"]:
        return self.__parse_engine

    @staticmethod
    def limited(loc: typing.Any, max_level: int, min_level: int) -> tuple[int, int]:
        return min(max_level, max(min_level, loc[0])), min(max_level, max(min_level, loc[1]))

    @staticmethod
    def parse_shape(dim_str: typing.Any) -> typing.Optional[tuple[int, int]]:
        """
        将字符串或列表解析为合法的图像尺寸元组 (W, H)，并限定在最大值范围内。
        """
        max_level, min_level = 9999, 0

        if type(dim_str) is list and len(dim_str) >= 2:
            if all(type(i) is int for i in dim_str):
                return Parser.limited(dim_str[:2], max_level, min_level)

        elif type(dim_str) is str:
            match_size_list = re.findall(r"-?\d*\.?\d+", dim_str)
            if len(match_size_list) >= 2:
                converted = []
                for num in match_size_list:
                    try:
                        converted_num = int(num)
                    except ValueError:
                        converted_num = float(num)
                    converted.append(converted_num)

                return Parser.limited(converted[:2], max_level, min_level)

        return None

    @staticmethod
    def parse_stage(dim_str: typing.Any) -> typing.Optional[tuple[int, int]]:
        """
        解析阶段区间参数，支持字符串或整数列表格式，范围在 [-9999, 9999]。
        """
        max_level, min_level = 9999, -9999

        if type(dim_str) is list and len(dim_str) >= 2:
            if all(type(i) is int for i in dim_str):
                return Parser.limited(dim_str[:2], max_level, min_level)

        elif type(dim_str) is str:
            stage_parts = []
            parts = re.split(r"[.,;:\s]+", dim_str)
            match_parts = [
                part for part in parts if re.match(r"-?\d+(\.\d+)?", part)
            ]
            for number in match_parts:
                try:
                    stage_parts.append(int(number))
                except ValueError:
                    stage_parts = []
                    break

            return Parser.limited(
                stage_parts[:2], max_level, min_level) if len(stage_parts) >= 2 else None

        return None

    @staticmethod
    def parse_times(dim_str: typing.Any) -> typing.Optional[str]:
        """
        将时间字符串或秒数（int/float）转换为标准格式字符串，如 '00:00:12.340'。
        """
        hour_scope, second_scope = 24, 86400

        if type(dim_str) is int or type(dim_str) is float:
            if dim_str >= second_scope:
                return None
            return str(datetime.timedelta(seconds=dim_str))

        elif type(dim_str) is str:
            time_pattern = re.compile(r"^(?:(\d+):)?(\d+):(\d+)(?:\.(\d+))?$|^\d+(?:\.\d+)?$")
            if match := time_pattern.match(dim_str):
                if ':' in dim_str:
                    hours = int(match.group(1)) if match.group(1) else 0
                    minutes = int(match.group(2)) if match.group(2) else 0
                    seconds = int(match.group(3)) if match.group(3) else 0
                    milliseconds = int(float("0." + match.group(4)) * 1000) if match.group(4) else 0

                    if hours > hour_scope:
                        return None
                    time_str = datetime.timedelta(
                        hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds
                    )
                    return str(time_str)

                else:
                    seconds = float(match.group(0))
                    milliseconds = int((seconds - int(seconds)) * 1000)
                    seconds = int(seconds)
                    time_str = datetime.timedelta(
                        seconds=seconds, milliseconds=milliseconds
                    )
                    return str(time_str)

        return None

    @staticmethod
    def parse_mills(dim_str: typing.Any) -> typing.Optional[typing.Union[int, float]]:
        """
        将时间字符串解析为毫秒数，支持标准时间格式与秒数格式。
        """
        if type(dim_str) is int or type(dim_str) is float:
            return float(dim_str)

        elif type(dim_str) is str:
            seconds_pattern = re.compile(r"^\d+(\.\d+)?$")
            full_pattern = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})(\.\d+)?")
            if match := full_pattern.match(dim_str):
                hours, minutes, seconds, milliseconds = match.groups()
                total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
                if milliseconds:
                    total_seconds += float(milliseconds)
                return total_seconds

            elif seconds_pattern.match(dim_str):
                return float(dim_str)

        return None

    @staticmethod
    def parse_hooks(dim_str: typing.Any) -> list[dict]:
        """
        解析区域裁剪或忽略参数列表为合法的矩形描述字典（x, y, x_size, y_size）。
        """
        effective_hook_list, requires = [], ["x", "y", "x_size", "y_size"]

        for hook in dim_str:
            if type(hook) is dict:
                if hook.keys() >= set(requires) and all(isinstance(hook[key], (int, float)) for key in requires):
                    if sum([hook[key] for key in requires]) > 0:
                        effective_hook_list.append({key: hook[key] for key in requires})

            elif type(hook) is str:
                if len(match_list := re.findall(r"-?\d*\.?\d+", hook)) == 4:
                    valid_list = [
                        float(num) if "." in num else int(num) for num in match_list
                    ]
                    if len(valid_list) == 4 and sum(valid_list) > 0:
                        valid_dict = {k: v for k, v in zip(requires, valid_list)}
                        effective_hook_list.append(dict(tuple(valid_dict.items())))

        return effective_hook_list

    @staticmethod
    def parse_waves(dim_str: typing.Any, min_v: int | float, max_v: int | float, decimal: int) -> typing.Optional[int]:
        """
        将浮点数参数限定在指定区间并进行小数位数限制，常用于参数校验。
        """
        try:
            value = float(dim_str)
        except (ValueError, TypeError):
            return None

        limited_value = round(max(min_v, min(max_v, value)), decimal)
        return int(limited_value) if decimal == 0 else limited_value


if __name__ == '__main__':
    pass
