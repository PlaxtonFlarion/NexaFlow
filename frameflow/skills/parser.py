#
#   ____
#  |  _ \ __ _ _ __ ___  ___ _ __
#  | |_) / _` | '__/ __|/ _ \ '__|
#  |  __/ (_| | |  \__ \  __/ |
#  |_|   \__,_|_|  |___/\___|_|
#

import re
import typing
import argparse
import datetime
import textwrap
from frameflow import argument
from nexaflow import const


class Parser(object):

    __parse_engine: typing.Optional["argparse.ArgumentParser"] = None

    def __init__(self):
        custom_made_usage = f"""\
        --------------------------------------------    
        \033[1;35m{const.NAME}\033[0m --flick --speed --alone --scale 0.5 
        \033[1;35m{const.NAME}\033[0m --paint --scale 0.5 --omits 0,0,0,0.1
        \033[1;35m{const.NAME}\033[0m --union <file.path> --union <file.path>  
        \033[1;35m{const.NAME}\033[0m --merge <file.path> --merge <file.path>     
        \033[1;35m{const.NAME}\033[0m --speed --scale 0.5 --video <video.file>
        \033[1;35m{const.NAME}\033[0m --speed --scale 0.5 --stack <file.dir> --stack <file.dir>  
        \033[1;35m{const.NAME}\033[0m --train <video.file> --scale 0.5
        \033[1;35m{const.NAME}\033[0m --build <file.path> --shape 200,200    
        \033[1;35m{const.NAME}\033[0m --basic --shape 350,700 --gauss=1 --grind=-1 --video <video.file>        
        \033[1;35m{const.NAME}\033[0m --keras --boost --whist --shine --shape 350,700 --carry <file.path,A,B,C> --total <file.dir>
        \033[1;35m{const.NAME}\033[0m --keras --boost --whist --shine --shape 350,700 --fully <file.path> --total <file.dir>                
        \033[1;35m{const.NAME}\033[0m --keras --boost --start 00:00:08 --close 00:00:10 --video <video.file>
        \033[1;35m{const.NAME}\033[0m --keras --boost --start 00:00:00 --limit 00:00:02 --video <video.file>
        \033[1;35m{const.NAME}\033[0m --keras --boost --color --alike --shine --group --scale 0.5 --video <video.file> --video <video.file>
        \033[1;35m{const.NAME}\033[0m --keras --boost --scale 0.5 --begin=0,1 --final=-1,-1 --video <video.file>
        \033[1;35m{const.NAME}\033[0m --keras --boost --scale 0.5 --thres 0.98 --shift 3 --block 3 --video <video.file>
        \033[1;35m{const.NAME}\033[0m --keras --boost --scale 0.5 --crops 0,0,1,0.8 --omits 0,0,0,0.1 --video <video.file>
        \033[1;35m{const.NAME}\033[0m --keras --boost --scale 0.5 --debug --video <video.file>
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
                cmds.add_argument(
                    key, **value["args"], help=textwrap.dedent(f'''\
                        \033[1;34m^*{value["help"]}*^\033[0m
                        ----------------------
                                                                   
                    ''')
                )

    @property
    def parse_cmd(self) -> typing.Optional["argparse.Namespace"]:
        return self.__parse_engine.parse_args()

    @property
    def parse_engine(self) -> typing.Optional["argparse.ArgumentParser"]:
        return self.__parse_engine

    @staticmethod
    def parse_shape(dim_str: typing.Any) -> typing.Optional[tuple[int, int]]:

        def limited(loc: typing.Any) -> tuple[int, int]:
            return min(9999, max(0, loc[0])), min(9999, max(0, loc[1]))

        if type(dim_str) is list and len(dim_str) >= 2:
            if all(type(i) is int for i in dim_str):
                return limited(dim_str[:2])
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
                return limited(converted[:2])
        return None

    @staticmethod
    def parse_stage(dim_str: typing.Any) -> typing.Optional[tuple[int, int]]:

        def limited(loc: typing.Any) -> tuple[int, int]:
            return min(999, max(-999, loc[0])), min(999, max(-999, loc[1]))

        if type(dim_str) is list and len(dim_str) >= 2:
            if all(type(i) is int for i in dim_str):
                return limited(dim_str[:2])
        elif type(dim_str) is str:
            stage_parts = []
            parts = re.split(r"[.,;:\s]+", dim_str)
            match_parts = [part for part in parts if re.match(r"-?\d+(\.\d+)?", part)]
            for number in match_parts:
                try:
                    stage_parts.append(int(number))
                except ValueError:
                    stage_parts = []
                    break
            return limited(stage_parts[:2]) if len(stage_parts) >= 2 else None
        return None

    @staticmethod
    def parse_times(dim_str: typing.Any) -> typing.Optional[str]:
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
        try:
            value = float(dim_str)
        except (ValueError, TypeError):
            return None
        limited_value = round(max(min_v, min(max_v, value)), decimal)
        return int(limited_value) if decimal == 0 else limited_value


if __name__ == '__main__':
    Parser().parse_engine.print_help()
    pass
