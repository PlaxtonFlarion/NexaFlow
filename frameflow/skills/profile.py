#
#   ____             __ _ _
#  |  _ \ _ __ ___  / _(_) | ___
#  | |_) | '__/ _ \| |_| | |/ _ \
#  |  __/| | | (_) |  _| | |  __/
#  |_|   |_|  \___/|_| |_|_|\___|
#

import os
import json
import typing
import inspect
from loguru import logger
from rich.table import Table
from frameflow.skills.parser import Parser
from frameflow.skills.show import Show
from frameflow.argument import Args
from nexaflow import const


def dump_parameters(src: str, dst: dict) -> None:
    os.makedirs(os.path.dirname(src), exist_ok=True)
    with open(src, "w", encoding=const.CHARSET) as file:
        json.dump(dst, file, indent=4, separators=(",", ":"), ensure_ascii=False)


def load_parameters(src: str) -> typing.Any:
    with open(src, "r", encoding=const.CHARSET) as file:
        return json.load(file)


class Deploy(object):

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
            "block": const.BLOCK,
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
    def block(self):
        return self.deploys["ALS"]["block"]

    @property
    def crops(self):
        return self.deploys["ALS"]["crops"]

    @property
    def omits(self):
        return self.deploys["ALS"]["omits"]

# Setter ###############################################################################################################

    @shape.setter
    def shape(self, value):
        if effective := Parser.parse_shape(value):
            self.deploys["FST"]["shape"] = effective

    @scale.setter
    def scale(self, value):
        if effective := Parser.parse_waves(value, min_val=0.0, max_val=1.0, decimal_places=1):
            self.deploys["FST"]["scale"] = effective

    @start.setter
    def start(self, value):
        self.deploys["FST"]["start"] = Parser.parse_times(value)

    @close.setter
    def close(self, value):
        self.deploys["FST"]["close"] = Parser.parse_times(value)

    @limit.setter
    def limit(self, value):
        self.deploys["FST"]["limit"] = Parser.parse_times(value)

    @gauss.setter
    def gauss(self, value):
        if effective := Parser.parse_waves(value, min_val=0.0, max_val=10.0, decimal_places=1):
            self.deploys["FST"]["gauss"] = effective

    @grind.setter
    def grind(self, value):
        if effective := Parser.parse_waves(value, min_val=-2.0, max_val=5.0, decimal_places=1):
            self.deploys["FST"]["grind"] = effective

    @frate.setter
    def frate(self, value):
        if effective := Parser.parse_waves(value, min_val=1, max_val=60, decimal_places=0):
            self.deploys["FST"]["frate"] = effective

# Setter ###############################################################################################################

    @boost.setter
    def boost(self, value):
        self.deploys["ALS"]["boost"] = value

    @color.setter
    def color(self, value):
        self.deploys["ALS"]["color"] = value

    @begin.setter
    def begin(self, value):
        if effective := Parser.parse_stage(value):
            self.deploys["ALS"]["begin"] = effective

    @final.setter
    def final(self, value):
        if effective := Parser.parse_stage(value):
            self.deploys["ALS"]["final"] = effective

    @thres.setter
    def thres(self, value):
        if (effective := Parser.parse_waves(value, min_val=0.0, max_val=1.0, decimal_places=2)) is None:
            effective = const.THRES
        self.deploys["ALS"]["thres"] = effective

    @shift.setter
    def shift(self, value):
        if (effective := Parser.parse_waves(value, min_val=0, max_val=10, decimal_places=0)) is None:
            effective = const.SHIFT
        self.deploys["ALS"]["shift"] = effective

    @block.setter
    def block(self, value):
        if effective := Parser.parse_waves(value, min_val=1, max_val=10, decimal_places=0):
            self.deploys["ALS"]["block"] = effective

    @crops.setter
    def crops(self, value):
        self.deploys["ALS"]["crops"] = Parser.parse_hooks(value)

    @omits.setter
    def omits(self, value):
        self.deploys["ALS"]["omits"] = Parser.parse_hooks(value)

    def dump_deploy(self, deploy_file: str) -> None:
        for attr in ["crops", "omits"]:
            if len(self.deploys["ALS"][attr]) == 0:
                self.deploys["ALS"][attr] = const.HOOKS
        dump_parameters(deploy_file, self.deploys)

    def load_deploy(self, deploy_file: str) -> None:
        try:
            parameters = load_parameters(deploy_file)
            for key, value in self.deploys.items():
                for k, v in value.items():
                    setattr(self, k, parameters.get(key, {}).get(k, v))
                    logger.debug(f"Load <{k}> = {v} -> {getattr(self, k)}")
        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            logger.debug(f"Use DP Because {e}")
            self.dump_deploy(deploy_file)
        except Exception as e:
            logger.debug(f"An unknown error occurred {e}")

    def view_deploy(self) -> None:
        deploys_group = {**Args.GROUP_FIRST, **Args.GROUP_EXTRA}

        for key, value in self.deploys.items():
            table = Table(
                title=f"[bold #87CEFA]{const.DESC} Deploys {key}",
                header_style=f"bold #B0C4DE",
                title_justify="center",
                show_header=True
            )
            table.add_column("配置", no_wrap=True, width=8)
            table.add_column("参数", no_wrap=True, width=12)
            table.add_column("范围", no_wrap=True, width=8)
            table.add_column("效果", no_wrap=True, min_width=30)
            information = [
                [f"[bold #D75F87]{v['help']}"] + (func(self, Parser) if len(
                    inspect.signature(func := v["push"]).parameters
                ) == 2 else func(self))
                for k, v in deploys_group.items() if k.lstrip("--") in value
            ]

            for info in information:
                table.add_row(*info)
            Show.console.print(table)


class Option(object):

    options = {
        "total_place": "", "model_place": ""
    }

    def __init__(self, option_file: str):
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

    @total_place.setter
    def total_place(self, value):
        if type(value) is str and os.path.isdir(value):
            self.options["total_place"] = value

    @model_place.setter
    def model_place(self, value):
        if type(value) is str and os.path.isdir(value):
            self.options["model_place"] = value

    def load_option(self, option_file: str) -> None:
        try:
            parameters = load_parameters(option_file)
            for k, v in self.options.items():
                setattr(self, k, parameters.get(k, v))
                logger.debug(f"Load <{k}> = {v} -> {getattr(self, k)}")
        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            logger.debug(f"Use DP Because {e}")
            self.dump_option(option_file)
        except Exception as e:
            logger.debug(f"An unknown error occurred {e}")

    def dump_option(self, option_file: str) -> None:
        dump_parameters(option_file, self.options)


class Script(object):

    @staticmethod
    def dump_script(script_file: str) -> None:
        scripts = {
            "command": [
                {
                    "ID-X": {
                        "parser": {"FST": {"frate": 60}, "ALS": {"boost": True}},
                        "header": ["script"],
                        "change": [],
                        "looper": 1,
                        "prefix": [
                            {"cmds": [], "args": []}, {"cmds": [], "args": []}
                        ],
                        "action": [
                            {"cmds": [], "args": []}, {"cmds": [], "args": []}
                        ],
                        "suffix": [
                            {"cmds": [], "args": []}, {"cmds": [], "args": []}
                        ]
                    }
                }
            ]
        }
        dump_parameters(script_file, scripts)


if __name__ == '__main__':
    pass
