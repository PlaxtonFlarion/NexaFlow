import os
import json
import typing
from loguru import logger
from rich.table import Table
from frameflow.skills.parser import Parser
from frameflow.skills.show import Show
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

# Heralds Getter #######################################################################################################

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

# Analyze Getter #######################################################################################################

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

# Heralds Setter #######################################################################################################

    @shape.setter
    def shape(self, value):
        self.deploys["FST"]["shape"] = Parser.parse_shape(value)

    @scale.setter
    def scale(self, value):
        self.deploys["FST"]["scale"] = Parser.parse_scale(value)

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
        if effective := Parser.parse_waves(value):
            self.deploys["FST"]["gauss"] = effective

    @grind.setter
    def grind(self, value):
        if effective := Parser.parse_waves(value):
            self.deploys["FST"]["grind"] = effective

    @frate.setter
    def frate(self, value):
        if effective := Parser.parse_frate(value):
            self.deploys["FST"]["frate"] = effective

# Analyze Setter #######################################################################################################

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
        if effective := Parser.parse_thres(value):
            self.deploys["ALS"]["thres"] = effective

    @shift.setter
    def shift(self, value):
        if effective := Parser.parse_other(value):
            self.deploys["ALS"]["shift"] = effective

    @block.setter
    def block(self, value):
        if effective := Parser.parse_other(value):
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
                self.deploys["ALS"][attr] = [{"x": 0, "y": 0, "x_size": 0, "y_size": 0}]
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
        except Exception as e:
            logger.debug(f"An unknown error occurred {e}")

    def view_deploy(self) -> None:
        cmd, arg, val = "#D75F87", "#87AFD7", "#7FFFD4"
        on, off, auto = "#CAFF70", "#FFB6C1", "#A4D3EE"
        tip = "#FFD700"

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

            if key == "FST":
                information = [
                    [
                        f"[bold {cmd}]尺寸定制",
                        f"[bold {arg}]{list(self.shape) if self.shape else 'Auto'}",
                        f"[bold][[bold {val}]? , ?[/] ]",
                        f"[bold]宽高 [bold {tip}]{self.shape[0]} x {self.shape[1]}" if self.shape else f"[bold {auto}]自动",
                    ],
                    [
                        f"[bold {cmd}]变幻缩放",
                        f"[bold {arg}]{self.scale if self.scale else 'Auto'}",
                        f"[bold][[bold {val}]0 , 1[/] ]",
                        f"[bold]压缩 [bold {tip}]{self.scale}[/]" if self.scale else f"[bold {auto}]自动[/]",
                    ],
                    [
                        f"[bold {cmd}]时刻启程",
                        f"[bold {arg}]{Parser.parse_mills(self.start) if self.start else 'Auto'}",
                        f"[bold][[bold {val}]0 , ?[/] ]",
                        f"[bold]开始 [bold {tip}]{self.start}[/]" if self.start else f"[bold {auto}]自动",
                    ],
                    [
                        f"[bold {cmd}]时光封印",
                        f"[bold {arg}]{Parser.parse_mills(self.close) if self.close else 'Auto'}",
                        f"[bold][[bold {val}]0 , ?[/] ]",
                        f"[bold]结束 [bold {tip}]{self.close}[/]" if self.close else f"[bold {auto}]自动",
                    ],
                    [
                        f"[bold {cmd}]持续历程",
                        f"[bold {arg}]{Parser.parse_mills(self.limit) if self.limit else 'Auto'}",
                        f"[bold][[bold {val}]0 , ?[/] ]",
                        f"[bold]持续 [bold {tip}]{self.limit}[/]" if self.limit else f"[bold {auto}]自动",
                    ],
                    [
                        f"[bold {cmd}]朦胧幻界",
                        f"[bold {arg}]{self.gauss if self.gauss else 'Auto'}",
                        f"[bold][[bold {val}]0 , ?[/] ]",
                        f"[bold]模糊 [bold {tip}]{self.gauss}[/]" if self.gauss else f"[bold {auto}]自动",
                    ],
                    [
                        f"[bold {cmd}]边缘觉醒",
                        f"[bold {arg}]{self.grind if self.grind else 'Auto'}",
                        f"[bold][[bold {val}]0 , ?[/] ]",
                        f"[bold]锐化 [bold {tip}]{self.grind}[/]" if self.grind else f"[bold {auto}]自动",
                    ],
                    [
                        f"[bold {cmd}]频率探测",
                        f"[bold {arg}]{self.frate}",
                        f"[bold][[bold {val}]1 , 60[/]]",
                        f"[bold]帧率 [bold {tip}]{self.frate}[/]",
                    ]
                ]

            else:
                information = [
                    [
                        f"[bold {cmd}]加速跳跃",
                        f"[bold {arg}]{self.boost}",
                        f"[bold][[bold {val}]T | F[/] ]",
                        f"[bold {on}]开启" if self.boost else f"[bold {off}]关闭",
                    ],
                    [
                        f"[bold {cmd}]彩绘世界",
                        f"[bold {arg}]{self.color}",
                        f"[bold][[bold {val}]T | F[/] ]",
                        f"[bold {on}]开启" if self.color else f"[bold {off}]关闭",
                    ],

                    [
                        f"[bold {cmd}]序章开启",
                        f"[bold {arg}]{list(self.begin)}",
                        f"[bold][[bold {val}]? , ?[/] ]",
                        f"[bold]非稳定阶段 [bold {tip}]{list(self.begin)}[/]",
                    ],
                    [
                        f"[bold {cmd}]终章落幕",
                        f"[bold {arg}]{list(self.final)}",
                        f"[bold][[bold {val}]? , ?[/] ]",
                        f"[bold]非稳定阶段 [bold {tip}]{list(self.final)}[/]",
                    ],
                    [
                        f"[bold {cmd}]稳定阈值",
                        f"[bold {arg}]{self.thres}",
                        f"[bold][[bold {val}]0 , 1[/] ]",
                        f"[bold]阈值超过 [bold {tip}]{self.thres}[/] 的帧为稳定帧",
                    ],
                    [
                        f"[bold {cmd}]偏移调整",
                        f"[bold {arg}]{self.shift}",
                        f"[bold][[bold {val}]0 , ?[/] ]",
                        f"[bold]合并 [bold {tip}]{self.shift}[/] 个变化不大的稳定区间",
                    ],
                    [
                        f"[bold {cmd}]矩阵分割",
                        f"[bold {arg}]{self.block}",
                        f"[bold][[bold {val}]1 , ?[/] ]",
                        f"[bold]每个图像分成 [bold {tip}]{self.block}[/] 块",
                    ],
                    [
                        f"[bold {cmd}]视界探索",
                        f"[bold {arg}]{['!' for _ in range(len(self.crops))]}",
                        f"[bold][[bold {val}]0 , 1[/] ]",
                        f"[bold]探索 [bold {tip}]{len(self.crops)}[/] 个区域的图像",
                    ],
                    [
                        f"[bold {cmd}]视界忽略",
                        f"[bold {arg}]{['!' for _ in range(len(self.omits))]}",
                        f"[bold][[bold {val}]0 , 1[/] ]",
                        f"[bold]忽略 [bold {tip}]{len(self.omits)}[/] 个区域的图像",
                    ]
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
                        "parser": {"boost": True},
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
