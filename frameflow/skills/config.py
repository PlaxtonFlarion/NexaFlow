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
        "boost": const.BOOST,
        "color": const.COLOR,

        "shape": const.SHAPE,
        "scale": const.SCALE,
        "start": const.START,
        "close": const.CLOSE,
        "limit": const.LIMIT,
        "begin": const.BEGIN,
        "final": const.FINAL,

        "frate": const.FRATE,
        "thres": const.THRES,
        "shift": const.SHIFT,
        "block": const.BLOCK,
        "crops": const.CROPS,
        "omits": const.OMITS
    }

    def __init__(self, deploy_file: str):
        self.load_deploy(deploy_file)

    def __getstate__(self):
        return self.deploys

    def __setstate__(self, state):
        self.deploys = state

# Getter ###############################################################################################################

    @property
    def boost(self):
        return self.deploys["boost"]

    @property
    def color(self):
        return self.deploys["color"]

    @property
    def shape(self):
        return self.deploys["shape"]

    @property
    def scale(self):
        return self.deploys["scale"]

    @property
    def start(self):
        return self.deploys["start"]

    @property
    def close(self):
        return self.deploys["close"]

    @property
    def limit(self):
        return self.deploys["limit"]

    @property
    def begin(self):
        return self.deploys["begin"]

    @property
    def final(self):
        return self.deploys["final"]

    @property
    def frate(self):
        return self.deploys["frate"]

    @property
    def thres(self):
        return self.deploys["thres"]

    @property
    def shift(self):
        return self.deploys["shift"]

    @property
    def block(self):
        return self.deploys["block"]

    @property
    def crops(self):
        return self.deploys["crops"]

    @property
    def omits(self):
        return self.deploys["omits"]

# Setter ###############################################################################################################

    @boost.setter
    def boost(self, value):
        self.deploys["boost"] = value

    @color.setter
    def color(self, value):
        self.deploys["color"] = value

    @shape.setter
    def shape(self, value):
        self.deploys["shape"] = Parser.parse_shape(value)

    @scale.setter
    def scale(self, value):
        self.deploys["scale"] = Parser.parse_scale(value)

    @start.setter
    def start(self, value):
        self.deploys["start"] = Parser.parse_times(value)

    @close.setter
    def close(self, value):
        self.deploys["close"] = Parser.parse_times(value)

    @limit.setter
    def limit(self, value):
        self.deploys["limit"] = Parser.parse_times(value)

    @begin.setter
    def begin(self, value):
        if effective := Parser.parse_stage(value):
            self.deploys["begin"] = effective

    @final.setter
    def final(self, value):
        if effective := Parser.parse_stage(value):
            self.deploys["final"] = effective

    @frate.setter
    def frate(self, value):
        if effective := Parser.parse_frate(value):
            self.deploys["frate"] = effective

    @thres.setter
    def thres(self, value):
        if effective := Parser.parse_thres(value):
            self.deploys["thres"] = effective

    @shift.setter
    def shift(self, value):
        if effective := Parser.parse_other(value):
            self.deploys["shift"] = effective

    @block.setter
    def block(self, value):
        if effective := Parser.parse_other(value):
            self.deploys["block"] = effective

    @crops.setter
    def crops(self, value):
        self.deploys["crops"] = Parser.parse_hooks(value)

    @omits.setter
    def omits(self, value):
        self.deploys["omits"] = Parser.parse_hooks(value)

    def dump_deploy(self, deploy_file: str) -> None:
        for attr in ["crops", "omits"]:
            if len(getattr(self, attr)) == 0:
                self.deploys[attr] = [{"x": 0, "y": 0, "x_size": 0, "y_size": 0}]
        dump_parameters(deploy_file, self.deploys)

    def load_deploy(self, deploy_file: str) -> None:
        try:
            parameters = load_parameters(deploy_file)
            for k, v in self.deploys.items():
                setattr(self, k, parameters.get(k, v))
                logger.debug(f"Load <{k}> = {v} -> {getattr(self, k)}")
        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            logger.debug(f"Use default parameters because {e}")
        except Exception as e:
            logger.error(f"{const.ERR}An unknown error occurred {e}")

    def view_deploy(self) -> None:
        cmd, arg, val = "#D75F87", "#87AFD7", "#7FFFD4"
        on, off, auto = "#CAFF70", "#FFB6C1", "#A4D3EE"
        tip = "#FFD700"

        table = Table(
            title=f"[bold #87CEFA]{const.DESC} Analyzer Deploy",
            header_style=f"bold #B0C4DE",
            title_justify="center",
            show_header=True
        )
        table.add_column("配置", no_wrap=True)
        table.add_column("参数", no_wrap=True, max_width=12)
        table.add_column("范围", no_wrap=True)
        table.add_column("效果", no_wrap=True)

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
                f"[bold {cmd}]尺寸定制",
                f"[bold {arg}]{list(self.shape) if self.shape else 'Auto'}",
                f"[bold][[bold {val}]? , ?[/] ]",
                f"[bold {tip}]{list(self.shape)}" if self.shape else f"[bold {auto}]自动",
            ],
            [
                f"[bold {cmd}]变幻缩放",
                f"[bold {arg}]{self.scale if self.scale else 'Auto'}",
                f"[bold][[bold {val}]0 , 1[/] ]",
                f"[bold]压缩图片至 [bold {tip}]{self.scale}[/] 倍" if self.scale else f"[bold {auto}]自动[/]",
            ],
            [
                f"[bold {cmd}]开始时间",
                f"[bold {arg}]{Parser.parse_mills(self.start) if self.start else 'Auto'}",
                f"[bold][[bold {val}]0 , ?[/] ]",
                f"[bold {tip}]{self.start}" if self.start else f"[bold {auto}]自动",
            ],
            [
                f"[bold {cmd}]结束时间",
                f"[bold {arg}]{Parser.parse_mills(self.close) if self.close else 'Auto'}",
                f"[bold][[bold {val}]0 , ?[/] ]",
                f"[bold {tip}]{self.close}" if self.close else f"[bold {auto}]自动",
            ],
            [
                f"[bold {cmd}]持续时间",
                f"[bold {arg}]{Parser.parse_mills(self.limit) if self.limit else 'Auto'}",
                f"[bold][[bold {val}]0 , ?[/] ]",
                f"[bold {tip}]{self.limit}" if self.limit else f"[bold {auto}]自动",
            ],
            [
                f"[bold {cmd}]开始阶段",
                f"[bold {arg}]{list(self.begin)}",
                f"[bold][[bold {val}]? , ?[/] ]",
                f"[bold]非稳定阶段 [bold {tip}]{list(self.begin)}[/]",
            ],
            [
                f"[bold {cmd}]结束阶段",
                f"[bold {arg}]{list(self.final)}",
                f"[bold][[bold {val}]? , ?[/] ]",
                f"[bold]非稳定阶段 [bold {tip}]{list(self.final)}[/]",
            ],
            [
                f"[bold {cmd}]帧采样率",
                f"[bold {arg}]{self.frate}",
                f"[bold][[bold {val}]1 , 60[/]]",
                f"[bold]每秒 [bold {tip}]{self.frate}[/] 帧",
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
                f"[bold {cmd}]空间构造",
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
        "total": "", "model": ""
    }

    def __init__(self, option_file: str):
        self.load_option(option_file)

    def __getstate__(self):
        return self.options

    def __setstate__(self, state):
        self.options = state

    @property
    def total(self):
        return self.options["total"]

    @property
    def model(self):
        return self.options["model"]

    @total.setter
    def total(self, value):
        if type(value) is str and os.path.isdir(value):
            self.options["total"] = value

    @model.setter
    def model(self, value):
        if type(value) is str and os.path.isdir(value):
            self.options["model"] = value

    def load_option(self, option_file: str) -> None:
        try:
            parameters = load_parameters(option_file)
            for k, v in self.options.items():
                setattr(self, k, parameters.get(k, v))
                logger.debug(f"Load <{k}> = {v} -> {getattr(self, k)}")
        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            logger.debug(f"Use default parameters because {e}")
            self.dump_option(option_file)
        except Exception as e:
            logger.error(f"{const.ERR}An unknown error occurred {e}")

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
