import os
import json
from loguru import logger
from rich.table import Table
from frameflow.skills.parser import Parser
from frameflow.skills.show import Show
from nexaflow import const


def dump_parameters(src, dst) -> None:
    os.makedirs(os.path.dirname(src), exist_ok=True)
    with open(src, "w", encoding=const.CHARSET) as file:
        json.dump(dst, file, indent=4, separators=(",", ":"), ensure_ascii=False)


def load_parameters(src, dst) -> None:
    with open(src, "r", encoding=const.CHARSET) as file:
        dst.update(json.load(file))


class Deploy(object):

    deploys = {
        "alone": const.ALONE,
        "group": const.GROUP,
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

# Getter ###############################################################################################################

    @property
    def alone(self):
        return self.deploys["alone"]

    @property
    def group(self):
        return self.deploys["group"]

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

    @alone.setter
    def alone(self, value):
        self.deploys["alone"] = value

    @group.setter
    def group(self, value):
        self.deploys["group"] = value

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
            load_parameters(deploy_file, self.deploys)
            for k, v in self.deploys.items():
                setattr(self, k, v)
            logger.debug(f"读取部署文件，使用部署参数 ...")
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            logger.debug(f"未找到部署文件或文件解析错误，使用默认参数 ...")
        except Exception as e:
            logger.error(f"发生未知错误: {e}")

    def view_deploy(self) -> None:
        c = {0: "#AF5FD7", 1: "#D75F87", 2: "#87AFD7", 3: "#00AF5F"}

        table = Table(
            title=f"[bold {c[0]}]{const.DESC} Analyzer Deploy",
            header_style=f"bold {c[0]}",
            title_justify="center",
            show_header=True
        )
        table.add_column("配置", no_wrap=True)
        table.add_column("参数", no_wrap=True, max_width=12)
        table.add_column("范围", no_wrap=True)
        table.add_column("效果", no_wrap=True)

        information = [
            [
                f"[bold {c[1]}]独立控制",
                f"[bold {c[2]}]{self.alone}",
                f"[bold][[bold {c[3]}]T | F[/bold {c[3]}] ]",
                f"[bold green]开启[/bold green]" if self.alone else "[bold red]关闭[/bold red]"
            ],
            [
                f"[bold {c[1]}]分组报告",
                f"[bold {c[2]}]{self.group}",
                f"[bold][[bold {c[3]}]T | F[/bold {c[3]}] ]",
                f"[bold green]开启[/bold green]" if self.group else "[bold red]关闭[/bold red]",
            ],
            [
                f"[bold {c[1]}]跳帧模式",
                f"[bold {c[2]}]{self.boost}",
                f"[bold][[bold {c[3]}]T | F[/bold {c[3]}] ]",
                f"[bold green]开启[/bold green]" if self.boost else "[bold red]关闭[/bold red]",
            ],
            [
                f"[bold {c[1]}]彩色模式",
                f"[bold {c[2]}]{self.color}",
                f"[bold][[bold {c[3]}]T | F[/bold {c[3]}] ]",
                f"[bold green]开启[/bold green]" if self.color else "[bold red]关闭[/bold red]",
            ],
            [
                f"[bold {c[1]}]图片尺寸",
                f"[bold {c[2]}]{self.shape if self.shape else 'Auto'}",
                f"[bold][[bold {c[3]}]? , ?[/bold {c[3]}] ]",
                f"[bold]宽 [bold yellow]{self.shape[0]}[/bold yellow] 高 [bold yellow]{self.shape[1]}[/bold yellow]" if self.shape else f"[bold green]自动[/bold green]",
            ],
            [
                f"[bold {c[1]}]缩放比例",
                f"[bold {c[2]}]{self.scale if self.scale else 'Auto'}",
                f"[bold][[bold {c[3]}]0 , 1[/bold {c[3]}] ]",
                f"[bold]压缩图片至 [bold yellow]{self.scale}[/bold yellow] 倍" if self.scale else f"[bold green]自动[/bold green]",
            ],
            [
                f"[bold {c[1]}]开始时间",
                f"[bold {c[2]}]{Parser.parse_mills(self.start) if self.start else 'Auto'}",
                f"[bold][[bold {c[3]}]0 , ?[/bold {c[3]}] ]",
                f"[bold]开始时间 [bold yellow]{self.start}[/bold yellow]" if self.start else f"[bold green]自动[/bold green]",
            ],
            [
                f"[bold {c[1]}]结束时间",
                f"[bold {c[2]}]{Parser.parse_mills(self.close) if self.close else 'Auto'}",
                f"[bold][[bold {c[3]}]0 , ?[/bold {c[3]}] ]",
                f"[bold]结束时间 [bold yellow]{self.close}[/bold yellow]" if self.close else f"[bold green]自动[/bold green]",
            ],
            [
                f"[bold {c[1]}]持续时间",
                f"[bold {c[2]}]{Parser.parse_mills(self.limit) if self.limit else 'Auto'}",
                f"[bold][[bold {c[3]}]0 , ?[/bold {c[3]}] ]",
                f"[bold]持续时间 [bold yellow]{self.limit}[/bold yellow]" if self.limit else f"[bold green]自动[/bold green]",
            ],
            [
                f"[bold {c[1]}]开始阶段",
                f"[bold {c[2]}]{self.begin}",
                f"[bold][[bold {c[3]}]? , ?[/bold {c[3]}] ]",
                f"[bold]第 [bold yellow]{self.begin[0]}[/bold yellow] 个非稳态,第 [bold yellow]{self.begin[1]}[/bold yellow] 帧",
            ],
            [
                f"[bold {c[1]}]结束阶段",
                f"[bold {c[2]}]{self.final}",
                f"[bold][[bold {c[3]}]? , ?[/bold {c[3]}] ]",
                f"[bold]第 [bold yellow]{self.final[0]}[/bold yellow] 个非稳态,第 [bold yellow]{self.final[1]}[/bold yellow] 帧",
            ],
            [
                f"[bold {c[1]}]帧采样率",
                f"[bold {c[2]}]{self.frate}",
                f"[bold][[bold {c[3]}]1 , 60[/bold {c[3]}]]",
                f"[bold]每秒 [bold yellow]{self.frate}[/bold yellow] 帧",
            ],
            [
                f"[bold {c[1]}]相似度",
                f"[bold {c[2]}]{self.thres}",
                f"[bold][[bold {c[3]}]0 , 1[/bold {c[3]}] ]",
                f"[bold]阈值超过 [bold yellow]{self.thres}[/bold yellow] 的帧为稳定帧",
            ],
            [
                f"[bold {c[1]}]补偿值",
                f"[bold {c[2]}]{self.shift}",
                f"[bold][[bold {c[3]}]0 , ?[/bold {c[3]}] ]",
                f"[bold]合并 [bold yellow]{self.shift}[/bold yellow] 个变化不大的稳定区间",
            ],
            [
                f"[bold {c[1]}]立方体",
                f"[bold {c[2]}]{self.block}",
                f"[bold][[bold {c[3]}]1 , ?[/bold {c[3]}] ]",
                f"[bold]每个图像分成 [bold yellow]{self.block}[/bold yellow] 块",
            ],
            [
                f"[bold {c[1]}]获取区域",
                f"[bold {c[2]}]{['!' for _ in range(len(self.crops))]}",
                f"[bold][[bold {c[3]}]0 , 1[/bold {c[3]}] ]",
                f"[bold]获取 [bold yellow]{len(self.crops)}[/bold yellow] 个区域的图像",
            ],
            [
                f"[bold {c[1]}]忽略区域",
                f"[bold {c[2]}]{['!' for _ in range(len(self.omits))]}",
                f"[bold][[bold {c[3]}]0 , 1[/bold {c[3]}] ]",
                f"[bold]忽略 [bold yellow]{len(self.omits)}[/bold yellow] 个区域的图像",
            ]
        ]

        for info in information:
            table.add_row(*info)
        Show.console.print(table)


class Option(object):

    options = {
        "Total Place": "",
        "Model Place": "",
        "Model Shape": const.MODEL_SHAPE,
        "Model Aisle": const.MODEL_AISLE
    }

    def __init__(self, option_file: str):
        self.load_option(option_file)

    @property
    def total_place(self):
        return self.options["Total Place"]

    @property
    def model_place(self):
        return self.options["Model Place"]

    @property
    def model_shape(self):
        return self.options["Model Shape"]

    @property
    def model_aisle(self):
        return self.options["Model Aisle"]

    @total_place.setter
    def total_place(self, value):
        if type(value) is str and os.path.isdir(value):
            self.total_place = value

    @model_place.setter
    def model_place(self, value):
        if type(value) is str and os.path.isfile(value):
            self.model_place = value

    @model_shape.setter
    def model_shape(self, value):
        if effective := Parser.parse_shape(value):
            self.options["Model Shape"] = effective

    @model_aisle.setter
    def model_aisle(self, value):
        if effective := Parser.parse_aisle(value):
            self.options["Model Aisle"] = effective

    def load_option(self, option_file: str) -> None:
        try:
            load_parameters(option_file, self.options)
            for k, v in self.options.items():
                setattr(self, k, v)
            logger.debug(f"读取配置文件，使用配置参数 ...")
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            logger.debug(f"未找到配置文件或文件解析错误，使用默认参数 ...")
            self.dump_option(option_file)
        except Exception as e:
            logger.error(f"发生未知错误: {e}")

    def dump_option(self, option_file: str) -> None:
        dump_parameters(option_file, self.options)


class Script(object):

    @staticmethod
    def dump_script(script_file: str) -> None:
        scripts = {
            "commands": [
                {"name": "script_1", "loop": 1, "actions": [{"command": "", "args": []}, {"command": "", "args": []}]},
                {"name": "script_2", "loop": 1, "actions": [{"command": "", "args": []}, {"command": "", "args": []}]},
                {"name": "script_3", "loop": 1, "actions": [{"command": "", "args": []}, {"command": "", "args": []}]}
            ]
        }
        dump_parameters(script_file, scripts)


if __name__ == '__main__':
    pass
