import os
import re
import json
import datetime
from loguru import logger
from rich.table import Table
from frameflow.skills.show import Show


class Deploy(object):

    _deploys = {
        "alone": False,
        "boost": False,
        "color": False,
        "group": False,
        "shape": None,
        "scale": None,
        "start": None,
        "close": None,
        "limit": None,
        "model_size": (350, 700),
        "fps": 60,
        "threshold": 0.97,
        "offset": 3,
        "block": 6,
        "crops": [],
        "omits": []
    }

    def __init__(self, deploy_file: str):
        self.load_deploy(deploy_file)

    @property
    def alone(self):
        return self._deploys["alone"]

    @property
    def boost(self):
        return self._deploys["boost"]

    @property
    def color(self):
        return self._deploys["color"]

    @property
    def group(self):
        return self._deploys["group"]

    @property
    def shape(self):
        return self._deploys["shape"]

    @property
    def scale(self):
        return self._deploys["scale"]

    @property
    def start(self):
        return self._deploys["start"]

    @property
    def close(self):
        return self._deploys["close"]

    @property
    def limit(self):
        return self._deploys["limit"]

    @property
    def model_size(self):
        return self._deploys["model_size"]

    @property
    def fps(self):
        return self._deploys["fps"]

    @property
    def threshold(self):
        return self._deploys["threshold"]

    @property
    def offset(self):
        return self._deploys["offset"]

    @property
    def block(self):
        return self._deploys["block"]

    @property
    def crops(self):
        return self._deploys["crops"]

    @property
    def omits(self):
        return self._deploys["omits"]

    @alone.setter
    def alone(self, value):
        self._deploys["alone"] = self.parse_bools(value)

    @boost.setter
    def boost(self, value):
        self._deploys["boost"] = self.parse_bools(value)

    @color.setter
    def color(self, value):
        self._deploys["color"] = self.parse_bools(value)

    @group.setter
    def group(self, value):
        self._deploys["group"] = self.parse_bools(value)

    @shape.setter
    def shape(self, value):
        self._deploys["shape"] = self.parse_sizes(value)

    @scale.setter
    def scale(self, value):
        if isinstance(value, int | float):
            self._deploys["scale"] = max(0.1, min(1.0, value))
        elif isinstance(value, str):
            if value.strip().upper() != "NONE":
                try:
                    self._deploys["scale"] = max(0.1, min(1.0, float(value)))
                except ValueError:
                    raise ValueError("scale 的值必须是一个可以转换为浮点数的数值 ...")

    @start.setter
    def start(self, value):
        self._deploys["start"] = self.parse_times(value)

    @close.setter
    def close(self, value):
        self._deploys["close"] = self.parse_times(value)

    @limit.setter
    def limit(self, value):
        self._deploys["limit"] = self.parse_times(value)

    @model_size.setter
    def model_size(self, value):
        self._deploys["model_size"] = self.parse_sizes(value)

    @fps.setter
    def fps(self, value):
        try:
            self._deploys["fps"] = max(1, min(60, int(value)))
        except ValueError:
            raise ValueError("fps 的值必须是一个可以转换为整数的数值 ...")

    @threshold.setter
    def threshold(self, value):
        try:
            self._deploys["threshold"] = max(0.1, min(1.0, round(float(value), 2)))
        except ValueError:
            raise ValueError("threshold 的值必须是一个可以转换为浮点数的数值 ...")

    @offset.setter
    def offset(self, value):
        try:
            self._deploys["offset"] = max(1, int(value))
        except ValueError:
            raise ValueError("offset 的值必须是一个可以转换为整数的数值 ...")

    @block.setter
    def block(self, value):
        try:
            self._deploys["block"] = max(1, int(value))
        except ValueError:
            raise ValueError("block 的值必须是一个可以转换为整数的数值 ...")

    @crops.setter
    def crops(self, value):
        self._deploys["crops"] = self.parse_hooks(value)

    @omits.setter
    def omits(self, value):
        self._deploys["omits"] = self.parse_hooks(value)

    @staticmethod
    def parse_bools(dim_str):
        if isinstance(dim_str, bool):
            return dim_str
        elif isinstance(dim_str, str):
            mode = dim_str.lower() if isinstance(dim_str, str) else "false"
            return True if mode == "true" else False
        else:
            return False

    @staticmethod
    def parse_sizes(dim_str):
        if isinstance(dim_str, tuple):
            return dim_str
        elif isinstance(dim_str, str):
            match_size_list = re.findall(r"-?\d*\.?\d+", dim_str)
            if len(match_size_list) >= 2:
                converted = []
                for num in match_size_list:
                    try:
                        converted_num = int(num)
                    except ValueError:
                        converted_num = float(num)
                    converted.append(converted_num)
                return tuple(converted[:2])
        return None

    @staticmethod
    def parse_hooks(dim_str):
        hooks_list, effective = dim_str, []
        for hook in hooks_list:
            if isinstance(hook, dict):
                data_list = [value for value in hook.values() if isinstance(value, int | float)]
                if len(data_list) == 4 and sum(data_list) > 0:
                    effective.append((hook["x"], hook["y"], hook["x_size"], hook["y_size"]))
            elif isinstance(hook, tuple):
                effective.append(hook)

        effective_list = list(set(effective)).copy()
        effective.clear()
        return effective_list

    @staticmethod
    def parse_times(dim_str):
        if isinstance(dim_str, (int, float)):
            if dim_str >= 86400:
                raise ValueError("时间不能超过 24 小时 ...")
            return str(datetime.timedelta(seconds=dim_str))
        elif isinstance(dim_str, str):
            time_pattern = re.compile(r"(?:(\d+):)?(\d+):(\d+)(?:\.(\d+))?|^\d*(?:\.\d+)?$")
            if match := time_pattern.match(dim_str):
                hours = int(match.group(1)) if match.group(1) else 0
                minutes = int(match.group(2)) if match.group(2) else 0
                seconds = int(match.group(3)) if match.group(3) else 0
                milliseconds = int(float("0." + match.group(4)) * 1000) if match.group(4) else 0
                if match.group(0) and '.' in match.group(0):
                    seconds = float(match.group(0))
                    milliseconds = int((seconds - int(seconds)) * 1000)
                    seconds = int(seconds)
                time_str = datetime.timedelta(
                    hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds
                )
                return str(time_str)
        return None

    @staticmethod
    def parse_mills(dim_str):
        if isinstance(dim_str, int | float):
            return float(dim_str)
        if isinstance(dim_str, str):
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

    def dump_deploy(self, deploy_file: str) -> None:
        os.makedirs(os.path.dirname(deploy_file), exist_ok=True)

        with open(file=deploy_file, mode="w", encoding="utf-8") as f:
            f.writelines('{')
            for k, v in self._deploys.items():
                f.writelines('\n')
                if isinstance(v, bool) or v is None:
                    f.writelines(f'    "{k}": "{v}",')
                elif k == "model_size" or k == "shape" or k == "start" or k == "close" or k == "limit":
                    f.writelines(f'    "{k}": "{v}",')
                elif k == "crops" or k == "omits":
                    if len(v) == 0:
                        default = '{"x": 0, "y": 0, "x_size": 0, "y_size": 0}'
                        f.writelines(f'    "{k}": [\n')
                        f.writelines(f'        {default}\n')
                        f.writelines('    ],') if k == "crops" else f.writelines('    ]')
                    else:
                        f.writelines(f'    "{k}": [\n')
                        for index, i in enumerate(v):
                            x, y, x_size, y_size = i
                            new_size = f'{{"x": {x}, "y": {y}, "x_size": {x_size}, "y_size": {y_size}}}'
                            if (index + 1) == len(v):
                                f.writelines(f'        {new_size}\n')
                            else:
                                f.writelines(f'        {new_size},\n')
                        f.writelines('    ],') if k == "crops" else f.writelines('    ]')
                else:
                    f.writelines(f'    "{k}": {v},')
            f.writelines('\n}')

    def load_deploy(self, deploy_file: str) -> None:
        try:
            with open(file=deploy_file, mode="r", encoding="utf-8") as f:
                data = json.loads(f.read())
                self.alone = data.get("alone", "false")
                self.boost = data.get("boost", "false")
                self.color = data.get("color", "false")
                self.group = data.get("group", "false")
                self.shape = data.get("shape", None)
                self.scale = data.get("scale", None)
                self.start = data.get("start", None)
                self.close = data.get("close", None)
                self.limit = data.get("limit", None)
                self.model_size = data.get("model_size", (350, 700))
                self.fps = data.get("fps", 60)
                self.threshold = data.get("threshold", 0.97)
                self.offset = data.get("offset", 3)
                self.block = data.get("block", 6)
                self.crops = data.get("crops", [])
                self.omits = data.get("omits", [])
        except FileNotFoundError:
            logger.debug("未找到部署文件,使用默认参数 ...")
        except json.decoder.JSONDecodeError:
            logger.warning("部署文件解析错误,文件格式不正确,使用默认参数 ...")
        except Exception as e:
            logger.error(e)
        else:
            logger.info("读取部署文件,使用部署参数 ...")

    def view_deploy(self) -> None:

        title_color = "#af5fd7"
        col_1_color = "#d75f87"
        col_2_color = "#87afd7"
        col_3_color = "#00af5f"

        table = Table(
            title=f"[bold {title_color}]Framix Analyzer Deploy",
            header_style=f"bold {title_color}", title_justify="center",
            show_header=True
        )
        table.add_column("配置", no_wrap=True)
        table.add_column("参数", no_wrap=True, max_width=12)
        table.add_column("范围", no_wrap=True)
        table.add_column("效果", no_wrap=True)

        table.add_row(
            f"[bold {col_1_color}]独立模式",
            f"[bold {col_2_color}]{self.alone}",
            f"[bold][[bold {col_3_color}]T | F[/bold {col_3_color}] ]",
            f"[bold green]开启[/bold green]" if self.alone else "[bold red]关闭[/bold red]",
        )
        table.add_row(
            f"[bold {col_1_color}]跳帧模式",
            f"[bold {col_2_color}]{self.boost}",
            f"[bold][[bold {col_3_color}]T | F[/bold {col_3_color}] ]",
            f"[bold green]开启[/bold green]" if self.boost else "[bold red]关闭[/bold red]",
        )
        table.add_row(
            f"[bold {col_1_color}]彩色模式",
            f"[bold {col_2_color}]{self.color}",
            f"[bold][[bold {col_3_color}]T | F[/bold {col_3_color}] ]",
            f"[bold green]开启[/bold green]" if self.color else "[bold red]关闭[/bold red]",
        )
        table.add_row(
            f"[bold {col_1_color}]分组模式",
            f"[bold {col_2_color}]{self.group}",
            f"[bold][[bold {col_3_color}]T | F[/bold {col_3_color}] ]",
            f"[bold green]开启[/bold green]" if self.group else "[bold red]关闭[/bold red]",
        )
        table.add_row(
            f"[bold {col_1_color}]图片尺寸",
            f"[bold {col_2_color}]{self.shape}" if self.shape else f"[bold {col_2_color}]Auto",
            f"[bold][[bold {col_3_color}]? , ?[/bold {col_3_color}] ]",
            f"[bold]宽 [bold yellow]{self.shape[0]}[/bold yellow] 高 [bold yellow]{self.shape[1]}[/bold yellow]" if self.shape else f"[bold green]自动[/bold green]",
        )
        table.add_row(
            f"[bold {col_1_color}]压缩比例",
            f"[bold {col_2_color}]{self.scale}" if self.scale else f"[bold {col_2_color}]Auto",
            f"[bold][[bold {col_3_color}]0 , 1[/bold {col_3_color}] ]",
            f"[bold]压缩图片至 [bold yellow]{self.scale}[/bold yellow] 倍" if self.scale else f"[bold green]自动[/bold green]",
        )
        table.add_row(
            f"[bold {col_1_color}]视频开始",
            f"[bold {col_2_color}]{self.parse_mills(self.start)}" if self.start else f"[bold {col_2_color}]Auto",
            f"[bold][[bold {col_3_color}]0 , ?[/bold {col_3_color}] ]",
            f"[bold]开始时间 [bold yellow]{self.start}[/bold yellow]" if self.start else f"[bold green]自动[/bold green]",
        )
        table.add_row(
            f"[bold {col_1_color}]视频结束",
            f"[bold {col_2_color}]{self.parse_mills(self.close)}" if self.close else f"[bold {col_2_color}]Auto",
            f"[bold][[bold {col_3_color}]0 , ?[/bold {col_3_color}] ]",
            f"[bold]结束时间 [bold yellow]{self.close}[/bold yellow]" if self.close else f"[bold green]自动[/bold green]",
        )
        table.add_row(
            f"[bold {col_1_color}]剪切长度",
            f"[bold {col_2_color}]{self.parse_mills(self.limit)}" if self.limit else f"[bold {col_2_color}]Auto",
            f"[bold][[bold {col_3_color}]0 , ?[/bold {col_3_color}] ]",
            f"[bold]剪切时间 [bold yellow]{self.limit}[/bold yellow]" if self.limit else f"[bold green]自动[/bold green]",
        )
        table.add_row(
            f"[bold {col_1_color}]模型尺寸",
            f"[bold {col_2_color}]{self.model_size}",
            f"[bold][[bold {col_3_color}]? , ?[/bold {col_3_color}] ]",
            f"[bold]宽 [bold yellow]{self.model_size[0]}[/bold yellow] 高 [bold yellow]{self.model_size[1]}[/bold yellow]",
        )
        table.add_row(
            f"[bold {col_1_color}]帧采样率",
            f"[bold {col_2_color}]{self.fps}",
            f"[bold][[bold {col_3_color}]1 , 60[/bold {col_3_color}]]",
            f"[bold]每秒 [bold yellow]{self.fps}[/bold yellow] 帧",
        )
        table.add_row(
            f"[bold {col_1_color}]相似度",
            f"[bold {col_2_color}]{self.threshold}",
            f"[bold][[bold {col_3_color}]0 , 1[/bold {col_3_color}] ]",
            f"[bold]阈值超过 [bold yellow]{self.threshold}[/bold yellow] 的帧为稳定帧",
        )
        table.add_row(
            f"[bold {col_1_color}]补偿值",
            f"[bold {col_2_color}]{self.offset}",
            f"[bold][[bold {col_3_color}]0 , ?[/bold {col_3_color}] ]",
            f"[bold]合并 [bold yellow]{self.offset}[/bold yellow] 个变化不大的稳定区间",
        )
        table.add_row(
            f"[bold {col_1_color}]切分程度",
            f"[bold {col_2_color}]{self.block}",
            f"[bold][[bold {col_3_color}]1 , ?[/bold {col_3_color}] ]",
            f"[bold]每个帧图像切分为 [bold yellow]{self.block}[/bold yellow] 块",
        )
        table.add_row(
            f"[bold {col_1_color}]获取区域",
            f"[bold {col_2_color}]{['!' for _ in range(len(self.crops))]}",
            f"[bold][[bold {col_3_color}]0 , 1[/bold {col_3_color}] ]",
            f"[bold]获取 [bold yellow]{len(self.crops)}[/bold yellow] 个区域的图像",
        )
        table.add_row(
            f"[bold {col_1_color}]忽略区域",
            f"[bold {col_2_color}]{['!' for _ in range(len(self.omits))]}",
            f"[bold][[bold {col_3_color}]0 , 1[/bold {col_3_color}] ]",
            f"[bold]忽略 [bold yellow]{len(self.omits)}[/bold yellow] 个区域的图像",
        )
        Show.console.print(table)


class Option(object):

    _options = {
        "Total Path": ""
    }

    def __init__(self, option_file: str):
        self.load_option(option_file)

    @property
    def total_path(self):
        return self._options["Total Path"]

    @total_path.setter
    def total_path(self, value):
        if value and os.path.isdir(value):
            if not os.path.exists(value):
                os.makedirs(value, exist_ok=True)
            self._options["Total Path"] = value

    def load_option(self, option_file: str) -> None:
        try:
            with open(file=option_file, mode="r", encoding="utf-8") as f:
                data = json.loads(f.read())
        except FileNotFoundError:
            logger.debug("未找到配置文件,使用默认路径 ...")
            self.dump_option(option_file)
        except json.decoder.JSONDecodeError:
            logger.debug("配置文件解析错误,文件格式不正确,使用默认路径 ...")
        else:
            logger.debug("读取配置文件,使用配置参数 ...")
            self.total_path = data.get("Total Path", "")

    def dump_option(self, option_file: str) -> None:
        os.makedirs(os.path.dirname(option_file), exist_ok=True)

        with open(file=option_file, mode="w", encoding="utf-8") as f:
            f.writelines('{')
            for k, v in self._options.items():
                f.writelines('\n')
                f.writelines(f'    "{k}": "{v}"')
            f.writelines('\n}')


if __name__ == '__main__':
    pass
