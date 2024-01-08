import os
import re
import json
from loguru import logger
from rich.table import Table
from frameflow.show import Show


class Deploy(object):

    _deploys = {
        "boost": False,
        "color": False,
        "focus": False,
        "target_size": (350, 700),
        "fps": 60,
        "compress_rate": 0.5,
        "threshold": 0.97,
        "offset": 3,
        "window_size": 1,
        "step": 1,
        "block": 6,
        "window_coefficient": 2,
        "crops": [],
        "omits": []
    }

    def __init__(
            self,
            boost: bool = None,
            color: bool = None,
            focus: bool = None,
            target_size: tuple = None,
            fps: int = None,
            compress_rate: int | float = None,
            threshold: int | float = None,
            offset: int = None,
            window_size: int = None,
            step: int = None,
            block: int = None,
            window_coefficient: int = None,
            crops: list = None,
            omits: list = None
    ):

        self._deploys["boost"] = boost or False
        self._deploys["color"] = color or False
        self._deploys["focus"] = focus or False
        self._deploys["target_size"] = target_size or (350, 700)
        self._deploys["fps"] = fps or 60
        self._deploys["compress_rate"] = compress_rate or 0.5
        self._deploys["threshold"] = threshold or 0.97
        self._deploys["offset"] = offset or 3
        self._deploys["window_size"] = window_size or 1
        self._deploys["step"] = step or 1
        self._deploys["block"] = block or 6
        self._deploys["window_coefficient"] = window_coefficient or 2
        self._deploys["crops"] = crops or []
        self._deploys["omits"] = omits or []

    @property
    def boost(self):
        return self._deploys["boost"]

    @property
    def color(self):
        return self._deploys["color"]

    @property
    def focus(self):
        return self._deploys["focus"]

    @property
    def target_size(self):
        return self._deploys["target_size"]

    @property
    def fps(self):
        return self._deploys["fps"]

    @property
    def compress_rate(self):
        return self._deploys["compress_rate"]

    @property
    def threshold(self):
        return self._deploys["threshold"]

    @property
    def offset(self):
        return self._deploys["offset"]

    @property
    def window_size(self):
        return self._deploys["window_size"]

    @property
    def step(self):
        return self._deploys["step"]

    @property
    def block(self):
        return self._deploys["block"]

    @property
    def window_coefficient(self):
        return self._deploys["window_coefficient"]

    @property
    def crops(self):
        return self._deploys["crops"]

    @property
    def omits(self):
        return self._deploys["omits"]

    def load_deploy(self, deploy_file: str) -> bool:
        is_load: bool = False
        try:
            with open(file=deploy_file, mode="r", encoding="utf-8") as f:
                data = json.loads(f.read())
                boost_mode = boost_data.lower() if isinstance(boost_data := data.get("boost", "false"), str) else "false"
                color_mode = color_data.lower() if isinstance(color_data := data.get("color", "false"), str) else "false"
                focus_mode = focus_data.lower() if isinstance(focus_data := data.get("focus", "false"), str) else "false"
                self._deploys["boost"] = True if boost_mode == "true" else False
                self._deploys["color"] = True if color_mode == "true" else False
                self._deploys["focus"] = True if focus_mode == "true" else False
                size = data.get("target_size", (350, 700))
                self._deploys["target_size"] = tuple(
                    max(100, min(3000, int(i))) for i in re.findall(r"-?\d*\.?\d+", size)
                ) if isinstance(size, str) else size
                self._deploys["fps"] = max(15, min(60, data.get("fps", 60)))
                self._deploys["compress_rate"] = max(0, min(1, data.get("compress_rate", 0.5)))
                self._deploys["threshold"] = max(0, min(1, data.get("threshold", 0.97)))
                self._deploys["offset"] = max(1, data.get("offset", 3))
                self._deploys["window_size"] = max(1, data.get("window_size", 1))
                self._deploys["step"] = max(1, data.get("step", 1))
                self._deploys["block"] = max(1, min(int(min(self.target_size[0], self.target_size[1]) / 10), data.get("block", 6)))
                self._deploys["window_coefficient"] = max(2, data.get("window_coefficient", 2))

                # Crops Hook
                crops_list = data.get("crops", [])
                for hook_dict in crops_list:
                    if len(
                            data_list := [
                                value for value in hook_dict.values() if isinstance(value, int | float)
                            ]
                    ) == 4 and sum(data_list) > 0:
                        self._deploys["crops"].append(
                            (hook_dict["x"], hook_dict["y"], hook_dict["x_size"], hook_dict["y_size"])
                        )
                if len(self.crops) >= 2:
                    self._deploys["crops"] = list(set(self.crops))

                # Omits Hook
                omits_list = data.get("omits", [])
                for hook_dict in omits_list:
                    if len(
                            data_list := [
                                value for value in hook_dict.values() if isinstance(value, int | float)
                            ]
                    ) == 4 and sum(data_list) > 0:
                        self._deploys["omits"].append(
                            (hook_dict["x"], hook_dict["y"], hook_dict["x_size"], hook_dict["y_size"])
                        )
                if len(self.omits) >= 2:
                    self._deploys["omits"] = list(set(self.omits))
        except FileNotFoundError:
            logger.debug("未找到部署文件,使用默认参数 ...")
        except json.decoder.JSONDecodeError:
            logger.debug("部署文件解析错误,文件格式不正确,使用默认参数 ...")
        else:
            logger.debug("读取部署文件,使用部署参数 ...")
            is_load = True
        finally:
            return is_load

    def dump_deploy(self, deploy_file: str) -> None:
        os.makedirs(os.path.dirname(deploy_file), exist_ok=True)

        with open(file=deploy_file, mode="w", encoding="utf-8") as f:
            f.writelines('{')
            for k, v in self._deploys.items():
                f.writelines('\n')
                if isinstance(v, bool):
                    f.writelines(f'    "{k}": "{v}",')
                elif k == "target_size":
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
            f"[bold {col_1_color}]快速模式",
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
            f"[bold {col_1_color}]视频转换",
            f"[bold {col_2_color}]{self.focus}",
            f"[bold][[bold {col_3_color}]T | F[/bold {col_3_color}] ]",
            f"[bold green]开启[/bold green]" if self.focus else "[bold red]关闭[/bold red]",
        )
        table.add_row(
            f"[bold {col_1_color}]图像尺寸",
            f"[bold {col_2_color}]{self.target_size}",
            f"[bold][[bold {col_3_color}]? , ?[/bold {col_3_color}] ]",
            f"[bold]宽 [bold red]{self.target_size[0]}[/bold red] 高 [bold red]{self.target_size[1]}[/bold red]",
        )
        table.add_row(
            f"[bold {col_1_color}]视频帧率",
            f"[bold {col_2_color}]{self.fps}",
            f"[bold][[bold {col_3_color}]15, 60[/bold {col_3_color}]]",
            f"[bold]转换视频为 [bold red]{self.fps}[/bold red] 帧每秒",
        )
        table.add_row(
            f"[bold {col_1_color}]压缩率",
            f"[bold {col_2_color}]{self.compress_rate}",
            f"[bold][[bold {col_3_color}]0 , 1[/bold {col_3_color}] ]",
            f"[bold]压缩视频大小为原来的 [bold red]{int(self.compress_rate * 100)}%[/bold red]",
        )
        table.add_row(
            f"[bold {col_1_color}]相似度",
            f"[bold {col_2_color}]{self.threshold}",
            f"[bold][[bold {col_3_color}]0 , 1[/bold {col_3_color}] ]",
            f"[bold]阈值超过 [bold red]{self.threshold}[/bold red] 的帧为稳定帧",
        )
        table.add_row(
            f"[bold {col_1_color}]补偿值",
            f"[bold {col_2_color}]{self.offset}",
            f"[bold][[bold {col_3_color}]0 , ?[/bold {col_3_color}] ]",
            f"[bold]合并 [bold red]{self.offset}[/bold red] 个变化不大的稳定区间",
        )
        table.add_row(
            f"[bold {col_1_color}]片段数量",
            f"[bold {col_2_color}]{self.window_size}",
            f"[bold][[bold {col_3_color}]1 , ?[/bold {col_3_color}] ]",
            f"[bold]每次处理 [bold red]{self.window_size}[/bold red] 个帧片段",
        )
        table.add_row(
            f"[bold {col_1_color}]处理数量",
            f"[bold {col_2_color}]{self.step}",
            f"[bold][[bold {col_3_color}]1 , ?[/bold {col_3_color}] ]",
            f"[bold]每个片段处理 [bold red]{self.step}[/bold red] 个帧图像",
        )
        table.add_row(
            f"[bold {col_1_color}]切分程度",
            f"[bold {col_2_color}]{self.block}",
            f"[bold][[bold {col_3_color}]1 , {int(min(self.target_size[0], self.target_size[1]) / 10)}[/bold {col_3_color}]]",
            f"[bold]每个帧图像切分为 [bold red]{self.block}[/bold red] 块",
        )
        table.add_row(
            f"[bold {col_1_color}]权重分布",
            f"[bold {col_2_color}]{self.window_coefficient}",
            f"[bold][[bold {col_3_color}]2 , ?[/bold {col_3_color}] ]",
            f"[bold]加权计算 [bold red]{self.window_coefficient}[/bold red]",
        )
        table.add_row(
            f"[bold {col_1_color}]获取区域",
            f"[bold {col_2_color}]{['!' for _ in range(len(self.crops))]}",
            f"[bold][[bold {col_3_color}]0 , 1[/bold {col_3_color}] ]",
            f"[bold]共 [bold red]{len(self.crops)}[/bold red] 个区域的图像参与计算",
        )
        table.add_row(
            f"[bold {col_1_color}]忽略区域",
            f"[bold {col_2_color}]{['!' for _ in range(len(self.omits))]}",
            f"[bold][[bold {col_3_color}]0 , 1[/bold {col_3_color}] ]",
            f"[bold]共 [bold red]{len(self.omits)}[/bold red] 个区域的图像不参与计算",
        )
        Show.console.print(table)


class Option(object):

    _options = {
        "Total Path": ""
    }

    @property
    def total_path(self):
        return self._options["Total Path"]

    @total_path.setter
    def total_path(self, value: str):
        self._options["Total Path"] = value

    def load_option(self, option_file: str) -> bool:
        is_load: bool = False
        try:
            with open(file=option_file, mode="r", encoding="utf-8") as f:
                data = json.loads(f.read())
                data_path = data.get("Total Path", "")
                if data_path and os.path.isdir(data_path):
                    if not os.path.exists(data_path):
                        os.makedirs(data_path, exist_ok=True)
                    self.total_path = data_path
        except FileNotFoundError:
            logger.debug("未找到配置文件,使用默认路径 ...")
        except json.decoder.JSONDecodeError:
            logger.debug("配置文件解析错误,文件格式不正确,使用默认路径 ...")
        else:
            logger.debug("读取配置文件,使用配置参数 ...")
            is_load = True
        finally:
            return is_load

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
