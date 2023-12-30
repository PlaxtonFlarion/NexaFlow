import os
import re
import sys
import cv2
import json
import time
import shutil
import random
import asyncio
import aiofiles
from loguru import logger
from rich.table import Table
from rich.prompt import Prompt
from rich.console import Console
from rich.progress import Progress

console = Console()
operation_system = sys.platform.strip().lower()
work_platform = os.path.basename(os.path.abspath(sys.argv[0])).lower()
exec_platform = ["framix.exe", "framix.bin", "framix", "framix.py"]

if work_platform == "framix.exe":
    _job_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    _universal = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
elif work_platform == "framix.bin":
    _job_path = os.path.dirname(sys.executable)
    _universal = os.path.dirname(os.path.dirname(sys.executable))
elif work_platform == "framix":
    _job_path = os.path.dirname(sys.executable)
    _universal = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))))
elif work_platform == "framix.py":
    _job_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _universal = os.path.dirname(os.path.abspath(__file__))
else:
    console.print("[bold red]Only compatible with Windows and macOS platforms ...")
    time.sleep(5)
    sys.exit(1)

_tools_path = os.path.join(_job_path, "archivix", "tools")
_model_path = os.path.join(_job_path, "archivix", "molds", "model.h5")
_total_path = os.path.join(_job_path, "archivix", "pages")
_major_path = os.path.join(_job_path, "archivix", "pages")
_proto_path = os.path.join(_job_path, "archivix", "pages", "template_extra.html")
_initial_report = os.path.join(_universal, "framix.report")
_initial_deploy = os.path.join(_universal, "framix.source")
_initial_option = os.path.join(_universal, "framix.source")

if operation_system == "win32":
    _adb = os.path.join(_tools_path, "win", "platform-tools", "adb.exe")
    _ffmpeg = os.path.join(_tools_path, "win", "ffmpeg", "bin", "ffmpeg.exe")
    _scrcpy = os.path.join(_tools_path, "win", "scrcpy", "scrcpy.exe")
elif operation_system == "darwin":
    _adb = os.path.join(_tools_path, "mac", "platform-tools", "adb")
    _ffmpeg = os.path.join(_tools_path, "mac", "ffmpeg", "bin", "ffmpeg")
    _scrcpy = os.path.join(_tools_path, "mac", "scrcpy", "bin", "scrcpy")
else:
    console.print("[bold red]Only compatible with Windows and macOS platforms ...")
    time.sleep(5)
    sys.exit(1)

os.environ["PATH"] = os.path.dirname(_adb) + os.path.pathsep + os.environ.get("PATH", "")
os.environ["PATH"] = os.path.dirname(_ffmpeg) + os.path.pathsep + os.environ.get("PATH", "")
os.environ["PATH"] = os.path.dirname(_scrcpy) + os.path.pathsep + os.environ.get("PATH", "")

try:
    from nexaflow import toolbox
    from nexaflow.terminal import Terminal
    from nexaflow.skills.report import Report
    from nexaflow.video import VideoObject, VideoFrame
    from nexaflow.cutter.cutter import VideoCutter
    from nexaflow.hook import OmitHook, FrameSaveHook, ShapeHook
    from nexaflow.classifier.keras_classifier import KerasClassifier
    from nexaflow.classifier.framix_classifier import FramixClassifier
except (RuntimeError, ModuleNotFoundError) as err:
    console.print(f"[bold]Error: {err}")
    time.sleep(5)
    sys.exit(1)


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
                hook_list = data.get("omits", [])
                for hook_dict in hook_list:
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
                elif k == "omits":
                    if len(v) == 0:
                        default = '{"x": 0, "y": 0, "x_size": 0, "y_size": 0}'
                        f.writelines(f'    "{k}": [\n')
                        f.writelines(f'        {default}\n')
                        f.writelines('    ]')
                    else:
                        f.writelines(f'    "{k}": [\n')
                        for index, i in enumerate(v):
                            x, y, x_size, y_size = i
                            new_size = f'{{"x": {x}, "y": {y}, "x_size": {x_size}, "y_size": {y_size}}}'
                            if (index + 1) == len(v):
                                f.writelines(f'        {new_size}\n')
                            else:
                                f.writelines(f'        {new_size},\n')
                        f.writelines('    ]')
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
            f"[bold {col_1_color}]忽略区域",
            f"[bold {col_2_color}]{['!' for _ in range(len(self.omits))]}",
            f"[bold][[bold {col_3_color}]0 , 1[/bold {col_3_color}] ]",
            f"[bold]共 [bold red]{len(self.omits)}[/bold red] 个区域的图像不参与计算",
        )

        console.print(table)


class Option(object):

    _options = {
        "Total Path": ""
    }

    @property
    def total_path(self):
        return self._options["Total Path"]

    @total_path.setter
    def total_path(self, value):
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


class Helper(object):

    @staticmethod
    def help_document():
        table_major = Table(
            title="[bold #FF851B]NexaFlow Framix Main Command Line",
            header_style="bold #FF851B", title_justify="center",
            show_header=True, show_lines=True
        )
        table_major.add_column("主要命令", justify="center", width=12)
        table_major.add_column("参数类型", justify="center", width=12)
        table_major.add_column("传递次数", justify="center", width=8)
        table_major.add_column("附加命令", justify="center", width=8)
        table_major.add_column("功能说明", justify="center", width=22)

        table_major.add_row(
            "[bold #FFDC00]--flick", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #D7FF00]支持", "[bold #39CCCC]录制分析视频帧"
        )
        table_major.add_row(
            "[bold #FFDC00]--alone", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "", "[bold #39CCCC]录制视频"
        )
        table_major.add_row(
            "[bold #FFDC00]--paint", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #D7FF00]支持", "[bold #39CCCC]绘制分割线条"
        )
        table_major.add_row(
            "[bold #FFDC00]--input", "[bold #7FDBFF]视频文件", "[bold #FFAFAF]多次", "[bold #D7FF00]支持", "[bold #39CCCC]分析单个视频"
        )
        table_major.add_row(
            "[bold #FFDC00]--whole", "[bold #7FDBFF]视频集合", "[bold #FFAFAF]多次", "[bold #D7FF00]支持", "[bold #39CCCC]分析全部视频"
        )
        table_major.add_row(
            "[bold #FFDC00]--merge", "[bold #7FDBFF]报告集合", "[bold #FFAFAF]多次", "", "[bold #39CCCC]聚合报告"
        )
        table_major.add_row(
            "[bold #FFDC00]--train", "[bold #7FDBFF]视频文件", "[bold #FFAFAF]多次", "", "[bold #39CCCC]归类图片文件"
        )
        table_major.add_row(
            "[bold #FFDC00]--build", "[bold #7FDBFF]图片集合", "[bold #FFAFAF]多次", "", "[bold #39CCCC]训练模型文件"
        )

        table_minor = Table(
            title="[bold #FF851B]NexaFlow Framix Extra Command Line",
            header_style="bold #FF851B", title_justify="center",
            show_header=True, show_lines=True
        )
        table_minor.add_column("附加命令", justify="center", width=12)
        table_minor.add_column("参数类型", justify="center", width=12)
        table_minor.add_column("传递次数", justify="center", width=8)
        table_minor.add_column("默认状态", justify="center", width=8)
        table_minor.add_column("功能说明", justify="center", width=22)

        table_minor.add_row(
            "[bold #FFDC00]--boost", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]快速模式"
        )
        table_minor.add_row(
            "[bold #FFDC00]--color", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]彩色模式"
        )
        table_minor.add_row(
            "[bold #FFDC00]--focus", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]转换视频"
        )
        table_minor.add_row(
            "[bold #FFDC00]--shape", "[bold #7FDBFF]数值", "[bold #8A8A8A]一次", "[bold #AFAFD7]自动", "[bold #39CCCC]图片尺寸"
        )
        table_minor.add_row(
            "[bold #FFDC00]--scale", "[bold #7FDBFF]数值", "[bold #8A8A8A]一次", "[bold #AFAFD7]自动", "[bold #39CCCC]缩放比例"
        )
        table_minor.add_row(
            "[bold #FFDC00]--omits", "[bold #7FDBFF]坐标", "[bold #FFAFAF]多次", "[bold #AFAFD7]自动", "[bold #39CCCC]忽略区域"
        )
        framix_logo = """[bold #D0D0D0]
              ███████╗██████╗  █████╗      ███╗   ███╗██╗██╗  ██╗
              ██╔════╝██╔══██╗██╔══██╗     ████╗ ████║██║╚██╗██╔╝
              █████╗  ██████╔╝███████║     ██╔████╔██║██║ ╚███╔╝
              ██╔══╝  ██╔══██╗██╔══██║     ██║╚██╔╝██║██║ ██╔██╗
              ██║     ██║  ██║██║  ██║     ██║ ╚═╝ ██║██║██╔╝ ██╗
              ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝     ╚═╝     ╚═╝╚═╝╚═╝  ╚═╝
        """
        nexaflow_logo = """[bold #D0D0D0]
    ███╗   ██╗███████╗██╗  ██╗ █████╗   ███████╗██╗      ██████╗ ██╗    ██╗
    ██╔██╗ ██║██╔════╝╚██╗██╔╝██╔══██╗  ██╔════╝██║     ██╔═══██╗██║    ██║
    ██║╚██╗██║█████╗   ╚███╔╝ ███████║  █████╗  ██║     ██║   ██║██║ █╗ ██║
    ██║ ╚████║██╔══╝   ██╔██╗ ██╔══██║  ██╔══╝  ██║     ██║   ██║██║███╗██║
    ██║  ╚███║███████╗██╔╝ ██╗██║  ██║  ██║     ███████╗╚██████╔╝╚███╔███╔╝
    ╚═╝   ╚══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝
        """

        console.print(nexaflow_logo)
        console.print(table_major)
        console.print(framix_logo)
        console.print(table_minor)
        with Progress() as progress:
            task = progress.add_task("[bold #FFFFD7]Framix Terminal Command.", total=100)
            while not progress.finished:
                progress.update(task, advance=1)
                time.sleep(0.1)

    @staticmethod
    def help_option():
        table = Table(show_header=True, header_style="bold #D7FF00", show_lines=True)
        table.add_column("选项", justify="center", width=12)
        table.add_column("参数", justify="center", width=12)
        table.add_column("说明", justify="center", width=44)
        table.add_row("[bold #FFAFAF]header", "[bold #AFD7FF]标题名", "[bold #DADADA]生成一个新标题文件夹")
        table.add_row("[bold #FFAFAF]serial", "", "[bold #DADADA]重新选择已连接的设备")
        table.add_row("[bold #FFAFAF]deploy", "", "[bold #DADADA]重新部署视频分析配置")
        table.add_row("[bold #FFAFAF]******", "", "[bold #DADADA]任意数字代表录制时长")
        console.print(table)


class Parser(object):

    @staticmethod
    def parse_cmd():

        def parse_shape(dim_str):
            if dim_str:
                shape = [int(i) for i in re.split(r'[\s,;]+', dim_str)]
                return tuple(shape) if len(shape) == 2 else (shape[0], shape[0])
            return None

        def parse_scale(dim_str):
            try:
                return int(dim_str)
            except ValueError:
                try:
                    return float(dim_str)
                except ValueError:
                    return None

        parser = ArgumentParser(description="Command Line Arguments Framix")

        parser.add_argument('--flick', action='store_true', help='录制分析视频帧')
        parser.add_argument('--alone', action='store_true', help='录制视频')
        parser.add_argument('--paint', action='store_true', help='绘制分割线条')
        parser.add_argument('--input', action='append', help='分析单个视频')
        parser.add_argument('--whole', action='append', help='分析全部视频')
        parser.add_argument('--merge', action='append', help='聚合报告')
        parser.add_argument('--train', action='append', help='归类图片文件')
        parser.add_argument('--build', action='append', help='训练模型文件')

        parser.add_argument('--boost', action='store_true', help='快速模式')
        parser.add_argument('--color', action='store_true', help='彩色模式')
        parser.add_argument('--focus', action='store_true', help='转换视频')
        parser.add_argument('--shape', nargs='?', const=None, type=parse_shape, help='图片尺寸')
        parser.add_argument('--scale', nargs='?', const=None, type=parse_scale, help='缩放比例')
        parser.add_argument('--omits', action='append', help='忽略区域')

        parser.add_argument('--debug', action='store_true', help='调试模式')

        return parser.parse_args()


class Missions(object):

    def __init__(self, *args, **kwargs):
        self.boost, self.color, self.focus, self.omits, self.shape, self.scale = args

        self.model_path = kwargs["model_path"]
        self.total_path = kwargs["total_path"]
        self.major_path = kwargs["major_path"]
        self.proto_path = kwargs["proto_path"]
        self.initial_report = kwargs["initial_report"]
        self.initial_deploy = kwargs["initial_deploy"]
        self.initial_option = kwargs["initial_option"]
        self.adb = kwargs["adb"]
        self.ffmpeg = kwargs["ffmpeg"]
        self.scrcpy = kwargs["scrcpy"]

    @staticmethod
    def only_video(folder: str):

        class Entry(object):

            def __init__(self, title: str, place: str, sheet: list):
                self.title = title
                self.place = place
                self.sheet = sheet

        return [
            Entry(
                os.path.basename(root), root,
                [os.path.join(root, f) for f in sorted(file) if "log" not in f]
            )
            for root, _, file in os.walk(folder) if file
        ]

    def video_task(self, input_video):
        reporter = Report(total_path=self.initial_report)
        reporter.title = f"Framix_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        reporter.query = f"{random.randint(10, 99)}"
        new_video_path = os.path.join(reporter.video_path, os.path.basename(input_video))

        shutil.copy(input_video, new_video_path)

        deploy = Deploy(
            boost=self.boost, color=self.color, focus=self.focus, target_size=self.shape,
            omits=self.omits
        )
        deploy.load_deploy(self.initial_deploy)

        kc = KerasClassifier(
            target_size=deploy.target_size, data_size=deploy.target_size
        )
        kc.load_model(self.model_path)

        looper = asyncio.get_event_loop()
        looper.run_until_complete(
            analyzer(
                reporter, kc, deploy, new_video_path,
                proto_path=self.proto_path,
                ffmpeg=self.ffmpeg
            )
        )
        looper.run_until_complete(
            reporter.ask_create_total_report(
                os.path.dirname(reporter.total_path), self.major_path, self.total_path
            )
        )

    def video_dir_task(self, folder):
        reporter = Report(total_path=self.initial_report)

        deploy = Deploy(
            boost=self.boost, color=self.color, focus=self.focus, target_size=self.shape,
            omits=self.omits
        )
        deploy.load_deploy(self.initial_deploy)

        kc = KerasClassifier(
            target_size=deploy.target_size, data_size=deploy.target_size
        )
        kc.load_model(self.model_path)

        looper = asyncio.get_event_loop()
        for video in self.only_video(folder):
            reporter.title = video.title
            for path in video.sheet:
                reporter.query = os.path.basename(path).split(".")[0]
                shutil.copy(path, reporter.video_path)
                new_video_path = os.path.join(reporter.video_path, os.path.basename(path))
                looper.run_until_complete(
                    analyzer(
                        reporter, kc, deploy, new_video_path,
                        proto_path=self.proto_path,
                        ffmpeg=self.ffmpeg
                    )
                )
        looper.run_until_complete(
            reporter.ask_create_total_report(
                os.path.dirname(reporter.total_path), self.major_path, self.total_path
            )
        )
        return reporter.total_path

    def train_model(self, video_file):
        reporter = Report(total_path=self.initial_report)
        reporter.title = f"Model_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}"
        if not os.path.exists(reporter.query_path):
            os.makedirs(reporter.query_path)

        deploy = Deploy(
            boost=self.boost, color=self.color, focus=self.focus, target_size=self.shape,
            omits=self.omits
        )
        deploy.load_deploy(self.initial_deploy)

        kc = KerasClassifier(
            target_size=deploy.target_size, data_size=deploy.target_size
        )
        kc.load_model(self.model_path)

        video_temp_file = os.path.join(reporter.query_path, f"tmp_fps60_{random.randint(100, 999)}.mp4")
        asyncio.run(ask_ffmpeg(self.ffmpeg, deploy.fps, video_file, video_temp_file))

        video = VideoObject(video_temp_file)
        video.load_frames()

        cutter = VideoCutter(
            step=deploy.step,
            compress_rate=deploy.compress_rate,
            target_size=deploy.target_size
        )
        res = cutter.cut(
            video=video,
            block=deploy.block,
            window_size=deploy.window_size,
            window_coefficient=deploy.window_coefficient
        )
        stable, unstable = res.get_range(
            threshold=deploy.threshold,
            offset=deploy.offset
        )
        res.pick_and_save(
            range_list=stable,
            frame_count=20,
            to_dir=reporter.query_path,
            meaningful_name=True
        )

        os.remove(video_temp_file)

    def build_model(self, src):
        if os.path.isdir(src):
            real_path, file_list = "", []
            logger.debug(f"搜索文件夹: {src}")
            for root, dirs, files in os.walk(src, topdown=False):
                for name in files:
                    file_list.append(os.path.join(root, name))
                for name in dirs:
                    if len(name) == 1 and re.search(r"0", name):
                        real_path = os.path.dirname(os.path.join(root, name))
                        logger.debug(f"分类文件夹: {real_path}")
                        break
            if real_path and len(file_list) > 0:
                new_model_path = os.path.join(real_path, f"Create_Model_{time.strftime('%Y%m%d%H%M%S')}")
                new_model_name = f"Keras_Model_{random.randint(10000, 99999)}.h5"

                deploy = Deploy(
                    boost=self.boost, color=self.color, focus=self.focus, target_size=self.shape,
                    omits=self.omits
                )
                deploy.load_deploy(self.initial_deploy)

                fc = FramixClassifier(data_size=deploy.target_size)
                fc.build(real_path, new_model_path, new_model_name)
            else:
                logger.error("文件夹未正确分类 ...")
        else:
            logger.error("训练模型需要一个分类文件夹 ...")

    async def combines(self, merge):
        tasks = [
            Report.ask_create_total_report(
                m, self.total_path, self.major_path
            ) for m in merge
        ]
        error = await asyncio.gather(*tasks)
        for e in error:
            if isinstance(e, Exception):
                logger.error(e)

    async def painting(self):
        import tempfile
        from PIL import Image, ImageDraw, ImageFont

        cellphone = await check_device(self.adb)
        image_folder = "/sdcard/Pictures/Shots"
        image = f"{time.strftime('%Y%m%d%H%M%S')}_{random.randint(100, 999)}_" + "Shot.png"
        await Terminal.cmd_line(self.adb, "-s", cellphone.serial, "wait-for-usb-device", "shell", "mkdir", "-p", image_folder)
        await Terminal.cmd_line(self.adb, "-s", cellphone.serial, "wait-for-usb-device", "shell", "screencap", "-p", f"{image_folder}/{image}")

        with tempfile.TemporaryDirectory() as temp_dir:
            image_save_path = os.path.join(temp_dir, image)
            await Terminal.cmd_line(self.adb, "-s", cellphone.serial, "wait-for-usb-device", "pull", f"{image_folder}/{image}", image_save_path)

            if self.color:
                old_image = toolbox.imread(image_save_path)
                new_image = VideoFrame(0, 0, old_image)
            else:
                old_image = toolbox.imread(image_save_path)
                old_image = toolbox.turn_grey(old_image)
                new_image = VideoFrame(0, 0, old_image)

            if len(self.omits) > 0:
                for omit in self.omits:
                    if len(omit) == 4 and sum(omit) > 0:
                        x, y, x_size, y_size = omit
                        shape_hook = ShapeHook((y_size, x_size), (y, x))
                        shape_hook.do(new_image)

            cv2.imencode(".png", new_image.data)[1].tofile(image_save_path)

            image_file = Image.open(image_save_path)
            image_file = image_file.convert("RGB")

            original_w, original_h = image_file.size
            if self.shape:
                shape_w, shape_h = self.shape
                twist_w, twist_h = min(original_w, shape_w), min(original_h, shape_h)
            else:
                twist_w, twist_h = original_w, original_h

            min_scale, max_scale = 0.3, 1.0
            if self.scale:
                image_scale = max_scale if self.scale > max_scale else (min_scale if self.scale < min_scale else self.scale)
            else:
                image_scale = min_scale if twist_w == original_w or twist_h == original_h else max_scale

            new_w, new_h = int(twist_w * image_scale), int(twist_h * image_scale)
            logger.debug(f"原始尺寸: {(original_w, original_h)} 调整尺寸: {(new_w, new_h)} 缩放比例: {int(image_scale * 100)}%")

            if new_w == new_h:
                x_line_num, y_line_num = 10, 10
            elif new_w > new_h:
                x_line_num, y_line_num = 10, 20
            else:
                x_line_num, y_line_num = 20, 10

            resized = image_file.resize((new_w, new_h))

            draw = ImageDraw.Draw(resized)
            font = ImageFont.load_default()

            if y_line_num > 0:
                for i in range(1, y_line_num):
                    x_line = int(new_w * (i * (1 / y_line_num)))
                    text = f"{i * int(100 / y_line_num):02}"
                    bbox = draw.textbbox((0, 0), text, font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    y_text_start = 3
                    draw.line([(x_line, text_width + 5 + y_text_start), (x_line, new_h)], fill=(0, 255, 255), width=1)
                    draw.text((x_line - text_height // 2, y_text_start), text, fill=(0, 255, 255), font=font)

            if x_line_num > 0:
                for i in range(1, x_line_num):
                    y_line = int(new_h * (i * (1 / x_line_num)))
                    text = f"{i * int(100 / x_line_num):02}"
                    bbox = draw.textbbox((0, 0), text, font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    x_text_start = 3
                    draw.line([(text_width + 5 + x_text_start, y_line), (new_w, y_line)], fill=(255, 182, 193), width=1)
                    draw.text((x_text_start, y_line - text_height // 2), text, fill=(255, 182, 193), font=font)

            resized.show()

        await Terminal.cmd_line(self.adb, "-s", cellphone.serial, "wait-for-usb-device", "shell", "rm", f"{image_folder}/{image}")

    async def analysis(self, alone: bool):

        head_event = asyncio.Event()
        done_event = asyncio.Event()
        stop_event = asyncio.Event()
        fail_event = asyncio.Event()

        async def timepiece(amount):
            while True:
                if head_event.is_set():
                    for i in range(amount):
                        if stop_event.is_set() and i != amount:
                            logger.warning(f"主动停止 ...")
                            logger.warning(f"剩余时间 -> 00 秒")
                            return
                        elif fail_event.is_set():
                            logger.warning(f"意外停止 ...")
                            logger.warning(f"剩余时间 -> 00 秒")
                            return
                        if amount - i <= 10:
                            logger.warning(f"剩余时间 -> {amount - i:02} 秒 {'----' * (amount - i)}")
                        else:
                            logger.warning(f"剩余时间 -> {amount - i:02} 秒 {'----' * 10} ...")
                        await asyncio.sleep(1)
                    logger.warning(f"剩余时间 -> 00 秒")
                    return
                elif fail_event.is_set():
                    logger.warning(f"意外停止 ...")
                    break
                await asyncio.sleep(0.2)

        async def input_stream(transports):
            async for line in transports.stdout:
                logger.info(stream := line.decode(encoding="UTF-8", errors="ignore").strip())
                if "Recording started" in stream:
                    head_event.set()
                elif "Recording complete" in stream:
                    stop_event.set()
                    done_event.set()
                    break

        async def error_stream(transports):
            async for line in transports.stderr:
                logger.info(stream := line.decode(encoding="UTF-8", errors="ignore").strip())
                if "Could not find" in stream or "connection failed" in stream or "Recorder error" in stream:
                    fail_event.set()
                    break

        async def start_record(serial: str, dst: str):
            temp_video = f"{os.path.join(dst, 'screen')}_{time.strftime('%Y%m%d%H%M%S')}_{random.randint(100, 999)}.mkv"
            cmd = [
                self.scrcpy, "-s", serial, "--no-audio", "--video-bit-rate", "8M", "--max-fps", deploy.fps, "--record", temp_video
            ]
            transports = await Terminal.cmd_link(*cmd)
            asyncio.create_task(input_stream(transports))
            asyncio.create_task(error_stream(transports))
            await asyncio.sleep(1)
            return temp_video, transports

        async def stop_record(temp_video, transports, solve: bool):
            if operation_system == "win32":
                await Terminal.cmd_line("taskkill", "/im", "scrcpy.exe")
            else:
                transports.terminate()
                await transports.wait()

            for _ in range(10):
                if done_event.is_set():
                    logger.success(f"视频录制成功: {temp_video}")
                    if solve:
                        await analyzer(
                            reporter, kc, deploy, temp_video,
                            proto_path=self.proto_path,
                            ffmpeg=self.ffmpeg
                        )
                    return
                elif fail_event.is_set():
                    break
                await asyncio.sleep(0.2)
            logger.error("录制视频失败,请重新录制视频 ...")

        async def start(serial):
            await Terminal.cmd_line(self.adb, "wait-for-device")
            if alone:
                temp_video, transports = await start_record(
                    serial, reporter.query_path
                )
                await timepiece(timer_mode)
                await stop_record(temp_video, transports, False)

            else:
                reporter.query = time.strftime('%Y%m%d%H%M%S')
                temp_video, transports = await start_record(
                    serial, reporter.video_path
                )
                await timepiece(timer_mode)
                await stop_record(temp_video, transports, True)

        # Start Tasks
        cellphone = await check_device(self.adb)

        reporter = Report(self.initial_report)
        if alone:
            reporter.title = f"Record_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        else:
            reporter.title = f"Framix_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

        deploy = Deploy(
            boost=self.boost, color=self.color, focus=self.focus, target_size=self.shape,
            omits=self.omits
        )
        deploy.load_deploy(self.initial_deploy)

        kc = KerasClassifier(
            target_size=deploy.target_size, data_size=deploy.target_size
        )
        kc.load_model(self.model_path)

        timer_mode = 5
        while True:
            try:
                console.print(f"[bold #00FFAF]Connect:[/bold #00FFAF] {cellphone}")
                if action := Prompt.ask(
                        prompt=f"[bold #5FD7FF]<<<按 Enter 开始 [bold #D7FF5F]{timer_mode}[/bold #D7FF5F] 秒>>>[/bold #5FD7FF]",
                        console=console
                ):
                    if "header" in action.strip():
                        if match := re.search(r"(?<=header\s).*", action):
                            if match.group().strip():
                                src_title = f"Record_{time.strftime('%Y%m%d_%H%M%S')}" if alone else f"Framix_{time.strftime('%Y%m%d_%H%M%S')}"
                                if title := match.group().strip():
                                    new_title = f"{src_title}_{title}"
                                else:
                                    new_title = f"{src_title}_{random.randint(10000, 99999)}"
                                logger.success("新标题设置成功 ...")
                                reporter.title = new_title
                            else:
                                raise ValueError
                        else:
                            raise ValueError
                        continue
                    elif action.strip() == "serial" and len(action.strip()) == 6:
                        cellphone = await check_device(self.adb)
                        continue
                    elif action.strip() == "deploy" and len(action.strip()) == 6:
                        deploy.dump_deploy(self.initial_deploy)
                        logger.warning("修改 deploy.json 文件后请完全退出编辑器进程再继续操作 ...")
                        if operation_system == "win32":
                            await Terminal.cmd_line("Notepad", self.initial_deploy)
                        else:
                            await Terminal.cmd_line("open", "-W", "-a", "TextEdit", self.initial_deploy)
                        deploy.omits.clear()
                        deploy.load_deploy(self.initial_deploy)
                        deploy.view_deploy()
                        continue
                    elif action.isdigit():
                        value, lower_bound, upper_bound = int(action), 5, 300
                        if value > 300 or value < 5:
                            console.print(
                                f"[bold #FFFF87]{lower_bound} <= [bold #FFD7AF]Time[/bold #FFD7AF] <= {upper_bound}[/bold #FFFF87]"
                            )
                        timer_mode = max(lower_bound, min(upper_bound, value))
                    else:
                        raise ValueError
            except ValueError:
                Helper.help_option()
            else:
                await start(cellphone.serial)
                if not done_event.is_set():
                    cellphone = await check_device(self.adb)
            finally:
                head_event.clear()
                done_event.clear()
                stop_event.clear()
                fail_event.clear()


def worker_init(log_level: str):
    logger.remove(0)
    log_format = "| <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sys.stderr, format=log_format, level=log_level.upper())


async def check_device(adb):

    class Phone(object):

        def __init__(self, *args):
            self.serial, self.brand, self.version, *_ = args

        def __str__(self):
            return f"<Phone brand={self.brand} version=OS{self.version} serial={self.serial}>"

        __repr__ = __str__

    async def check(serial):
        brand, version = await asyncio.gather(
            Terminal.cmd_line(adb, "-s", serial, "wait-for-usb-device", "shell", "getprop", "ro.product.brand"),
            Terminal.cmd_line(adb, "-s", serial, "wait-for-usb-device", "shell", "getprop", "ro.build.version.release")
        )
        return Phone(serial, brand, version)

    while True:
        devices = await Terminal.cmd_line(adb, "devices")
        if len(device_list := [i.split()[0] for i in devices.split("\n")[1:]]) == 1:
            return await check(device_list[0])
        elif len(device_list) > 1:
            console.print(f"[bold yellow]已连接多台设备[/bold yellow] {device_list}")
            device_dict = {}
            tasks = [check(serial) for serial in device_list]
            result = await asyncio.gather(*tasks)
            for idx, cur in enumerate(result):
                device_dict.update({str(idx + 1): cur})
                console.print(f"[{idx + 1}] {cur}")
            while True:
                try:
                    return device_dict[Prompt.ask("[bold #5FD7FF]请输入编号选择一台设备")]
                except KeyError:
                    console.print(f"[bold red]没有该序号,请重新选择 ...")
        else:
            console.print(f"[bold yellow]设备未连接,等待设备连接 ...")
            await asyncio.sleep(3)


async def ask_ffmpeg(ffmpeg, fps, src, dst):
    cmd = [
        ffmpeg,
        "-i", src, "-vf", f"fps={fps}", "-c:v", "libx264", "-crf", "18", "-c:a", "copy", dst
    ]
    await Terminal.cmd_line(*cmd)


async def analyzer(
        reporter: "Report",
        cl: "KerasClassifier",
        deploy: "Deploy",
        vision_path: str,
        **kwargs
):

    proto_path = kwargs["proto_path"]
    ffmpeg = kwargs["ffmpeg"]

    async def validate():
        screen_tag, screen_cap = None, None
        if os.path.isfile(vision_path):
            screen = cv2.VideoCapture(vision_path)
            if screen.isOpened():
                screen_tag = os.path.basename(vision_path)
                screen_cap = vision_path
            screen.release()
        elif os.path.isdir(vision_path):
            if len(
                    file_list := [
                        file for file in os.listdir(vision_path) if os.path.isfile(
                            os.path.join(vision_path, file)
                        )
                    ]
            ) > 1 or len(file_list) == 1:
                screen = cv2.VideoCapture(os.path.join(vision_path, file_list[0]))
                if screen.isOpened():
                    screen_tag = os.path.basename(file_list[0])
                    screen_cap = os.path.join(vision_path, file_list[0])
                screen.release()
        return screen_tag, screen_cap

    async def frame_flip():
        if deploy.focus:
            change_record = os.path.join(
                os.path.dirname(vision_path),
                f"screen_fps60_{random.randint(100, 999)}.mp4"
            )
            await ask_ffmpeg(ffmpeg, deploy.fps, vision_path, change_record)
            logger.info(f"视频转换完成: {os.path.basename(change_record)}")
            os.remove(vision_path)
            logger.info(f"移除旧的视频: {os.path.basename(vision_path)}")
        else:
            change_record = screen_record

        video = VideoObject(change_record)
        task, hued = video.load_frames(deploy.color)
        return video, task, hued

    async def frame_flow():
        video, task, hued = await frame_flip()
        cutter = VideoCutter(
            step=deploy.step,
            compress_rate=deploy.compress_rate,
            target_size=deploy.target_size
        )

        if len(deploy.omits) > 0:
            for omit in deploy.omits:
                x, y, x_size, y_size = omit
                omit_hook = OmitHook((y_size, x_size), (y, x))
                cutter.add_hook(omit_hook)
        save_hook = FrameSaveHook(reporter.extra_path)
        cutter.add_hook(save_hook)

        res = cutter.cut(
            video=video,
            block=deploy.block,
            window_size=deploy.window_size,
            window_coefficient=deploy.window_coefficient
        )

        stable, unstable = res.get_range(
            threshold=deploy.threshold,
            offset=deploy.offset
        )

        files = os.listdir(reporter.extra_path)
        files.sort(key=lambda n: int(n.split("(")[0]))
        total_images = len(files)
        interval = total_images // 11 if total_images > 12 else 1
        for index, file in enumerate(files):
            if index % interval != 0:
                os.remove(
                    os.path.join(reporter.extra_path, file)
                )

        draws = os.listdir(reporter.extra_path)
        for draw in draws:
            toolbox.draw_line(
                os.path.join(reporter.extra_path, draw)
            )

        classify = cl.classify(video=video, valid_range=stable, keep_data=True)

        important_frames = classify.get_important_frame_list()

        pbar = toolbox.show_progress(classify.get_length(), 50, "Faster")
        frames_list = []
        if deploy.boost:
            frames_list.append(previous := important_frames[0])
            pbar.update(1)
            for current in important_frames[1:]:
                frames_list.append(current)
                pbar.update(1)
                frames_diff = current.frame_id - previous.frame_id
                if not previous.is_stable() and not current.is_stable() and frames_diff > 1:
                    for specially in classify.data[previous.frame_id: current.frame_id - 1]:
                        frames_list.append(specially)
                        pbar.update(1)
                previous = current
            pbar.close()
        else:
            for current in classify.data:
                frames_list.append(current)
                pbar.update(1)
            pbar.close()

        if deploy.color:
            video.hued_data = tuple(hued.result())
            logger.info(f"彩色帧已加载: {video.frame_details(video.hued_data)}")
            task.shutdown()
            frames = [video.hued_data[frame.frame_id - 1] for frame in frames_list]
        else:
            frames = [frame for frame in frames_list]

        return classify, frames

    async def frame_flick(classify):
        try:
            start_frame = classify.get_not_stable_stage_range()[0][1]
            end_frame = classify.get_not_stable_stage_range()[-1][-1]
        except AssertionError:
            start_frame = classify.get_important_frame_list()[0]
            end_frame = classify.get_important_frame_list()[-1]

        if start_frame == end_frame:
            start_frame = classify.data[0]
            end_frame = classify.data[-1]

        time_cost = end_frame.timestamp - start_frame.timestamp
        before, after, final = f"{start_frame.timestamp:.5f}", f"{end_frame.timestamp:.5f}", f"{time_cost:.5f}"
        logger.info(f"图像分类结果: [开始帧: {before}] [结束帧: {after}] [总耗时: {final}]")

        with open(proto_path, mode="r", encoding="utf-8") as t:
            proto_file = t.read()
            original_inform = reporter.draw(
                classifier_result=classify,
                proto_path=reporter.proto_path,
                target_size=deploy.target_size,
                framix_template=proto_file
            )

        result = {
            "total_path": reporter.total_path,
            "title": reporter.title,
            "query_path": reporter.query_path,
            "query": reporter.query,
            "stage": {
                "start": start_frame.frame_id,
                "end": end_frame.frame_id,
                "cost": f"{time_cost:.5f}"
            },
            "frame": reporter.frame_path,
            "extra": reporter.extra_path,
            "proto": original_inform,
        }
        logger.debug(f"Restore: {result}")
        reporter.load(result)
        return before, after, final

    async def frame_forge(frame):
        try:
            short_timestamp = format(round(frame.timestamp, 5), ".5f")
            pic_name = f"{frame.frame_id}_{short_timestamp}.png"
            pic_path = os.path.join(reporter.frame_path, pic_name)
            _, codec = cv2.imencode(".png", frame.data)
            async with aiofiles.open(pic_path, "wb") as f:
                await f.write(codec.tobytes())
        except Exception as e:
            return e

    async def analytics():
        classify, frames = await frame_flow()

        if operation_system == "win32":
            logger.debug(f"运行环境: {operation_system}")
            flick_result, *forge_result = await asyncio.gather(
                frame_flick(classify), *(frame_forge(frame) for frame in frames),
                return_exceptions=True
            )
        else:
            logger.debug(f"运行环境: {operation_system}")
            tasks = [
                [frame_forge(frame) for frame in chunk]
                for chunk in
                [frames[i:i + 100] for i in range(0, len(frames), 100)]
            ]
            flick_task = asyncio.create_task(frame_flick(classify))
            forge_list = []
            for task in tasks:
                task_result = await asyncio.gather(*task, return_exceptions=True)
                forge_list.extend(task_result)
            forge_result = tuple(forge_list)
            flick_result = await flick_task

        for result in forge_result:
            if isinstance(result, Exception):
                logger.error(f"Error: {result}")

        return flick_result

    tag, screen_record = await validate()
    if not tag or not screen_record:
        logger.error(f"{tag} 不是一个标准的mp4视频文件，或视频文件已损坏 ...")
        return None
    logger.info(f"{tag} 可正常播放，准备加载视频 ...")

    start, end, cost = await analytics()
    return start, end, cost


async def main():
    if cmd_lines.flick or cmd_lines.alone:
        await missions.analysis(cmd_lines.alone)
    elif cmd_lines.paint:
        await missions.painting()
    elif cmd_lines.merge and len(cmd_lines.merge) > 0:
        await missions.combines(cmd_lines.merge)
    else:
        Helper.help_document()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        Helper.help_document()
        sys.exit(1)

    from multiprocessing import Pool, freeze_support
    freeze_support()

    from argparse import ArgumentParser
    cmd_lines = Parser.parse_cmd()
    _level = "DEBUG" if cmd_lines.debug else "INFO"
    worker_init(_level)

    # Debug Mode
    logger.debug(f"Level: {_level}")

    logger.debug(f"System: {operation_system}")
    logger.debug(f"Worker: {work_platform}")

    logger.debug(f"Tools: {_tools_path}")
    logger.debug(f"Model: {_model_path}")
    logger.debug(f"Html-Template: {_total_path}")
    logger.debug(f"Html-Template: {_major_path}")
    logger.debug(f"Html-Template: {_proto_path}")

    logger.debug(f"adb: {_adb}")
    logger.debug(f"ffmpeg: {_ffmpeg}")
    logger.debug(f"scrcpy: {_scrcpy}")

    for env in os.environ["PATH"].split(os.path.pathsep):
        logger.debug(env)

    _boost, _color, _focus = cmd_lines.boost, cmd_lines.color, cmd_lines.focus
    _shape, _scale = cmd_lines.shape, cmd_lines.scale

    cpu = os.cpu_count()
    logger.debug(f"CPU Core: {cpu}")

    _omits = []
    if cmd_lines.omits and len(cmd_lines.omits) > 0:
        for hook in cmd_lines.omits:
            if len(match_list := re.findall(r"-?\d*\.?\d+", hook)) == 4:
                valid_list = [float(num) if "." in num else int(num) for num in match_list]
                if sum(valid_list) > 0:
                    _omits.append(tuple(valid_list))
    if len(_omits) >= 2:
        _omits = list(set(_omits))

    _initial_deploy = os.path.join(_initial_deploy, "deploy.json")
    _initial_option = os.path.join(_initial_option, "option.json")

    option = Option()
    option.load_option(_initial_option)
    option.dump_option(_initial_option)
    _initial_report = option.total_path if option.total_path else _initial_report

    # Debug Mode
    logger.debug(f"Initial-Report: {_initial_report}")
    logger.debug(f"Initial-Deploy: {_initial_deploy}")
    logger.debug(f"Initial-Option: {_initial_option}")

    missions = Missions(
        _boost, _color, _focus, _omits, _shape, _scale,
        model_path=_model_path, total_path=_total_path, major_path=_major_path, proto_path=_proto_path,
        initial_report=_initial_report, initial_deploy=_initial_deploy, initial_option=_initial_option,
        adb=_adb, ffmpeg=_ffmpeg, scrcpy=_scrcpy,
    )

    if cmd_lines.whole and len(cmd_lines.whole) > 0:
        members = len(cmd_lines.whole)
        if members == 1:
            missions.video_dir_task(cmd_lines.whole[0])
        else:
            processes = members if members <= cpu else cpu
            with Pool(processes=processes, initializer=worker_init, initargs=("ERROR", )) as pool:
                results = pool.starmap(missions.video_dir_task, [(i, ) for i in cmd_lines.whole])
            Report.merge_report(results, _total_path)
        sys.exit(0)
    elif cmd_lines.input and len(cmd_lines.input) > 0:
        members = len(cmd_lines.input)
        if members == 1:
            missions.video_task(cmd_lines.input[0])
        else:
            processes = members if members <= cpu else cpu
            with Pool(processes=processes, initializer=worker_init, initargs=("ERROR", )) as pool:
                pool.starmap(missions.video_task, [(i, ) for i in cmd_lines.input])
        sys.exit(0)
    elif cmd_lines.train and len(cmd_lines.train) > 0:
        members = len(cmd_lines.train)
        if members == 1:
            missions.train_model(cmd_lines.train[0])
        else:
            processes = members if members <= cpu else cpu
            with Pool(processes=processes, initializer=worker_init, initargs=("ERROR", )) as pool:
                pool.starmap(missions.train_model, [(i, ) for i in cmd_lines.train])
        sys.exit(0)
    elif cmd_lines.build and len(cmd_lines.build) > 0:
        members = len(cmd_lines.build)
        if members == 1:
            missions.build_model(cmd_lines.build[0])
        else:
            processes = members if members <= cpu else cpu
            with Pool(processes=processes, initializer=worker_init, initargs=("ERROR", )) as pool:
                pool.starmap(missions.build_model, [(i, ) for i in cmd_lines.build])
        sys.exit(0)
    else:
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
            sys.exit(0)
        except KeyboardInterrupt:
            sys.exit(0)
