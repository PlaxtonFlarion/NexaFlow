#
#   ____  _
#  / ___|| |__   _____      __
#  \___ \| '_ \ / _ \ \ /\ / /
#   ___) | | | | (_) \ V  V /
#  |____/|_| |_|\___/ \_/\_/
#

import os
import time
import random
import typing
from rich.tree import Tree
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn,
    SpinnerColumn, TimeRemainingColumn
)
from frameflow import argument
from nexaflow import const


class Design(object):
    """
    负责 CLI 界面展示与美化输出的核心类。

    提供丰富的终端展示方法，包括彩色日志输出、面板渲染、目录树展示、
    进度条、动画加载、控制台表格等，依赖 `rich` 库实现高可读性和美学化的 CLI 交互体验。

    Attributes
    ----------
    console : Optional[Console]
        rich 控制台对象，用于渲染文本、表格、面板和动画。

    design_level : str
        日志等级。
    """

    console: typing.Optional["Console"] = Console()

    def __init__(self, design_level: str):
        self.design_level = design_level

    @staticmethod
    def notes(text: typing.Any) -> None:
        """输出常规日志信息，使用 bold 样式强调。"""
        Design.console.print(f"[bold]{const.DESC} | Analyzer | {text}[/]")

    @staticmethod
    def annal(text: typing.Any) -> None:
        """输出结构化强调文本，适用于模型状态或分析摘要。"""
        Design.console.print(f"[bold]{const.DESC} | Analyzer |[/]", Text(text, "bold"))

    def show_panel(self, text: typing.Any, wind: dict) -> None:
        """根据日志等级和样式参数渲染面板式输出。"""
        if self.design_level == const.SHOW_LEVEL:
            panel = Panel(
                Text(
                    f"{text}", **wind["文本"]
                ), **wind["边框"], width=int(Design.console.width * 0.7)
            )
            Design.console.print(panel)

    @staticmethod
    def show_tree(file_path: str) -> None:
        """构建并展示文件夹结构树，支持视频文件和子目录的可视链接。"""

        def add_nodes(current_node: "Tree", current_path: str) -> None:
            try:
                with os.scandir(current_path) as scamper:
                    for cur in scamper:
                        folder_path = cur.path
                        if cur.is_dir():
                            sub_node = current_node.add(
                                f"[link file://{folder_path}]📁 {cur.name}[/]", guide_style="bold green"
                            )
                            add_nodes(sub_node, folder_path)
                        elif cur.is_file() and cur.name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                            current_node.add(f"[link file://{folder_path}]🎥 {cur.name}[/]")
            except PermissionError:
                current_node.add("[red]Access denied[/]", style="bold red")

        tree = Tree(
            f"[link file://{file_path}]📁 {os.path.basename(file_path)}[/]", guide_style="bold blue"
        )
        add_nodes(tree, file_path)
        Design.console.print(tree)

    @staticmethod
    def show_progress() -> "Progress":
        """创建并返回自定义进度条组件，适用于异步任务的状态展示。"""
        return Progress(
            TextColumn(text_format=f"[bold]{const.DESC} | {{task.description}} |", justify="right"),
            SpinnerColumn(
                style="bold #FFF68F", speed=1, finished_text="[bold #9AFF9A]Done"
            ),
            BarColumn(
                bar_width=int(Design.console.width * 0.4),
                style="bold #FF6347", complete_style="bold #FFEC8B", finished_style="bold #98FB98"
            ),
            TimeRemainingColumn(),
            "[progress.percentage][bold #E0FFFF]{task.completed:>5.0f}[/]/[bold #FFDAB9]{task.total}[/]",
            expand=False
        )

    @staticmethod
    def simulation_progress(desc: str) -> None:
        """启动模拟进度条，用于快速任务的视觉反馈。"""
        with Progress(
            TextColumn(text_format="[bold #FFFFD7]{task.description}", justify="right"),
            SpinnerColumn(
                style="bold #FFF68F", speed=1, finished_text="[bold #9AFF9A]Done"
            ),
            BarColumn(
                bar_width=int(Design.console.width * 0.4),
                style="bold #FF6347", complete_style="bold #FFEC8B", finished_style="bold #98FB98"
            ),
            TimeRemainingColumn(),
            "[progress.percentage][bold #E0FFFF]{task.percentage:>5.0f}%[/]",
            expand=False
        ) as progress:
            task = progress.add_task(desc, total=100)
            while not progress.finished:
                progress.update(task, advance=1)
                time.sleep(0.05)

    @staticmethod
    def done() -> None:
        """显示任务完成状态的 ASCII 区块框提示。"""
        Design.console.print(f"""[bold]
    ╔══════════════════════════════════╗
    ║          [bold #00FF00]Missions  Done[/]          ║
    ╚══════════════════════════════════╝""")

    @staticmethod
    def fail() -> None:
        """显示任务失败状态的 ASCII 区块框提示。"""
        Design.console.print(f"""[bold]
    ╔══════════════════════════════════╗
    ║          [bold #FF0000]Missions  Fail[/]          ║
    ╚══════════════════════════════════╝""")

    @staticmethod
    def exit() -> None:
        """显示任务退出状态的 ASCII 区块框提示。"""
        Design.console.print(f"""[bold]
    ╔══════════════════════════════════╗
    ║          [bold #FFFF00]Missions  Exit[/]          ║
    ╚══════════════════════════════════╝""")

    @staticmethod
    def closure() -> str:
        """返回格式化的退出提示文本。"""
        return f"""
    <*=> {const.DESC} will now automatically exit <=*>
    <*=> {const.DESC} see you next <=*>
        """

    @staticmethod
    def major_logo() -> None:
        """打印主 Logo（带 ASCII 图形和配色），适用于程序启动欢迎界面。"""
        logo = """[bold #D0D0D0]
    ███╗   ██╗███████╗██╗  ██╗ █████╗   ███████╗██╗      ██████╗ ██╗    ██╗
    ██╔██╗ ██║██╔════╝╚██╗██╔╝██╔══██╗  ██╔════╝██║     ██╔═══██╗██║    ██║
    ██║╚██╗██║█████╗   ╚███╔╝ ███████║  █████╗  ██║     ██║   ██║██║ █╗ ██║
    ██║ ╚████║██╔══╝   ██╔██╗ ██╔══██║  ██╔══╝  ██║     ██║   ██║██║███╗██║
    ██║  ╚███║███████╗██╔╝ ██╗██║  ██║  ██║     ███████╗╚██████╔╝╚███╔███╔╝
    ╚═╝   ╚══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝
        """
        Design.console.print(logo)

    @staticmethod
    def minor_logo() -> None:
        """打印次 Logo，逐行动态加载并附带版权信息。"""
        logo = """[bold #D0D0D0]
            ███████╗ ██████╗   █████╗      ███╗   ███╗ ██╗ ██╗  ██╗
            ██╔════╝ ██╔══██╗ ██╔══██╗     ████╗ ████║ ██║ ╚██╗██╔╝
            █████╗   ██████╔╝ ███████║     ██╔████╔██║ ██║  ╚███╔╝
            ██╔══╝   ██╔══██╗ ██╔══██║     ██║╚██╔╝██║ ██║  ██╔██╗
            ██║      ██║  ██║ ██║  ██║     ██║ ╚═╝ ██║ ██║ ██╔╝ ██╗
            ╚═╝      ╚═╝  ╚═╝ ╚═╝  ╚═╝     ╚═╝     ╚═╝ ╚═╝ ╚═╝  ╚═╝
        """
        for line in logo.split("\n"):
            Design.console.print(line)
            time.sleep(0.05)
        Design.console.print(const.DECLARE)

    @staticmethod
    def help_document() -> None:
        """展示命令行参数文档（来自 ARGUMENT 配置），以表格形式高亮各类参数分组。"""
        for keys, values in argument.Args.ARGUMENT.items():
            description = "[bold #FFE4E1]参数互斥[/]" if keys in [
                "核心操控", "辅助利器", "视控精灵"] else "[bold #C1FFC1]参数兼容[/]"

            table = Table(
                title=f"[bold #FFDAB9]{const.ITEM} {const.DESC} CLI [bold #66CDAA]<{keys}>[/] <{description}>",
                header_style="bold #FF851B",
                title_justify="center",
                show_header=True,
                show_lines=True
            )
            table.add_column("命令参数", justify="left", width=14)
            table.add_column("参数类型", justify="left", width=12)
            table.add_column("传递次数", justify="left", width=10)
            table.add_column("功能说明", justify="left", width=30)
            information = [
                [key, *value["view"], value["help"]] for key, value in values.items()
            ]
            for info in information:
                cmds, kind, push, desc = info
                push_color = "[bold #FFAFAF]" if push == "多次" else "[bold #CFCFCF]"
                table.add_row(
                    *[f"[bold #FFDC00]{cmds}", f"[bold #7FDBFF]{kind}", f"{push_color}{push}", f"[bold #39CCCC]{desc}"]
                )
            Design.console.print(table, "\t")

    @staticmethod
    def tips_document() -> None:
        """显示简化参数提示文档，适用于交互式命令输入提示。"""
        table = Table(
            title=f"[bold #FFDAB9]{const.ITEM} {const.DESC} CLI",
            header_style="bold #FF851B",
            title_justify="center",
            show_header=True,
            show_lines=True
        )
        table.add_column("选项", justify="left", width=12)
        table.add_column("参数", justify="left", width=12)
        table.add_column("说明", justify="left", width=12)

        information = [
            ["[bold #FFAFAF]header", "[bold #AFD7FF]标题名", "[bold #FFD39B]生成标题"],
            ["[bold #FFAFAF]device", "[bold #CFCFCF]无参数", "[bold #FFD39B]连接设备"],
            ["[bold #FFAFAF]deploy", "[bold #CFCFCF]无参数", "[bold #FFD39B]部署配置"],
            ["[bold #FFAFAF]create", "[bold #CFCFCF]无参数", "[bold #FFD39B]生成报告"],
            ["[bold #FFAFAF]cancel", "[bold #CFCFCF]无参数", "[bold #FFD39B]退出"]
        ]
        for info in information:
            table.add_row(*info)
        Design.console.print(table)

    @staticmethod
    def load_animation() -> None:
        """随机展示启动动画，包括多种渐进式加载风格（点阵、图解等）。"""

        colors = {
            1: "bold #D7AFAF", 2: "bold #5FD75F", 3: "bold #5FD7FF", 4: "bold #D7AF5F"
        }

        def speed_engine(stage):
            engine_stages = [
                Text("\n●", style=colors[1]),
                Text("●——●", style=colors[2]),
                Text("●——●——●", style=colors[3]),
                Text("●——●——●——●\n", style=colors[4]),
            ]
            return engine_stages[stage % len(engine_stages)]

        def basic_engine(stage):
            engine_stages = [
                Text("\n●", style=colors[1]),
                Text("●——●", style=colors[2]),
                Text("●——●——●", style=colors[3]),
                Text("●——●——●——●", style=colors[4]),
                Text("●——●——●——●——●", style=colors[1]),
                Text("●——●——●——●——●——●", style=colors[2]),
                Text("●——●——●——●——●——●——●", style=colors[3]),
                Text("●——●——●——●——●——●——●——●\n", style=colors[4])
            ]
            return engine_stages[stage % len(engine_stages)]

        def keras_engine(stage):
            engine_stages = [
                Text("""                  
                  (●)
                   |
                   |""", style=colors[1]),
                Text("""         (●)------(●)
                   |       |
                   |       |""", style=colors[2]),
                Text("""         (●)------(●)
                   | \\     |
                   |  \\    |
                  (●)---(●)""", style=colors[3]),
                Text("""         (●)------(●)
                 / | \\   / |
                (●) (●)---(●)
                     |     |
                    (●)---(●)
                """, style=colors[4])
            ]
            return engine_stages[stage % len(engine_stages)]

        def other_engine(stage):
            engine_stages = [
                Text("\n○   ○", style=colors[1]),
                Text("○──┐○──┐", style=colors[2]),
                Text("○──┤○──┤", style=colors[3]),
                Text("○──┤○──┤◉\n", style=colors[4])
            ]
            return engine_stages[stage % len(engine_stages)]

        def animation(step, secs, function):
            Design.notes(f"[bold #C1FFC1]Engine Initializing[/] ...")
            for i in range(step):
                Design.console.print(function(i), justify="left")
                time.sleep(secs)
            Design.notes(f"[bold #C1FFC1]Engine Loaded[/] ...")

        stochastic = [
            lambda: animation(4, 0.2, speed_engine),
            lambda: animation(8, 0.1, basic_engine),
            lambda: animation(4, 0.2, keras_engine),
            lambda: animation(4, 0.2, other_engine),
        ]
        random.choice(stochastic)()

    def content_pose(self, rlt, avg, dur, org, vd_start, vd_close, vd_limit, video_temp, frate) -> None:
        """根据日志等级展示当前视频处理过程中的关键帧率与时长信息。"""
        if self.design_level == const.SHOW_LEVEL:
            table_style = {
                "title_justify": "center", "show_header": True, "show_lines": True
            }

            table_info = Table(
                title=f"[bold #F5F5DC]Video Info {os.path.basename(video_temp)}",
                header_style="bold #F5F5DC", **table_style
            )
            table_info.add_column("视频尺寸", justify="left", width=14)
            table_info.add_column("实际帧率", justify="left", width=22)
            table_info.add_column("平均帧率", justify="left", width=22)
            table_info.add_column("转换帧率", justify="left", width=22)

            table_clip = Table(
                title=f"[bold #D8BFD8]Video Clip {os.path.basename(video_temp)}",
                header_style="bold #7FFFD4", **table_style
            )
            table_clip.add_column("视频时长", justify="left", width=14)
            table_clip.add_column("开始时间", justify="left", width=22)
            table_clip.add_column("结束时间", justify="left", width=22)
            table_clip.add_column("持续时间", justify="left", width=22)

            info_list = [
                f"[bold #87CEEB]{org}", f"[bold #87CEEB]{rlt}",
                f"[bold #87CEEB]{avg}", f"[bold #87CEEB]{frate}"
            ]
            table_info.add_row(*info_list)

            clip_list = [
                f"[bold #87CEEB]{dur}",
                f"[bold][[bold #EE82EE]{vd_start}[/]][/]",
                f"[bold][[bold #EE82EE]{vd_close}[/]][/]",
                f"[bold][[bold #EE82EE]{vd_limit}[/]][/]"
            ]
            table_clip.add_row(*clip_list)

            Design.console.print(table_info)
            Design.console.print(table_clip)

    def assort_frame(self, begin_fr, final_fr, stage_cs) -> None:
        """根据日志等级输出帧片段处理的起止帧号及耗时统计。"""
        if self.design_level == const.SHOW_LEVEL:
            table_style = {
                "title_justify": "center", "show_header": True, "show_lines": True
            }

            table = Table(
                title=f"[bold #EED5D2]{const.DESC} Assort Frame",
                header_style="bold #D3D3D3", **table_style
            )
            table.add_column("开始帧", justify="left", width=22)
            table.add_column("结束帧", justify="left", width=22)
            table.add_column("总耗时", justify="left", width=22)

            assort_list = [
                f"[bold][[bold #C1FFC1]{begin_fr}[/]][/]",
                f"[bold][[bold #FF4040]{final_fr}[/]][/]",
                f"[bold][[bold #F4A460]{stage_cs}[/]][/]"
            ]
            table.add_row(*assort_list)

            Design.console.print(table)


if __name__ == '__main__':
    pass
