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
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)
from frameflow import argument
from nexaflow import const


class Show(object):

    console = Console()

    @staticmethod
    def notes(text: typing.Any):
        Show.console.print(f"[bold]{const.DESC} | Analyzer | {text}[/]")

    @staticmethod
    def annal(text: typing.Any):
        Show.console.print(f"[bold]{const.DESC} | Analyzer |[/]", Text(text, "bold"))

    @staticmethod
    def show_panel(level: str, text: typing.Any, wind: dict) -> None:
        if level == const.SHOW_LEVEL:
            panel = Panel(
                Text(f"{text}", **wind["文本"]), **wind["边框"], width=int(Show.console.width * 0.7)
            )
            Show.console.print(panel)

    @staticmethod
    def show_tree(path: str) -> None:
        tree = Tree(f"[link file://{path}]📁 {os.path.basename(path)}[/]", guide_style="bold blue")

        def add_nodes(current_node, current_path):
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

            add_nodes(tree, path)
            Show.console.print(tree)

    @staticmethod
    def show_progress():
        return Progress(
            TextColumn(text_format=f"[bold]{const.DESC} | {{task.description}} |", justify="right"),
            SpinnerColumn(
                style="bold #FFF68F", speed=1, finished_text="[bold #9AFF9A]Done"
            ),
            BarColumn(
                bar_width=int(Show.console.width * 0.4),
                style="bold #FF6347", complete_style="bold #FFEC8B", finished_style="bold #98FB98"
            ),
            TimeRemainingColumn(),
            "[progress.percentage][bold #E0FFFF]{task.completed:>5.0f}[/]/[bold #FFDAB9]{task.total}[/]",
            expand=False
        )

    @staticmethod
    def simulation_progress(desc: str):
        with Progress(
            TextColumn(text_format="[bold #FFFFD7]{task.description}", justify="right"),
            SpinnerColumn(
                style="bold #FFF68F", speed=1, finished_text="[bold #9AFF9A]Done"
            ),
            BarColumn(
                bar_width=int(Show.console.width * 0.4),
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
    def done():
        return f"""
    ╔════════════════════════════════╗
    ║       \033[1;32mMissions  Complete\033[0m       ║
    ╚════════════════════════════════╝

    <*=> \033[1;35m{const.DESC}\033[0m will now automatically exit <=*>
    <*=> \033[1;35m{const.DESC}\033[0m see you next <=*>
    """

    @staticmethod
    def fail():
        return f"""
    ╔════════════════════════════════╗
    ║        \033[1;31mMissions  Failed\033[0m        ║
    ╚════════════════════════════════╝

    <*=> \033[1;35m{const.DESC}\033[0m will now automatically exit <=*>
    <*=> \033[1;35m{const.DESC}\033[0m see you next <=*>
    """

    @staticmethod
    def exit():
        return f"""
    ╔════════════════════════════════╗
    ║        \033[1;33mMissions  Exited\033[0m        ║
    ╚════════════════════════════════╝

    <*=> \033[1;35m{const.DESC}\033[0m will now automatically exit <=*>
    <*=> \033[1;35m{const.DESC}\033[0m see you next <=*>
    """

    @staticmethod
    def major_logo():
        logo = """[bold #D0D0D0]
    ███╗   ██╗███████╗██╗  ██╗ █████╗   ███████╗██╗      ██████╗ ██╗    ██╗
    ██╔██╗ ██║██╔════╝╚██╗██╔╝██╔══██╗  ██╔════╝██║     ██╔═══██╗██║    ██║
    ██║╚██╗██║█████╗   ╚███╔╝ ███████║  █████╗  ██║     ██║   ██║██║ █╗ ██║
    ██║ ╚████║██╔══╝   ██╔██╗ ██╔══██║  ██╔══╝  ██║     ██║   ██║██║███╗██║
    ██║  ╚███║███████╗██╔╝ ██╗██║  ██║  ██║     ███████╗╚██████╔╝╚███╔███╔╝
    ╚═╝   ╚══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝
        """
        Show.console.print(logo)

    @staticmethod
    def minor_logo():
        logo = """[bold #D0D0D0]
            ███████╗ ██████╗   █████╗      ███╗   ███╗ ██╗ ██╗  ██╗
            ██╔════╝ ██╔══██╗ ██╔══██╗     ████╗ ████║ ██║ ╚██╗██╔╝
            █████╗   ██████╔╝ ███████║     ██╔████╔██║ ██║  ╚███╔╝
            ██╔══╝   ██╔══██╗ ██╔══██║     ██║╚██╔╝██║ ██║  ██╔██╗
            ██║      ██║  ██║ ██║  ██║     ██║ ╚═╝ ██║ ██║ ██╔╝ ██╗
            ╚═╝      ╚═╝  ╚═╝ ╚═╝  ╚═╝     ╚═╝     ╚═╝ ╚═╝ ╚═╝  ╚═╝
        """
        for line in logo.split("\n"):
            Show.console.print(line)
            time.sleep(0.05)
        Show.console.print(const.DECLARE)

    @staticmethod
    def help_document():
        for keys, values in argument.Args.ARGUMENT.items():
            description = "[bold #FFE4E1]互斥[/]" if keys in ["核心操控", "辅助利器", "视控精灵"] else "[bold #C1FFC1]兼容[/]"
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
            Show.console.print(table, "\t")

    @staticmethod
    def tips_document():
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
        Show.console.print(table)

    @staticmethod
    def load_animation():

        c = {
            1: "bold #D7AFAF", 2: "bold #5FD75F", 3: "bold #5FD7FF", 4: "bold #D7AF5F",
        }

        def speed_engine(stage):
            engine_stages = [
                Text("\n●", style=c[1]),
                Text("●——●", style=c[2]),
                Text("●——●——●", style=c[3]),
                Text("●——●——●——●\n", style=c[4]),
            ]
            return engine_stages[stage % len(engine_stages)]

        def basic_engine(stage):
            engine_stages = [
                Text("\n●", style=c[1]),
                Text("●——●", style=c[2]),
                Text("●——●——●", style=c[3]),
                Text("●——●——●——●", style=c[4]),
                Text("●——●——●——●——●", style=c[1]),
                Text("●——●——●——●——●——●", style=c[2]),
                Text("●——●——●——●——●——●——●", style=c[3]),
                Text("●——●——●——●——●——●——●——●\n", style=c[4])
            ]
            return engine_stages[stage % len(engine_stages)]

        def keras_engine(stage):
            engine_stages = [
                Text("""                  
                  (●)
                   |
                   |""", style=c[1]),
                Text("""         (●)------(●)
                   |       |
                   |       |""", style=c[2]),
                Text("""         (●)------(●)
                   | \\     |
                   |  \\    |
                  (●)---(●)""", style=c[3]),
                Text("""         (●)------(●)
                 / | \\   / |
                (●) (●)---(●)
                     |     |
                    (●)---(●)
                """, style=c[4])
            ]
            return engine_stages[stage % len(engine_stages)]

        def other_engine(stage):
            engine_stages = [
                Text("\n○   ○", style=c[1]),
                Text("○──┐○──┐", style=c[2]),
                Text("○──┤○──┤", style=c[3]),
                Text("○──┤○──┤◉\n", style=c[4])
            ]
            return engine_stages[stage % len(engine_stages)]

        def animation(step, secs, function):
            Show.notes(f"[bold #C1FFC1]Engine Initializing[/] ...")
            for i in range(step):
                Show.console.print(function(i), justify="left")
                time.sleep(secs)
            Show.notes(f"[bold #C1FFC1]Engine Loaded[/] ...")

        stochastic = [
            lambda: animation(4, 0.2, speed_engine),
            lambda: animation(8, 0.1, basic_engine),
            lambda: animation(4, 0.2, keras_engine),
            lambda: animation(4, 0.2, other_engine),
        ]
        random.choice(stochastic)()

    @staticmethod
    def content_pose(rlt, avg, dur, org, vd_start, vd_close, vd_limit, video_temp, frate):
        table_info = Table(
            title=f"[bold #F5F5DC]Video Info {os.path.basename(video_temp)}",
            header_style="bold #F5F5DC",
            title_justify="center",
            show_header=True,
            show_lines=True
        )
        table_info.add_column("视频尺寸", justify="left", width=14)
        table_info.add_column("实际帧率", justify="left", width=22)
        table_info.add_column("平均帧率", justify="left", width=22)
        table_info.add_column("转换帧率", justify="left", width=22)

        table_clip = Table(
            title=f"[bold #D8BFD8]Video Clip {os.path.basename(video_temp)}",
            header_style="bold #7FFFD4",
            title_justify="center",
            show_header=True,
            show_lines=True
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

        Show.console.print(table_info)
        Show.console.print(table_clip)

    @staticmethod
    def assort_frame(begin_fr, final_fr, stage_cs):
        table = Table(
            title=f"[bold #EED5D2]{const.DESC} Assort Frame",
            header_style="bold #D3D3D3",
            title_justify="center",
            show_header=True,
            show_lines=True
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

        Show.console.print(table)


if __name__ == '__main__':
    pass
