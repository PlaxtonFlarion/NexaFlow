import os
import time
import random
import typing
from loguru import logger
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
from nexaflow import const


class Show(object):

    console = Console()

    @staticmethod
    def show_panel(title: typing.Any, write: typing.Any, tc: typing.Any, bc: typing.Any, jf: typing.Any) -> None:
        """
        展示控制板
        @param title: 标题
        @param write: 文本
        @param tc: 文本颜色
        @param bc: 边框颜色
        @param jf: 对齐方式
        @return:
        """
        panel = Panel(
            Text(write, style=f"bold {tc}", justify=jf),
            title=f"{title}",
            border_style=f"bold {bc}",
            width=int(Show.console.width * 0.7)
        )
        Show.console.print(panel)

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
    def simulation_progress(desc: str, advance: int | float, interval: int | float):
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
                progress.update(task, advance=advance)
                time.sleep(interval)

    @staticmethod
    def done():
        return f"""
    \033[1m╔════════════════════════════════╗
    ║       \033[1m\033[32mMissions  Complete\033[0m       ║
    ╚════════════════════════════════╝

    ✦✦✦ \033[35m{const.DESC}\033[0m will now automatically exit ✦✦✦
    ✧✧✧ \033[35m{const.DESC}\033[0m see you next ✧✧✧\033[0m
    """

    @staticmethod
    def fail():
        return f"""
    \033[1m╔════════════════════════════════╗
    ║        \033[1m\033[31mMissions  Failed\033[0m        ║
    ╚════════════════════════════════╝

    ✦✦✦ \033[35m{const.DESC}\033[0m will now automatically exit ✦✦✦
    ✧✧✧ \033[35m{const.DESC}\033[0m see you next ✧✧✧\033[0m
    """

    @staticmethod
    def exit():
        return f"""
    \033[1m╔════════════════════════════════╗
    ║        \033[1m\033[33mMissions  Exited\033[0m        ║
    ╚════════════════════════════════╝

    ✦✦✦ \033[35m{const.DESC}\033[0m will now automatically exit ✦✦✦
    ✧✧✧ \033[35m{const.DESC}\033[0m see you next ✧✧✧\033[0m
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
              ███████╗██████╗  █████╗      ███╗   ███╗██╗██╗  ██╗
              ██╔════╝██╔══██╗██╔══██╗     ████╗ ████║██║╚██╗██╔╝
              █████╗  ██████╔╝███████║     ██╔████╔██║██║ ╚███╔╝
              ██╔══╝  ██╔══██╗██╔══██║     ██║╚██╔╝██║██║ ██╔██╗
              ██║     ██║  ██║██║  ██║     ██║ ╚═╝ ██║██║██╔╝ ██╗
              ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝     ╚═╝     ╚═╝╚═╝╚═╝  ╚═╝
        """
        Show.console.print(logo)

    @staticmethod
    def help_document():
        table_major = Table(
            title=f"[bold #FF851B]{const.ITEM} {const.DESC} Main Command Line",
            header_style="bold #FF851B",
            title_justify="center",
            show_header=True,
            show_lines=True
        )
        table_major.add_column("主要命令", justify="center", width=12)
        table_major.add_column("参数类型", justify="center", width=12)
        table_major.add_column("传递次数", justify="center", width=8)
        table_major.add_column("附加命令", justify="center", width=8)
        table_major.add_column("功能说明", justify="center", width=22)

        table_minor = Table(
            title=f"[bold #FF851B]{const.ITEM} {const.DESC} Extra Command Line",
            header_style="bold #FF851B",
            title_justify="center",
            show_header=True,
            show_lines=True
        )
        table_minor.add_column("附加命令", justify="center", width=12)
        table_minor.add_column("参数类型", justify="center", width=12)
        table_minor.add_column("传递次数", justify="center", width=8)
        table_minor.add_column("默认状态", justify="center", width=8)
        table_minor.add_column("功能说明", justify="center", width=22)

        major_information = [
            ["[bold #FFDC00]--video", "[bold #7FDBFF]视频文件", "[bold #FFAFAF]多次", "[bold #D7FF00]支持", "[bold #39CCCC]分析视频文件"],
            ["[bold #FFDC00]--stack", "[bold #7FDBFF]视频集合", "[bold #FFAFAF]多次", "[bold #D7FF00]支持", "[bold #39CCCC]分析视频集合"],
            ["[bold #FFDC00]--train", "[bold #7FDBFF]视频文件", "[bold #FFAFAF]多次", "[bold #D7FF00]支持", "[bold #39CCCC]训练模型"],
            ["[bold #FFDC00]--build", "[bold #7FDBFF]图片集合", "[bold #FFAFAF]多次", "[bold #D7FF00]支持", "[bold #39CCCC]编译模型"],
            ["[bold #FFDC00]--flick", "[bold #7FDBFF]命令参数", "[bold #8A8A8A]一次", "[bold #D7FF00]支持", "[bold #39CCCC]循环运行模式"],
            ["[bold #FFDC00]--carry", "[bold #7FDBFF]脚本名称", "[bold #FFAFAF]多次", "[bold #D7FF00]支持", "[bold #39CCCC]运行指定脚本"],
            ["[bold #FFDC00]--fully", "[bold #7FDBFF]文件路径", "[bold #FFAFAF]多次", "[bold #D7FF00]支持", "[bold #39CCCC]运行全部脚本"],
            ["[bold #FFDC00]--paint", "[bold #7FDBFF]命令参数", "[bold #8A8A8A]一次", "[bold #D7FF00]支持", "[bold #39CCCC]绘制分割线条"],
            ["[bold #FFDC00]--union", "[bold #7FDBFF]报告集合", "[bold #FFAFAF]多次", "[bold #D7FF00]支持", "[bold #39CCCC]聚合视频帧报告"],
            ["[bold #FFDC00]--merge", "[bold #7FDBFF]报告集合", "[bold #FFAFAF]多次", "[bold #D7FF00]支持", "[bold #39CCCC]聚合时间戳报告"]
        ]

        minor_information = [
            ["[bold #FFDC00]--speed", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]快速模式"],
            ["[bold #FFDC00]--basic", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]基础模式"],
            ["[bold #FFDC00]--keras", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]智能模式"],
            ["[bold #FFDC00]--alone", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]独立控制"],
            ["[bold #FFDC00]--whist", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]静默录制"],
            ["[bold #FFDC00]--alike", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]平衡时间"],
            ["[bold #FFDC00]--group", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]分组报告"],
            ["[bold #FFDC00]--boost", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]跳帧模式"],
            ["[bold #FFDC00]--color", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]彩色模式"],
            ["[bold #FFDC00]--shape", "[bold #7FDBFF]数值", "[bold #8A8A8A]一次", "[bold #4CAF50]自动", "[bold #39CCCC]图片尺寸"],
            ["[bold #FFDC00]--scale", "[bold #7FDBFF]数值", "[bold #8A8A8A]一次", "[bold #4CAF50]自动", "[bold #39CCCC]缩放比例"],
            ["[bold #FFDC00]--start", "[bold #7FDBFF]数值", "[bold #8A8A8A]一次", "[bold #4CAF50]自动", "[bold #39CCCC]开始时间"],
            ["[bold #FFDC00]--close", "[bold #7FDBFF]数值", "[bold #8A8A8A]一次", "[bold #4CAF50]自动", "[bold #39CCCC]结束时间"],
            ["[bold #FFDC00]--limit", "[bold #7FDBFF]数值", "[bold #8A8A8A]一次", "[bold #4CAF50]自动", "[bold #39CCCC]持续时间"],
            ["[bold #FFDC00]--begin", "[bold #7FDBFF]数值", "[bold #8A8A8A]一次", "[bold #4CAF50]自动", "[bold #39CCCC]开始阶段"],
            ["[bold #FFDC00]--final", "[bold #7FDBFF]数值", "[bold #8A8A8A]一次", "[bold #4CAF50]自动", "[bold #39CCCC]结束阶段"],
            ["[bold #FFDC00]--frate", "[bold #7FDBFF]数值", "[bold #8A8A8A]一次", "[bold #4CAF50]自动", "[bold #39CCCC]帧采样率"],
            ["[bold #FFDC00]--thres", "[bold #7FDBFF]数值", "[bold #8A8A8A]一次", "[bold #4CAF50]自动", "[bold #39CCCC]相似度"],
            ["[bold #FFDC00]--shift", "[bold #7FDBFF]数值", "[bold #8A8A8A]一次", "[bold #4CAF50]自动", "[bold #39CCCC]补偿值"],
            ["[bold #FFDC00]--block", "[bold #7FDBFF]数值", "[bold #8A8A8A]一次", "[bold #4CAF50]自动", "[bold #39CCCC]立方体"],
            ["[bold #FFDC00]--crops", "[bold #7FDBFF]坐标", "[bold #FFAFAF]多次", "[bold #4CAF50]自动", "[bold #39CCCC]获取区域"],
            ["[bold #FFDC00]--omits", "[bold #7FDBFF]坐标", "[bold #FFAFAF]多次", "[bold #4CAF50]自动", "[bold #39CCCC]忽略区域"]
        ]

        for major in major_information:
            table_major.add_row(*major)

        for minor in minor_information:
            table_minor.add_row(*minor)

        Show.major_logo()
        Show.console.print(table_major)

        Show.minor_logo()
        Show.console.print(table_minor)

    @staticmethod
    def tips_document():
        table = Table(
            title=f"[bold #FF851B]{const.ITEM} {const.DESC} Select Command Line",
            header_style="bold #D7FF00",
            title_justify="center",
            show_header=True,
            show_lines=True
        )
        table.add_column("选项", justify="center", width=12)
        table.add_column("参数", justify="center", width=12)
        table.add_column("说明", justify="center", width=44)

        information = [
            ["[bold #FFAFAF]header", "[bold #AFD7FF]标题名", "[bold #DADADA]生成新标题文件夹"],
            ["[bold #FFAFAF]device", "", "[bold #DADADA]选择已连接的设备"],
            ["[bold #FFAFAF]deploy", "", "[bold #DADADA]部署视频分析配置"],
            ["[bold #FFAFAF]create", "", "[bold #DADADA]生成汇总报告"],
            ["[bold #FFAFAF]invent", "", "[bold #DADADA]生成汇总报告"],
            ["[bold #FFAFAF]cancel", "", "[bold #DADADA]退出"],
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
            logger.info(f"[bold #C1FFC1]Engine Initializing[/] ...")
            for i in range(step):
                Show.console.print(function(i), justify="left")
                time.sleep(secs)
            return logger.info(f"[bold #C1FFC1]Engine Loaded[/] ...")

        stochastic = [
            lambda: animation(4, 0.2, speed_engine),
            lambda: animation(8, 0.1, basic_engine),
            lambda: animation(4, 0.2, keras_engine),
            lambda: animation(4, 0.2, other_engine),
        ]
        random.choice(stochastic)()

    @staticmethod
    def content_pose(rlt, avg, dur, org, pnt, video_temp, frate):
        start, close, limit = pnt
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

        table_info.add_row(
            f"[bold #87CEEB]{org}",
            f"[bold #87CEEB]{rlt}",
            f"[bold #87CEEB]{avg}",
            f"[bold #87CEEB]{frate}"
        )

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

        table_clip.add_row(
            f"[bold #87CEEB]{dur}",
            f"[bold][bold #FFA500]start[/]=[[bold #EE82EE]{start}[/]][/]",
            f"[bold][bold #FFA500]close[/]=[[bold #EE82EE]{close}[/]][/]",
            f"[bold][bold #FFA500]limit[/]=[[bold #EE82EE]{limit}[/]][/]"
        )

        Show.console.print(table_info)
        Show.console.print(table_clip)


if __name__ == '__main__':
    pass
