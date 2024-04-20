import time
import typing
from rich.text import Text
from rich.table import Table
from rich.console import Console
from rich.progress import Progress
from nexaflow import const


class Show(object):

    console = Console()

    @staticmethod
    def show(msg: typing.Any):
        Show.console.print(
            f"[bold][bold #D7AF87]{const.DESC} | Analyzer |[/bold #D7AF87] {msg}[/bold]"
        )

    @staticmethod
    def simulation_progress(desc: str, advance: int | float, interval: int | float):
        with Progress() as progress:
            task = progress.add_task(f"[bold #FFFFD7]{desc}", total=100)
            while not progress.finished:
                progress.update(task, advance=advance)
                time.sleep(interval)

    @staticmethod
    def normal_exit():
        return f"""
    \033[1m╔════════════════════════════════╗
    ║       \033[1m\033[32mMissions  Complete\033[0m       ║
    ╚════════════════════════════════╝

    ✦✦✦ {const.DESC} will now automatically exit ✦✦✦
    ✧✧✧ See you next ✧✧✧\033[0m
    """

    @staticmethod
    def abnormal_exit():
        return f"""
    \033[1m╔════════════════════════════════╗
    ║        \033[1m\033[31mMissions  Failed\033[0m        ║
    ╚════════════════════════════════╝

    ✦✦✦ {const.DESC} will now automatically exit ✦✦✦
    ✧✧✧ See you next ✧✧✧\033[0m
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
            ["[bold #FFDC00]--quick", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]快速模式"],
            ["[bold #FFDC00]--basic", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]基础模式"],
            ["[bold #FFDC00]--keras", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]智能模式"],
            ["[bold #FFDC00]--alone", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]独立控制"],
            ["[bold #FFDC00]--whist", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]静默录制"],
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
        Show.simulation_progress(f"{const.DESC} Terminal Command.", 1, 0.05)

        Show.minor_logo()
        Show.console.print(table_minor)
        Show.simulation_progress(f"{const.DESC} Terminal Command.", 1, 0.05)

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
            ["[bold #FFAFAF]header", "[bold #AFD7FF]标题名", "[bold #DADADA]生成一个新标题文件夹"],
            ["[bold #FFAFAF]serial", "[bold #8A8A8A]无参数", "[bold #DADADA]重新选择已连接的设备"],
            ["[bold #FFAFAF]deploy", "[bold #8A8A8A]无参数", "[bold #DADADA]重新部署视频分析配置"],
            ["[bold #FFAFAF]create", "[bold #8A8A8A]无参数", "[bold #DADADA]生成视频分析汇总报告"],
            ["[bold #FFAFAF]invent", "[bold #8A8A8A]无参数", "[bold #DADADA]生成视频拆帧汇总报告"]
        ]
        for info in information:
            table.add_row(*info)

        Show.console.print(table)
        Show.simulation_progress(f"{const.DESC} Terminal Command.", 1, 0.05)

    @staticmethod
    def load_animation(style):

        c = {
            1: "bold #D7AFAF", 2: "bold #5FD75F", 3: "bold #5FD7FF", 4: "bold #D7AF5F",
        }

        def quick_engine(stage):
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

        def photo_engine(stage):
            engine_stages = [
                Text("\n○   ○", style=c[1]),
                Text("○──┐○──┐", style=c[2]),
                Text("○──┤○──┤", style=c[3]),
                Text("○──┤○──┤◉\n", style=c[4])
            ]
            return engine_stages[stage % len(engine_stages)]

        def animation(step, function):
            Show.console.print(start_view, style="bold 	#FFD700")
            for i in range(step):
                Show.console.print(function(i), justify="left")
                time.sleep(0.5)
            return Show.console.print(close_view, style="bold #FFD700")

        start_view = f"[bold][bold #D7AF87]{const.DESC} |[/bold #D7AF87] Analyzer | Engine Initializing ..."
        close_view = f"[bold][bold #D7AF87]{const.DESC} |[/bold #D7AF87] Analyzer | Engine Loaded ...\n"

        if style.quick:
            animation(4, quick_engine)
        elif style.basic:
            animation(8, basic_engine)
        elif style.keras:
            animation(4, keras_engine)
        else:
            animation(4, photo_engine)


if __name__ == '__main__':
    pass
