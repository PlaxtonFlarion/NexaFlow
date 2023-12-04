from rich.table import Table
from rich.console import Console
from rich.prompt import Prompt
import time
from rich.progress import Progress

console = Console()


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
        "[bold #FFDC00]--paint", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "", "[bold #39CCCC]绘制分割线条"
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
        "[bold #FFDC00]--datum", "[bold #7FDBFF]视频文件", "[bold #FFAFAF]多次", "", "[bold #39CCCC]归类图片文件"
    )
    table_major.add_row(
        "[bold #FFDC00]--train", "[bold #7FDBFF]图片集合", "[bold #FFAFAF]多次", "", "[bold #39CCCC]训练模型文件"
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
        "[bold #FFDC00]--omits", "[bold #7FDBFF]坐标", "[bold #FFAFAF]多次", "", "[bold #39CCCC]忽略区域"
    )

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
    console.print(table_minor)


def help_option():
    table = Table(show_header=True, header_style="bold #D7FF00", show_lines=True)
    table.add_column("选项", justify="center", width=12)
    table.add_column("参数", justify="center", width=12)
    table.add_column("说明", justify="center", width=44)
    table.add_row("[bold #FFAFAF]header", "[bold #AFD7FF]标题名", "[bold #DADADA]生成一个新标题文件夹")
    table.add_row("[bold #FFAFAF]serial", "", "[bold #DADADA]重新选择已连接的设备")
    table.add_row("[bold #FFAFAF]******", "", "[bold #DADADA]任意数字代表录制时长")
    console.print(table)


action = Prompt.ask("[bold #00AF87]<<<按 Enter 开始>>>", console=console, default=5)
console.print(action)
# a = "<Phone brand=Pixel version=OS14 serial=asdfghjkl>"
# console.print(f"[bold #00FFAF]Connect:[/bold #00FFAF] {a}", highlight=True)
# print("\033[0;32m*-* 按 Enter 开始 *-*\033[0m  ")
# help_document()
# help_option()
# with Progress() as progress:
#     task = progress.add_task("[bold #FFFFD7]Framix Terminal Command.", total=100)
#     while not progress.finished:
#         progress.update(task, advance=1)
#         time.sleep(0.1)





