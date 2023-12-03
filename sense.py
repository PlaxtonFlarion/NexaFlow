import os

from rich.table import Table
from rich.console import Console
import time
from rich.progress import Progress

console = Console()


def help_document():
    table_major = Table(
        title="[bold #FF851B]NexaFlow Framix Main Command Line[/bold #FF851B]",
        header_style="bold #FF851B", title_justify="center",
        show_header=True, show_lines=True
    )
    table_major.add_column("主要命令", justify="center", width=12)
    table_major.add_column("参数类型", justify="center", width=12)
    table_major.add_column("传递次数", justify="center", width=8)
    table_major.add_column("附加命令", justify="center", width=8)
    table_major.add_column("功能说明", justify="center", width=22)

    table_major.add_row(
        "[bold #FFDC00]--flick[/bold #FFDC00]  [bold]-f[/bold]", "[bold #7FDBFF]布尔[/bold #7FDBFF]", "[bold #8A8A8A]一次[/bold #8A8A8A]", "[bold #D7FF00]支持[/bold #D7FF00]", "[bold #39CCCC]录制分析视频帧[/bold #39CCCC]"
    )
    table_major.add_row(
        "[bold #FFDC00]--alone[/bold #FFDC00]  [bold]-a[/bold]", "[bold #7FDBFF]布尔[/bold #7FDBFF]", "[bold #8A8A8A]一次[/bold #8A8A8A]", "", "[bold #39CCCC]录制视频[/bold #39CCCC]"
    )
    table_major.add_row(
        "[bold #FFDC00]--paint[/bold #FFDC00]  [bold]-p[/bold]", "[bold #7FDBFF]布尔[/bold #7FDBFF]", "[bold #8A8A8A]一次[/bold #8A8A8A]", "", "[bold #39CCCC]绘制分割线条[/bold #39CCCC]"
    )
    table_major.add_row(
        "[bold #FFDC00]--input[/bold #FFDC00]  [bold]-i[/bold]", "[bold #7FDBFF]视频文件[/bold #7FDBFF]", "[bold #FFAFAF]多次[/bold #FFAFAF]", "[bold #D7FF00]支持[/bold #D7FF00]", "[bold #39CCCC]分析单个视频[/bold #39CCCC]"
    )
    table_major.add_row(
        "[bold #FFDC00]--whole[/bold #FFDC00]  [bold]-w[/bold]", "[bold #7FDBFF]视频集合[/bold #7FDBFF]", "[bold #FFAFAF]多次[/bold #FFAFAF]", "[bold #D7FF00]支持[/bold #D7FF00]", "[bold #39CCCC]分析全部视频[/bold #39CCCC]"
    )
    table_major.add_row(
        "[bold #FFDC00]--merge[/bold #FFDC00]  [bold]-m[/bold]", "[bold #7FDBFF]报告集合[/bold #7FDBFF]", "[bold #FFAFAF]多次[/bold #FFAFAF]", "", "[bold #39CCCC]聚合报告[/bold #39CCCC]"
    )
    table_major.add_row(
        "[bold #FFDC00]--datum[/bold #FFDC00]  [bold]-d[/bold]", "[bold #7FDBFF]视频文件[/bold #7FDBFF]", "[bold #FFAFAF]多次[/bold #FFAFAF]", "", "[bold #39CCCC]归类图片文件[/bold #39CCCC]"
    )
    table_major.add_row(
        "[bold #FFDC00]--train[/bold #FFDC00]  [bold]-t[/bold]", "[bold #7FDBFF]图片文件[/bold #7FDBFF]", "[bold #FFAFAF]多次[/bold #FFAFAF]", "", "[bold #39CCCC]训练模型文件[/bold #39CCCC]"
    )

    table_minor = Table(
        title="[bold #FF851B]NexaFlow Framix Extra Command Line[/bold #FF851B]",
        header_style="bold #FF851B", title_justify="center",
        show_header=True, show_lines=True
    )
    table_minor.add_column("附加命令", justify="center", width=12)
    table_minor.add_column("参数类型", justify="center", width=12)
    table_minor.add_column("传递次数", justify="center", width=8)
    table_minor.add_column("默认状态", justify="center", width=8)
    table_minor.add_column("功能说明", justify="center", width=22)

    table_minor.add_row(
        "[bold #FFDC00]--boost[/bold #FFDC00]  [bold]-b[/bold]", "[bold #7FDBFF]布尔[/bold #7FDBFF]", "[bold #8A8A8A]一次[/bold #8A8A8A]", "[bold #AFAFD7]关闭[/bold #AFAFD7]", "[bold #39CCCC]快速模式[/bold #39CCCC]"
    )
    table_minor.add_row(
        "[bold #FFDC00]--color[/bold #FFDC00]  [bold]-c[/bold]", "[bold #7FDBFF]布尔[/bold #7FDBFF]", "[bold #8A8A8A]一次[/bold #8A8A8A]", "[bold #AFAFD7]关闭[/bold #AFAFD7]", "[bold #39CCCC]彩色模式[/bold #39CCCC]"
    )
    table_minor.add_row(
        "[bold #FFDC00]--omits[/bold #FFDC00]  [bold]-o[/bold]", "[bold #7FDBFF]坐标[/bold #7FDBFF]", "[bold #FFAFAF]多次[/bold #FFAFAF]", "", "[bold #39CCCC]忽略区域[/bold #39CCCC]"
    )

    nexaflow_logo = """[bold #D0D0D0]
    ███╗   ██╗███████╗██╗  ██╗ █████╗   ███████╗██╗      ██████╗ ██╗    ██╗
    ██╔██╗ ██║██╔════╝╚██╗██╔╝██╔══██╗  ██╔════╝██║     ██╔═══██╗██║    ██║
    ██║╚██╗██║█████╗   ╚███╔╝ ███████║  █████╗  ██║     ██║   ██║██║ █╗ ██║
    ██║ ╚████║██╔══╝   ██╔██╗ ██╔══██║  ██╔══╝  ██║     ██║   ██║██║███╗██║
    ██║  ╚███║███████╗██╔╝ ██╗██║  ██║  ██║     ███████╗╚██████╔╝╚███╔███╔╝
    ╚═╝   ╚══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝
    [/bold #D0D0D0]"""
    console.print(nexaflow_logo)
    console.print(table_major)
    console.print(table_minor)


a = "<Phone brand=Pixel version=OS14 serial=asdfghjkl>"
console.print(f"[bold #00FFAF]Connect:[/bold #00FFAF] {a}", highlight=True)
print("\033[0;32m*-* 按 Enter 开始 *-*\033[0m  ")
help_document()
with Progress() as progress:
    task = progress.add_task("[bold #FFFFD7]Framix Terminal Command.", total=100)
    while not progress.finished:
        progress.update(task, advance=1)
        time.sleep(0.1)





