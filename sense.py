from rich.table import Table
from rich.console import Console

console = Console()

table_major = Table(
    title="[bold #FF851B]Framix Command Line[/bold #FF851B]",
    header_style="bold #FF851B", title_justify="center",
    show_header=True, show_lines=True
)
table_major.add_column("主要选项", justify="center", width=22)
table_major.add_column("参数", justify="center", width=22)
table_major.add_column("说明", justify="center", width=22)

table_major.add_row(
    "[bold #FFDC00]--flick[/bold #FFDC00] [bold]-f[/bold]", "[bold #7FDBFF]布尔值[/bold #7FDBFF]", "[bold #39CCCC]录制分析视频帧[/bold #39CCCC]"
)
table_major.add_row(
    "[bold #FFDC00]--alone[/bold #FFDC00] [bold]-a[/bold]", "[bold #7FDBFF]布尔值[/bold #7FDBFF]", "[bold #39CCCC]单独录制视频[/bold #39CCCC]"
)
table_major.add_row(
    "[bold #FFDC00]--paint[/bold #FFDC00] [bold]-p[/bold]", "[bold #7FDBFF]布尔值[/bold #7FDBFF]", "[bold #39CCCC]绘制分割线条[/bold #39CCCC]"
)
table_major.add_row(
    "[bold #FFDC00]--input[/bold #FFDC00] [bold]-i[/bold]", "[bold #7FDBFF]视频文件[/bold #7FDBFF]", "[bold #39CCCC]单独分析视频[/bold #39CCCC]"
)
table_major.add_row(
    "[bold #FFDC00]--whole[/bold #FFDC00] [bold]-w[/bold]", "[bold #7FDBFF]测试集合文件夹[/bold #7FDBFF]", "[bold #39CCCC]分析全部视频帧[/bold #39CCCC]"
)
table_major.add_row(
    "[bold #FFDC00]--merge[/bold #FFDC00] [bold]-m[/bold]", "[bold #7FDBFF]测试集合文件夹[/bold #7FDBFF]", "[bold #39CCCC]聚合测试报告[/bold #39CCCC]"
)
table_major.add_row(
    "[bold #FFDC00]--datum[/bold #FFDC00] [bold]-d[/bold]", "[bold #7FDBFF]视频文件[/bold #7FDBFF]", "[bold #39CCCC]归类模型[/bold #39CCCC]"
)
table_major.add_row(
    "[bold #FFDC00]--train[/bold #FFDC00] [bold]-t[/bold]", "[bold #7FDBFF]归类文件夹[/bold #7FDBFF]", "[bold #39CCCC]训练模型[/bold #39CCCC]"
)

table_minor = Table(
    title="[bold #FF851B]Extra Command Line[/bold #FF851B]",
    header_style="bold #FF851B", title_justify="center",
    show_header=True, show_lines=True
)
table_minor.add_column("额外选项", justify="center", width=22)
table_minor.add_column("参数", justify="center", width=22)
table_minor.add_column("说明", justify="center", width=22)

table_minor.add_row(
    "[bold #FFDC00]--boost[/bold #FFDC00] [bold]-b[/bold]", "[bold #7FDBFF]布尔值[/bold #7FDBFF]", "[bold #39CCCC]快速分析视频帧[/bold #39CCCC]"
)
table_minor.add_row(
    "[bold #FFDC00]--color[/bold #FFDC00] [bold]-c[/bold]", "[bold #7FDBFF]布尔值[/bold #7FDBFF]", "[bold #39CCCC]加载彩色帧[/bold #39CCCC]"
)
table_minor.add_row(
    "[bold #FFDC00]--omits[/bold #FFDC00] [bold]-o[/bold]", "[bold #7FDBFF]横纵坐标和大小[/bold #7FDBFF]", "[bold #39CCCC]忽略坐标位置[/bold #39CCCC]"
)

nexaflow_logo = """
███╗   ██╗███████╗██╗  ██╗ █████╗ ███████╗██╗      ██████╗ ██╗    ██╗
██╔██╗ ██║██╔════╝╚██╗██╔╝██╔══██╗██╔════╝██║     ██╔═══██╗██║    ██║
██║╚██╗██║█████╗   ╚███╔╝ ███████║█████╗  ██║     ██║   ██║██║ █╗ ██║
██║ ╚████║██╔══╝   ██╔██╗ ██╔══██║██╔══╝  ██║     ██║   ██║██║███╗██║
██║  ╚███║███████╗██╔╝ ██╗██║  ██║██║     ███████╗╚██████╔╝╚███╔███╔╝
╚═╝   ╚══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝
"""
console.print(nexaflow_logo)
console.print(table_major)
console.print(table_minor)


