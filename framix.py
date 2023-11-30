from rich.table import Table
from rich.console import Console

console = Console()


def help_document():
    table = Table(
        title="[bold #FF851B]Framix Command Line[/bold #FF851B]",
        header_style="bold #FF851B", title_justify="center",
        show_header=True, show_lines=True
    )

    table.add_column("选项", justify="center", width=22)
    table.add_column("参数", justify="center", width=22)
    table.add_column("说明", justify="center", width=22)

    table.add_row(
        "[bold #FFDC00]--flick[/bold #FFDC00]   [bold]-f[/bold]", "[bold #7FDBFF]布尔值[/bold #7FDBFF]",
        "[bold #39CCCC]录制分析视频帧[/bold #39CCCC]"
    )
    table.add_row(
        "[bold #FFDC00]--alone[/bold #FFDC00]   [bold]-a[/bold]", "[bold #7FDBFF]布尔值[/bold #7FDBFF]",
        "[bold #39CCCC]单独录制视频[/bold #39CCCC]"
    )
    table.add_row(
        "[bold #FFDC00]--paint[/bold #FFDC00]   [bold]-p[/bold]", "[bold #7FDBFF]布尔值[/bold #7FDBFF]",
        "[bold #39CCCC]绘制分割线条[/bold #39CCCC]"
    )
    table.add_row(
        "[bold #FFDC00]--boost[/bold #FFDC00]   [bold]-b[/bold]", "[bold #7FDBFF]布尔值[/bold #7FDBFF]",
        "[bold #39CCCC]快速分析视频帧[/bold #39CCCC]"
    )
    table.add_row(
        "[bold #FFDC00]--color[/bold #FFDC00]   [bold]-c[/bold]", "[bold #7FDBFF]布尔值[/bold #7FDBFF]",
        "[bold #39CCCC]加载彩色帧[/bold #39CCCC]"
    )
    table.add_row(
        "[bold #FFDC00]--input[/bold #FFDC00]   [bold]-i[/bold]", "[bold #7FDBFF]视频文件[/bold #7FDBFF]",
        "[bold #39CCCC]单独分析视频[/bold #39CCCC]"
    )
    table.add_row(
        "[bold #FFDC00]--whole[/bold #FFDC00]   [bold]-w[/bold]", "[bold #7FDBFF]测试集合文件夹[/bold #7FDBFF]",
        "[bold #39CCCC]分析全部视频帧[/bold #39CCCC]"
    )
    table.add_row(
        "[bold #FFDC00]--merge[/bold #FFDC00]   [bold]-m[/bold]", "[bold #7FDBFF]测试集合文件夹[/bold #7FDBFF]",
        "[bold #39CCCC]聚合测试报告[/bold #39CCCC]"
    )
    table.add_row(
        "[bold #FFDC00]--omits[/bold #FFDC00]   [bold]-o[/bold]", "[bold #7FDBFF]x,y,x_size,y_size[/bold #7FDBFF]",
        "[bold #39CCCC]忽略坐标位置[/bold #39CCCC]"
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
    console.print(table)


def help_option():
    table = Table(show_header=True, header_style="bold #D7FF00")
    table.add_column("[bold]选项[/bold]", justify="center", width=22)
    table.add_column("[bold]参数[/bold]", justify="center", width=22)
    table.add_column("[bold]说明[/bold]", justify="center", width=22)
    table.add_row("[bold #FFAFAF]header[/bold #FFAFAF]", "[bold #AFD7FF]标题名[/bold #AFD7FF]", "[bold #DADADA]生成一个新标题文件夹[/bold #DADADA]")
    table.add_row("[bold #FFAFAF]serial[/bold #FFAFAF]", "[bold #AFD7FF]无参数[/bold #AFD7FF]", "[bold #DADADA]重新选择已连接的设备[/bold #DADADA]")
    table.add_row("[bold #FFAFAF]******[/bold #FFAFAF]", "[bold #AFD7FF]无参数[/bold #AFD7FF]", "[bold #DADADA]任意数字代表录制时长[/bold #DADADA]")
    console.print(table)


help_document()
help_option()
