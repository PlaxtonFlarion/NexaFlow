import time
from rich.table import Table
from rich.console import Console
from rich.progress import Progress


class Show(object):

    console = Console()

    @staticmethod
    def retry_fail_logo():
        logo = """[bold]
        ╔════════════════════════════════╗
        ║          Retry Failed          ║
        ╚════════════════════════════════╝

        抱歉，尝试次数已达上限，无法完成操作。
        请稍后再试或联系技术支持寻求帮助。

        您的理解与耐心是我们不断进步的动力！
        """
        Show.console.print(logo)

    @staticmethod
    def connect_fail_logo():
        logo = """[bold]
        ╔════════════════════════════════╗
        ║         Connect Failed         ║
        ╚════════════════════════════════╝

        🚫 连接超时 - 程序退出 🚫

        由于长时间无法建立连接，程序现在将自动退出。
        请检查您的设备或联系技术支持。
        感谢您的耐心，期待下次再见！
        """
        Show.console.print(logo)

    @staticmethod
    def simulation_progress(desc: str, advance: int | float, interval: int | float):
        with Progress() as progress:
            task = progress.add_task(f"[bold #FFFFD7]{desc}", total=100)
            while not progress.finished:
                progress.update(task, advance=advance)
                time.sleep(interval)

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
            title="[bold #FF851B]NexaFlow Framix Main Command Line",
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

        major_information = [
            ["[bold #FFDC00]--flick", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #D7FF00]支持", "[bold #39CCCC]循环模式"],
            ["[bold #FFDC00]--paint", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #D7FF00]支持", "[bold #39CCCC]绘制分割线条"],
            ["[bold #FFDC00]--video", "[bold #7FDBFF]视频文件", "[bold #FFAFAF]多次", "[bold #D7FF00]支持", "[bold #39CCCC]分析视频文件"],
            ["[bold #FFDC00]--stack", "[bold #7FDBFF]视频集合", "[bold #FFAFAF]多次", "[bold #D7FF00]支持", "[bold #39CCCC]分析视频文件集合"],
            ["[bold #FFDC00]--union", "[bold #7FDBFF]报告集合", "[bold #FFAFAF]多次", "[bold #D7FF00]支持", "[bold #39CCCC]聚合视频帧报告"],
            ["[bold #FFDC00]--merge", "[bold #7FDBFF]报告集合", "[bold #FFAFAF]多次", "[bold #D7FF00]支持", "[bold #39CCCC]聚合时间戳报告"],
            ["[bold #FFDC00]--train", "[bold #7FDBFF]视频文件", "[bold #FFAFAF]多次", "[bold #D7FF00]支持", "[bold #39CCCC]归类图片文件"],
            ["[bold #FFDC00]--build", "[bold #7FDBFF]图片集合", "[bold #FFAFAF]多次", "[bold #D7FF00]支持", "[bold #39CCCC]训练模型文件"]
        ]

        for major in major_information:
            table_major.add_row(*major)

        table_minor = Table(
            title="[bold #FF851B]NexaFlow Framix Extra Command Line",
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

        minor_information = [
            ["[bold #FFDC00]--carry", "[bold #7FDBFF]名称", "[bold #FFAFAF]多次", "[bold #AFAFD7]关闭", "[bold #39CCCC]运行指定脚本"],
            ["[bold #FFDC00]--fully", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]运行全部脚本"],
            ["[bold #FFDC00]--quick", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]快速模式"],
            ["[bold #FFDC00]--basic", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]基础模式"],
            ["[bold #FFDC00]--keras", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]智能模式"],
            ["[bold #FFDC00]--alone", "[bold #7FDBFF]布尔", "[bold #8A8A8A]一次", "[bold #AFAFD7]关闭", "[bold #39CCCC]独立控制"],
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

        for minor in minor_information:
            table_minor.add_row(*minor)

        Show.major_logo()
        Show.console.print(table_major)
        Show.simulation_progress(f"Framix Terminal Command.", 1, 0.05)

        Show.minor_logo()
        Show.console.print(table_minor)
        Show.simulation_progress(f"Framix Terminal Command.", 1, 0.05)

    @staticmethod
    def tips_document():
        table = Table(
            title="[bold #FF851B]NexaFlow Framix Select Command Line",
            header_style="bold #D7FF00",
            title_justify="center",
            show_header=True,
            show_lines=True
        )
        table.add_column("选项", justify="center", width=12)
        table.add_column("参数", justify="center", width=12)
        table.add_column("说明", justify="center", width=44)
        table.add_row("[bold #FFAFAF]header", "[bold #AFD7FF]标题名", "[bold #DADADA]生成一个新标题文件夹")
        table.add_row("[bold #FFAFAF]serial", "[bold #8A8A8A]无参数", "[bold #DADADA]重新选择已连接的设备")
        table.add_row("[bold #FFAFAF]deploy", "[bold #8A8A8A]无参数", "[bold #DADADA]重新部署视频分析配置")
        table.add_row("[bold #FFAFAF]create", "[bold #8A8A8A]无参数", "[bold #DADADA]生成视频分析汇总报告")
        table.add_row("[bold #FFAFAF]invent", "[bold #8A8A8A]无参数", "[bold #DADADA]生成视频拆帧汇总报告")
        table.add_row("[bold #FFAFAF]******", "[bold #8A8A8A]无参数", "[bold #DADADA]任意数字代表录制时长")
        Show.console.print(table)
        Show.simulation_progress(f"Framix Terminal Command.", 1, 0.05)


if __name__ == '__main__':
    pass
