#   ____            _
#  |  _ \  ___  ___(_) __ _ _ __
#  | | | |/ _ \/ __| |/ _` | '_ \
#  | |_| |  __/\__ \ | (_| | | | |
#  |____/ \___||___/_|\__, |_| |_|
#                     |___/
#

"""
版权所有 (c) 2024  Framix(画帧秀)
此文件受 Framix(画帧秀) 许可证的保护。您可以在 LICENSE.md 文件中查看详细的许可条款。

Copyright (c) 2024  Framix(画帧秀)
This file is licensed under the Framix(画帧秀) License. See the LICENSE.md file for more details.

# Copyright (c) 2024  Framix(画帧秀)
# このファイルは Framix(画帧秀) ライセンスの下でライセンスされています。詳細は LICENSE.md ファイルを参照してください。
"""

import os
import time
import random
import typing
import asyncio
from pathlib import Path
from rich.live import Live
from rich.tree import Tree
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn,
    SpinnerColumn, TimeRemainingColumn
)
from nexacore.argument import Args
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

    def __init__(self, design_level: str = "INFO"):
        self.design_level = design_level.upper()

    @staticmethod
    def notes(text: typing.Any) -> None:
        """
        输出常规日志信息，使用 bold 样式强调。
        """
        Design.console.print(f"[bold]{const.DESC} | Analyzer | {text}[/]")

    @staticmethod
    def annal(text: typing.Any) -> None:
        """
        输出结构化强调文本，适用于模型状态或分析摘要。
        """
        Design.console.print(f"[bold]{const.DESC} | Analyzer |[/]", Text(text, "bold"))

    @staticmethod
    def show_tree(file_path: str, *args: str) -> None:
        """
        构建并展示文件夹结构树，根据文件后缀显示图标，可自定义展示哪些类型的文件。
        """
        choice_icon: typing.Any = lambda x: {
                '.mp4': '🎞️',
                '.avi': '🎞️',
                '.mov': '🎞️',
                '.mkv': '🎞️',
                '.html': '🌐',
                '.db': '🗄️',
                '.log': '📜',
                '.py': '🐍',
                '.json': '🧾',
                '.txt': '📄',
                '.png': '🖼️',
                '.jpg': '🖼️',
                '.zip': '🗜️',
                '.exe': '⚙️',
            }.get(os.path.splitext(x)[1].lower(), '📄')

        def add_nodes(current_node: "Tree", current_path: str) -> None:
            try:
                with os.scandir(current_path) as scamper:
                    for cur in scamper:
                        folder_path = cur.path
                        if cur.is_dir():
                            sub_node = current_node.add(
                                f"[link file:///{folder_path}]📁 {cur.name}[/]", guide_style="bold #7CFC00"
                            )
                            add_nodes(sub_node, folder_path)
                        elif cur.is_file() and cur.name.endswith(('.mp4', '.avi', '.mov', '.mkv', *args)):
                            current_node.add(
                                f"[link file:///{folder_path}]{choice_icon(cur.name)} {cur.name}[/] <<<",
                                style="bold #FF69B4"
                            )
            except PermissionError:
                current_node.add("Access denied", style="bold #FF6347")

        tree = Tree(
            f"[link file:///{file_path}]📁 {os.path.basename(file_path)}[/]", guide_style="bold #00CED1"
        )
        add_nodes(tree, file_path)
        Design.console.print(tree)

    @staticmethod
    def show_progress() -> "Progress":
        """
        创建并返回自定义进度条组件，适用于异步任务的状态展示。
        """
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
        """
        启动模拟进度条，用于快速任务的视觉反馈。
        """
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
        """
        显示任务完成状态的 ASCII 区块框提示，柔和浅绿，视觉愉悦。
        """
        Design.console.print(f"""[bold]
    ╔══════════════════════════════════╗
    ║          [bold #A8F5B5]Missions  Done[/]          ║
    ╚══════════════════════════════════╝""")

    @staticmethod
    def fail() -> None:
        """
        显示任务失败状态的 ASCII 区块框提示，柔和奶黄，温和不过亮。
        """
        Design.console.print(f"""[bold]
    ╔══════════════════════════════════╗
    ║          [bold #F5A8A8]Missions  Fail[/]          ║
    ╚══════════════════════════════════╝""")

    @staticmethod
    def exit() -> None:
        """
        显示任务退出状态的 ASCII 区块框提示，柔和淡红，不刺眼。
        """
        Design.console.print(f"""[bold]
    ╔══════════════════════════════════╗
    ║          [bold #FFF6AA]Missions  Exit[/]          ║
    ╚══════════════════════════════════╝""")

    @staticmethod
    def closure() -> str:
        """
        返回格式化的退出提示文本。
        """
        return f"""
    <*=> {const.DESC} will now automatically exit <=*>
    <*=> {const.DESC} see you next <=*>
        """

    @staticmethod
    def specially_logo() -> None:
        """
        打印程序启动时的 ASCII 风格 Logo，并以随机柔和色渲染输出。
        """
        logo = f"""\
__________                        _____        
___  ____/____________ _______ ______(_)___  __
__  /_   __  ___/  __ `/_  __ `__ \\_  /__  |/_/
_  __/   _  /   / /_/ /_  / / / / /  / __>  <  
/_/      /_/    \\__,_/ /_/ /_/ /_//_/  /_/|_|                                                 
        """
        soft_bright_colors = random.choice(
            [
                "#FFE4B5",  # 浅橙 Moccasin
                "#E0FFFF",  # 浅青 LightCyan
                "#FFFACD",  # 柠檬绸 LemonChiffon
                "#E6E6FA",  # 薰衣草 Lavender
                "#F0FFF0",  # 蜜瓜白 Honeydew
                "#F5F5DC",  # 米色 Beige
                "#F0F8FF",  # 爱丽丝蓝 AliceBlue
                "#D8BFD8",  # 蓟色 Thistle
                "#FFF0F5",  # 藕色 LavenderBlush
                "#F5FFFA",  # 薄荷奶油 MintCream
                "#FFEFD5",  # 木瓜奶油 PapayaWhip
                "#F8F8FF",  # 幽灵白 GhostWhite
            ]
        )
        for line in logo.split("\n"):
            Design.console.print(
                Text.from_markup(f"[bold {soft_bright_colors}]{line}[/]")
            )
            time.sleep(0.05)
        Design.console.print(const.DECLARE)

    @staticmethod
    def major_logo() -> None:
        """
        打印主 Logo（带 ASCII 图形和配色），适用于程序启动欢迎界面。
        """
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
        """
        打印次 Logo，逐行动态加载并附带版权信息。
        """
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
        """
        展示命令行参数文档（来自 ARGUMENT 配置），以表格形式高亮各类参数分组。
        """
        table_style = {
            "title_justify": "center", "show_header": True, "show_lines": True
        }

        for keys, values in Args.ARGUMENT.items():
            know = "[bold #FFE4E1]参数互斥" if keys in Args.discriminate else "[bold #C1FFC1]参数兼容"

            table = Table(
                title=f"[bold #FFDAB9]{const.DESC} | {const.ALIAS} CLI [bold #66CDAA]<{keys}>[/] <{know}>",
                header_style="bold #FF851B", **table_style
            )
            table.add_column("命令", justify="left", no_wrap=True, width=7)
            table.add_column("传递", justify="left", no_wrap=True, width=4)
            table.add_column("功能说明", justify="left", no_wrap=True, width=16)
            table.add_column("使用方法", justify="left", no_wrap=True, width=39)

            information = [
                [key, *value["view"], value["help"]] for key, value in values.items()
            ]
            for info in information:
                cmds, push, kind, desc = info
                push_color = "#FFAFAF" if push == "多次" else "#CFCFCF"
                table.add_row(
                    f"[bold #FFDC00]{cmds}", f"[bold {push_color}]{push}", f"[bold #39CCCC]{desc}",
                    f"[bold #D7AFD7]{const.NAME} [bold #FFDC00]{cmds}[bold #7FDBFF]{kind}"
                )
            Design.console.print(table, "\t")

    @staticmethod
    def tips_document() -> None:
        """
        显示简化参数提示文档，适用于交互式命令输入提示。
        """
        table_style = {
            "title_justify": "center", "show_header": True, "show_lines": True
        }

        table = Table(
            title=f"[bold #FFDAB9]{const.DESC} | {const.ALIAS} CLI",
            header_style="bold #FF851B", **table_style
        )
        table.add_column("选项", justify="left", width=12)
        table.add_column("说明", justify="left", width=12)
        table.add_column("用法", justify="left", width=16)

        information = [
            ["[bold #FFAFAF]header", "[bold #FFD39B]生成标题", "[bold #AFD7FF]header new_title"],
            ["[bold #FFAFAF]device", "[bold #FFD39B]连接设备", "[bold #AFD7FF]device"],
            ["[bold #FFAFAF]deploy", "[bold #FFD39B]部署配置", "[bold #AFD7FF]deploy"],
            ["[bold #FFAFAF]digest", "[bold #FFD39B]分析模式", "[bold #AFD7FF]digest"],
            ["[bold #FFAFAF]create", "[bold #FFD39B]生成报告", "[bold #AFD7FF]create"],
            ["[bold #FFAFAF]cancel", "[bold #FFD39B]退出程序", "[bold #AFD7FF]cancel"]
        ]
        for info in information:
            table.add_row(*info)
        Design.console.print("\n", table, "\n")

    @staticmethod
    def engine_topology_wave() -> None:
        """
        启动时加载动画。
        """
        neon_flow = [
            "#39FF14", "#00FFFF", "#FF00FF", "#FFFF33"
        ]
        dark_matter = [
            "#8A2BE2", "#FF1493", "#00CED1", "#FF4500"
        ]
        core_pulse = [
            "#FFD700", "#00FFAA", "#FF6EC7", "#87CEFA"
        ]
        ether = [
            "#B0E0E6", "#D8BFD8", "#C1FFC1", "#E0FFFF",
        ]
        magma = [
            "#FF6347", "#FF4500", "#FF8C00", "#FFD700",
        ]
        prism = [
            "#FFB6C1", "#ADD8E6", "#98FB98", "#FFFACD",
        ]
        circuit = [
            "#00FF7F", "#7FFFD4", "#FFA07A", "#CCCCCC",
        ]
        frost = [
            "#AFEEEE", "#E0FFFF", "#B0C4DE", "#FFFFFF",
        ]
        nova = [
            "#FF69B4", "#00BFFF", "#BA55D3", "#FFFF66",
        ]
        cloud_memory = [
            "#D0F0FF", "#FFEFD5", "#FFC0CB", "#E6E6FA"
        ]
        pastel_link = [
            "#C1FFC1", "#FFFACD", "#AFEEEE", "#DDA0DD"
        ]
        serene_flux = [
            "#D8BFD8", "#E0FFFF", "#F5F5DC", "#F0FFF0"
        ]
        obsidian_flux = [
            "#708090", "#A9A9B0", "#3399FF", "#FF7F50"
        ]
        nimbus_bloom = [
            "#F5FFFA", "#FFE4B5", "#E0F8F7", "#FADADD"
        ]
        cyber_blossom = [
            "#CC66FF", "#FF66B2", "#66CCFF", "#FFCCFF"
        ]

        # 随机选一个配色列表，打乱顺序
        random.shuffle(
            colors := random.choice(
                [
                    neon_flow, dark_matter, core_pulse, ether, magma, prism, circuit,
                    frost, nova, cloud_memory, pastel_link, serene_flux, obsidian_flux,
                    nimbus_bloom, cyber_blossom
                ]
            )
        )

        stages = [
            f"""                  
              (●)
               *
               |""",
            f"""         (●)-------(●)
               *        |
               |        |""",
            f"""         (●)-------(●)
               * \\      |
               |  \\     |
              (●)---(●)---(●)
             / | \\   |""",
            f"""        (●)---------(●)   [bold {colors[0]}](● ● ●)[/]                  
             / | \\     \\     |
            (●) (●)-----(●)-----(●)
                 *       *       *  \\
                (●)-----(●)-----(●)---(●)
            """
        ]

        chars = [char for char in const.DESC.upper()]

        replace_star: callable = lambda x, y=iter(chars): [
            "".join(next(y, "|") if c == "*" else c for c in z)
            if isinstance(z, str) else (next(y, "|") if z == "*" else z) for z in x
        ]
        after_replacement = replace_star(stages)

        Design.notes(f"[bold][bold {colors[0]}]{const.DESC} Engine Initializing[/] ...")
        for index, i in enumerate(after_replacement):
            Design.console.print(
                Text.from_markup(i, style=f"bold {colors[index]}")
            )
            time.sleep(0.2)
        Design.notes(f"[bold][bold {colors[0]}]Engine Loaded Successfully[/] ...\n")

    @staticmethod
    async def show_quantum_intro() -> None:
        """
        星域构形动画（Quantum Star Boot）。
        """
        frames = [
            f"""\

    [#808080]        ░░░░░░░░░░
            ░░░░        ░░░░
         ░░░░    [#00FFFF]●[/]    ░░░░
            ░░░░        ░░░░
                ░░░░░░░░░░[/]
            """,
            f"""\

    [#999999]        ████▓▓████
          ▓▓██            ██▓▓
      ██▓▓    [#00FFDD]◉[/]     ▓▓██
          ▓▓██            ██▓▓
              ████████████[/]
            """,
            f"""\

    [#AAAAAA]        ▓▓▓▓▓▓▓▓▓▓
          ████    [#00FFB7]◎[/]    ████
        ██▓▓            ▓▓██
          ████        ████
              ▓▓▓▓▓▓▓▓▓▓[/]
            """,
            f"""\

[bold #00FFC0]╔═════════════════════════════╗
║                             ║
║       [bold #FFD700]{const.DESC} Compiler[/]       ║
║                             ║
╚═════════════════════════════╝
            """
        ]

        with Live(console=Design.console, refresh_per_second=10, transient=True) as live:
            for _ in range(10):  # 播放次数
                for frame in frames[:-1]:
                    live.update(Text.from_markup(frame))
                    await asyncio.sleep(0.2)

        # 渲染最终面板
        final_panel = Panel.fit(
            frames[-1], border_style="bold #7FFFD4"
        )

        Design.console.print(final_panel)
        Design.console.print(f"\n{const.DECLARE}")
        await asyncio.sleep(1)

        Design.simulation_progress("Compiler Ready")

    def show_panel(self, text: typing.Any, wind: dict) -> None:
        """
        根据日志等级和样式参数渲染面板式输出。
        """
        if self.design_level != const.SHOW_LEVEL:
            return None

        panel = Panel(
            Text(
                f"{text}", **wind["文本"]
            ), **wind["边框"], width=int(self.console.width * 0.7)
        )
        self.console.print(panel)

    def show_file(self, file_path: "Path") -> None:
        """
        显示树状图。
        """
        if self.design_level != const.SHOW_LEVEL:
            return None

        color_schemes = {
            "Ocean Breeze": ["#AFD7FF", "#87D7FF", "#5FAFD7"],  # 根 / 中间 / 文件
            "Forest Pulse": ["#A8FFB0", "#87D75F", "#5FAF5F"],
            "Neon Sunset": ["#FFAF87", "#FF875F", "#D75F5F"],
            "Midnight Ice": ["#C6D7FF", "#AFAFD7", "#8787AF"],
            "Cyber Mint": ["#AFFFFF", "#87FFFF", "#5FD7D7"]
        }
        file_icons = {
            "folder": "📁",
            ".json": "📦",
            ".yaml": "🧾",
            ".yml": "🧾",
            ".md": "📝",
            ".log": "📄",
            ".html": "🌐",
            ".sh": "🔧",
            ".bat": "🔧",
            ".db": "🗃️",
            ".sqlite": "🗃️",
            ".zip": "📦",
            ".tar": "📦",
            "default": "📄"
        }
        text_color = random.choice([
            "#8A8A8A", "#949494", "#9E9E9E", "#A8A8A8", "#B2B2B2"
        ])

        root_color, folder_color, file_color = random.choice(list(color_schemes.values()))

        choice_icon: callable = lambda x: file_icons["folder"] if (y := Path(x)).is_dir() else (
            file_icons[n] if (n := y.name.lower()) in file_icons else file_icons["default"]
        )

        parts = file_path.parts

        # 根节点
        root = parts[0]
        tree = Tree(
            f"[bold {text_color}]{choice_icon(root)} {root}[/]", guide_style=f"bold {root_color}"
        )
        current_path = parts[0]
        current_node = tree

        # 处理中间的文件夹
        for part in parts[1:-1]:
            current_path = Path(current_path, part)
            current_node = current_node.add(
                f"[bold {text_color}]{choice_icon(current_path)} {part}[/]", guide_style=f"bold {folder_color}"
            )

        ext = (file := Path(parts[-1])).suffix.lower()
        current_node.add(f"[bold {file_color}]{choice_icon(ext)} {file.name}[/]")

        Design.console.print("\n", tree, "\n")

    async def frame_grid_initializer(self, animation_event: "asyncio.Event") -> None:
        """
        视频拆帧前的初始化动画，用于模拟帧网格构建过程。
        """
        if self.design_level != const.SHOW_LEVEL:
            return None

        self.console.print(
            f"\n[bold #00F5FF]▶ {const.DESC} grid loading ...\n"
        )

        rows, cols = 3, 13
        grid = [[" " for _ in range(cols)] for _ in range(rows)]
        symbols = ["░", "▒", "▓", "□"]

        def render_grid() -> "Text":
            lines = []
            for row in grid:
                colored_line = " ".join(
                    cell if cell.startswith("[") else f"[bold #00FFAA]{cell}"
                    for cell in row
                )
                lines.append(f"[bold #00DDDD][{const.DESC}::Grid][/] {colored_line}")
            return Text.from_markup("\n".join(lines))

        def fill_flow() -> None:
            for row in range(rows):
                for col in range(cols):
                    grid[row][col] = random.choice(symbols)
            live.update(render_grid())

        live = Live(render_grid(), console=self.console, refresh_per_second=20)
        live.start()

        expanded_event = asyncio.Event()

        try:
            # 展开格点
            for r in range(rows):
                for c in range(cols):
                    grid[r][c] = "·"
                    live.update(render_grid())
                    await asyncio.sleep(0.02)
            expanded_event.set()  # 格点展开完毕

            # 激活填充流动
            while not animation_event.is_set():
                fill_flow()
                await asyncio.sleep(0.12)

        except asyncio.CancelledError:
            # 格点未展开完毕，填充流动
            if not expanded_event.is_set():
                fill_flow()

            # 中心节点点亮
            grid[rows // 2][cols // 2] = "[bold #39FF14]▣[/]"
            live.update(render_grid())
            await asyncio.sleep(0.5)

            live.stop()

        finally:
            self.console.print(
                f"\n[bold #7CFC00]>>> {const.DESC} frame grid online. Awaiting extraction. <<<\n"
            )

    async def boot_html_renderer(self, animation_event: "asyncio.Event") -> None:
        """
        HTML 渲染动画，模拟 DOM 构建与样式注入过程，分为结构展开、DOM 成树、样式渗透三阶段。
        """
        if self.design_level != const.SHOW_LEVEL:
            return None

        self.console.print(
            f"\n[bold #00F5FF]▶ {const.DESC} booting html render engine ...\n"
        )

        prefix = f"[bold #1E90FF][{const.DESC}::Render][/]"

        tags = [
            "<html>", "<head>", "<body>", "<div>", "<section>", "<span>", "<meta>", "<title>",
            "<img>", "<button>", "<script>", "<canvas>", "<figure>", "<keygen>", "<template>",
            "<meter>", "<output>", "<progress>", "<ruby>", "<source>", "<summary>", "<audio>"
            "<time>", "<track>", "<video>", "<bdi>", "<code>", "<mark>",
        ]
        lines_dom = ["─┬─", " ├─", " │", " ╰─", "╚══", "╠══", "╩══"]
        styles = ["≡", "#", ":", "{", "}", "▓", "●", "░", "▤", "─"]

        def render(tags_state: str, dom_state: str, css_state: str, phase_label: str) -> "Text":
            out = [
                f"{prefix} [bold #00CED1]{tags_state}[/]",
                f"{prefix} [bold #FFD700]{dom_state}[/]",
                f"{prefix} [bold #FF69B4]{css_state}[/]",
                f"{prefix} [dim]{phase_label}[/]"
            ]
            return Text.from_markup("\n".join(out))

        live = Live(console=self.console, refresh_per_second=24)
        live.start()

        try:
            while not animation_event.is_set():
                num: callable = lambda x, y: random.randint(x, y)
                tag_line = " ".join(random.choice(tags) for _ in range(num(4, 5)))
                dom_line = " ".join(random.choice(lines_dom) for _ in range(num(6, 8)))
                css_line = " ".join(random.choice(styles) for _ in range(num(16, 18)))
                phase = random.choice([
                    "Building DOM tree...", "Injecting layout nodes...", "Applying inline styles...",
                    "Parsing CSS rules...", "Finalizing structure..."
                ])
                live.update(render(tag_line, dom_line, css_line, phase))
                await asyncio.sleep(0.12)

        except asyncio.CancelledError:
            live.stop()

        finally:
            self.console.print(
                f"\n[bold #7CFC00]>>> {const.DESC} HTML layout finalized. Styles applied successfully. <<<\n"
            )

    async def render_horizontal_pulse(self, animation_event: "asyncio.Event") -> None:
        """
        渲染报告时的横向光柱动画，表现为左右流动的亮块。
        """
        if self.design_level != const.SHOW_LEVEL:
            return None

        self.console.print(
            f"\n[bold #00F5FF]▶ {const.DESC} rendering html content ...\n"
        )

        width = int(self.console.width * 0.25)
        charset = "⣿"

        border_style = random.choice([
            "bold #00CED1",  # 青蓝 | 科技感
            "bold #7CFC00",  # 荧光绿 | 活力
            "bold #FF69B4",  # 樱花粉 | 灵动
            "bold #FFA500",  # 暖橙色 | 醒目
            "bold #8A2BE2",  # 紫色光晕 | 魔幻科技
        ])

        color_pair = random.choice([
            "[bold #000000 on #A8FF60]",  # 黑字荧光绿底
            "[bold #FFFFFF on #3F3F46]",  # 白字暗灰底
            "[bold #FFD700 on #000000]",  # 金字黑底
            "[bold #00FFFF on #1E1E1E]",  # 霓虹蓝字 深灰底
            "[bold #FF00FF on #2F004F]",  # 品红字 暗紫底
        ])

        title_color = random.choice([
            "#00F5FF",  # 极光青蓝 · 清亮醒目
            "#FFAFD7",  # 霓虹粉紫 · 柔光梦感
            "#A6E22E",  # 荧光绿 · 聚焦提示
            "#FFD700",  # 金黄 · 荣耀与完成状态
            "#5FD7FF",  # 冰蓝 · 冷静科技感
        ])

        live = Live(console=self.console, refresh_per_second=20)
        live.start()

        try:
            while not animation_event.is_set():
                if (offset := int((time.time() * 10) % (width * 2))) >= width:
                    offset = width * 2 - offset

                frame = charset * offset + color_pair + charset + "[/]" + charset * (
                        width - offset
                )
                panel = Panel.fit(
                    Text.from_markup(frame),
                    title=f"[bold {title_color}]{const.DESC}", border_style=border_style, padding=(0, 2)
                )
                live.update(panel)
                await asyncio.sleep(0.12)

        except asyncio.CancelledError:
            live.stop()

        finally:
            self.console.print(
                "\n[bold #7CFC00]>>> HTML output successfully generated. <<<\n"
            )

    async def frame_stream_flux(self) -> None:
        """
        高级帧流动画：双行刷新 + 状态提示 + 矩阵感 HUD 效果。
        """
        if self.design_level != const.SHOW_LEVEL:
            return None

        self.console.print(
            f"\n[bold #00F5FF]▶ {const.DESC} visual frame stream channel ...\n"
        )

        symbols = ["▓", "▒", "░", "□", "▣"]
        prefix = f"[bold #00FFCC][{const.DESC}::Flux]"

        make_line: callable = lambda: " ".join(random.choice(symbols) for _ in range(12))

        with Live(console=self.console, refresh_per_second=20) as live:
            for _ in range(30):
                top = make_line()
                bottom = make_line()
                content = (
                    f"{prefix} [bold #99FFFF]{top}\n"
                    f"{prefix} [bold #66FFCC]{bottom}"
                )
                live.update(Text.from_markup(content))
                await asyncio.sleep(0.12)

            # 最终完成状态
            content = (
                f"{prefix} [bold #39FF14]<< SYNCED >>\n"
                f"{prefix} [bold #00FF88]Frame Flux Ready."
            )
            live.update(Text.from_markup(content))
            await asyncio.sleep(0.4)

        self.console.print(
            f"\n[bold #7CFC00]>>> {const.DESC} frame flux online. Ready for extraction. <<<\n"
        )

    async def pulse_track(self) -> None:
        """
        脉冲轨道动画，光点推进并带渐变尾迹。
        """
        if self.design_level != const.SHOW_LEVEL:
            return None

        self.console.print(
            f"\n[bold #00F5FF]▶ {const.DESC} engine linking ...\n"
        )

        trail_colors = ["#FFA500", "#FF8C00", "#FF6347", "#444444"]
        width = int(self.console.width * 0.3) - 4

        with Live(console=self.console, refresh_per_second=60) as live:
            for _ in range(3):
                for i in range(width):
                    track = []
                    for j in range(width):
                        if j == i:
                            track.append(f"[bold #FFD700]•")
                        elif j < i and (i - j - 1) < len(trail_colors):
                            color = trail_colors[i - j - 1]
                            track.append(f"[bold {color}]⟶")
                        else:
                            track.append(f"[bold #00FF87]⟶")
                    frame = "".join(track)
                    live.update(
                        Text.from_markup(frame)
                    )
                    await asyncio.sleep(0.01)

        self.console.print(
            f"\n[bold #7CFC00]>>> {const.DESC} engine link complete. <<<\n"
        )

    async def collapse_star_expanded(self) -> None:
        """
        恒星坍缩动画（多粒子版本），粒子收缩到核心 ▣，带渐变色。
        """
        if self.design_level != const.SHOW_LEVEL:
            return None

        status_messages = [
            f"core engine linking",
            f"neural matrix syncing",
            f"AI kernel initializing",
            f"thread grid converging",
            f"bus system in handshake",
            f"logic shell activating",
            f"data lattice aligning",
            f"engine core stabilizing",
            f"runtime environment loading",
            f"execution unit binding",
            f"cognitive field forming",
            f"synaptic plane connecting",
            f"cortex node handshake",
            f"signal path entangling",
            f"neuron loop synchronization",
            f"axon stream initialized",
            f"logic thread weaving",
            f"impulse array streaming",
            f"resonance field awakening",
            f"quantum seed dispatching",
            f"shell orbiting axis",
            f"singularity calibration",
            f"echo wave expanding",
            f"stellar engine tethering",
            f"temporal gate alignment",
        ]

        self.console.print(
            f"[bold #00F5FF]\n▶ {const.DESC} {random.choice(status_messages)} ...\n"
        )

        # 基本主题
        basic_theme = [
            "#FFA07A", "#FF8C00", "#FF7F50", "#FF6347",
            "#FF4500", "#FF3030", "#FF1493", "#FF00FF",
            "#FFD700", "#FFFF33"
        ]
        # Neon Pulse（霓虹脉冲）
        neon_pulse = [
            "#4B0082", "#8A2BE2", "#9400D3", "#BA55D3",
            "#DA70D6", "#EE82EE", "#FF00FF", "#FF33CC",
            "#FF69B4", "#FFB6C1"
        ]
        # Solar Flare（日冕喷发）
        solar_flare = [
            "#8B0000", "#B22222", "#DC143C", "#FF4500",
            "#FF6347", "#FF7F50", "#FFA07A", "#FFD700",
            "#FFFF33", "#FFFF99"
        ]
        # Ocean Core（深海能核）
        ocean_core = [
            "#003366", "#004080", "#005C99", "#0073B7",
            "#0099CC", "#33CCCC", "#66FFCC", "#99FFCC",
            "#CCFFFF", "#E0FFFF"
        ]
        # 唤醒词
        wake_up_word = [
            f"{const.DESC} neural fabric linked. Consciousness online.",
            f"{const.DESC} matrix stabilized. Phase sync complete.",
            f"Core pulse achieved. {const.DESC} is now live.",
            f"{const.DESC} boot sequence resolved. Quantum path active.",
            f"{const.DESC} perception grid online. Awaiting target mapping.",
            f"{const.DESC} core in resonance. All systems synchronized.",
            f"{const.DESC} synaptic grid activated. {const.DESC} perception fully engaged.",
            f"Quantum lattice stabilized. {const.DESC} now self-aware.",
            f"{const.DESC} info stream linked. Cognitive loop complete.",
            f"Drive pulse stabilized. {const.DESC} ready for deployment."
        ]

        gradient = random.choice(
            [basic_theme, neon_pulse, solar_flare, ocean_core]
        )[::-1]  # 从外到内

        particles, offset, cycles = 27, 3, 3

        def generate_cycle() -> list[str]:
            frames = []

            # Phase 1: 收缩
            for i in range(particles, 0, -1):
                dots = [
                    f"[bold {gradient[min(j, len(gradient) - 1)]}]●"
                    for j in range(i)
                ]
                padding = " " * (particles - i + offset)
                frame = f"{padding}(" + " ".join(dots) + ")"
                frames.append(frame)

            # Phase 2: 爆发
            frames += [
                " " * (particles + offset) + "[bold #FFFF99]▣",
                " " * (particles + offset) + "[bold #9AFF9A]▣",
                " " * (particles + offset) + "[bold #00F5FF]▣",
            ]

            # Phase 3: 扩散
            for i in range(1, particles + 1):
                dots = [
                    f"[dim {gradient[min(j, len(gradient) - 1)]}]·"
                    for j in range(i)
                ]
                padding = " " * (particles - i)
                frame = f"{padding}<<< " + " ".join(dots) + " >>>"
                frames.append(frame)

            return frames

        async def flash_logo() -> typing.Coroutine | None:
            # 打字效果
            for i in range(1, len(view_char) + 1):
                typed = view_char[:i]
                live.update(
                    Text.from_markup(f"[bold {random.choice(gradient)}]{spacing}{typed}[/]")
                )  # 居中打字
                await asyncio.sleep(0.06)

            await asyncio.sleep(0.2)

            # 闪烁完整彩色版本
            for _ in range(5):
                live.update(Text.from_markup(""))  # 清空
                await asyncio.sleep(0.12)
                live.update(
                    Text.from_markup(f"[bold {random.choice(gradient)}]{spacing}{view_char}[/]")
                )  # 显示完整带样式版本
                await asyncio.sleep(0.12)

        with Live(console=self.console, refresh_per_second=30) as live:
            for _ in range(cycles):
                for c in generate_cycle():
                    live.update(Text.from_markup(c))
                    await asyncio.sleep(0.02)

            view_char = f"{const.DESC} (●) Engine"
            # view_mode = random.choice(gradient)
            spacing = " " * (particles + offset - len(view_char) // 2)
            # live.update(
            #     Text.from_markup(f"[bold {view_mode}]{spacing}{view_char}[/]")
            # )

            await flash_logo()
            await asyncio.sleep(0.2)

        self.console.print(
            f"\n[bold #7CFC00]>>> ✔ {random.choice(wake_up_word)} <<<\n"
        )

    async def neural_sync_loading(self) -> None:
        """
        神经链接激活（Neural Sync Initiation）。
        """
        if self.design_level != const.SHOW_LEVEL:
            return None

        self.console.print(
            f"\n[bold #00F5FF]▶ {const.DESC} neural engine sync ...\n"
        )

        internal_width = (width := int(self.console.width * 0.3)) - 4
        total_steps = internal_width

        sequence = []

        # 滑动渐变色：亮色向右滑动，制造“光扫”感
        gradient = [
            "#555555", "#777777", "#00F5FF", "#66FFFF", "#D7FFFF"
        ]

        gradient_len = len(gradient)

        # Phase 1: 从左向右推进 `•`，颜色跟随推进点滑动
        for step in range(1, total_steps + 1):
            frame = ""
            for i in range(step):
                color = gradient[min(step - i - 1, gradient_len - 1)]  # 从右往左套渐变色
                frame += f"[bold {color}]•"
            sequence.append(frame)

        # Phase 2: 从右向左将 • 替换为 ▦，渐变颜色从右侧推入
        fill_gradient = ["#9AFF9A", "#50F0B3", "#00D2FF", "#00BFFF", "#D7FFFF"]
        fill_len = len(fill_gradient)

        for i in range(1, total_steps + 1):
            frame = ""
            num_dots = total_steps - i
            # 灰色残留 •
            frame += "".join(f"[bold #444444]•" for _ in range(num_dots))
            # 渐变推进 ▦
            for j in range(i):
                color = fill_gradient[min(j, fill_len - 1)]
                frame += f"[bold {color}]▦"
            sequence.append(frame)

        with Live(auto_refresh=False, console=self.console) as live:
            for frame in sequence:
                panel = Panel(
                    Text.from_markup(frame), border_style="bold #00E5EE",
                    title=f"[bold #20B2AA]{const.DESC}", width=width
                )
                live.update(panel, refresh=True)
                await asyncio.sleep(0.02)

        self.console.print(
            f"\n[bold #7CFC00]>>> Sync complete. {const.DESC} intelligence online. <<<\n"
        )

    async def boot_core_sequence(self) -> None:
        """
        模拟模型核心唤醒的启动动画，融合 AI 构建感与路径感知图式。
        """
        if self.design_level != const.SHOW_LEVEL:
            return None

        self.console.print(
            f"\n[bold #00F5FF]▶ {const.DESC} waking up the sequence model ...\n"
        )

        prefix = f"[bold #FFA500][{const.DESC}::Model][/]"

        # 动态模块符号构成
        lines = {
            "Neural Link": ["═", "╪", "╋", "╂", "╫", "╬"],
            "Tensor Flow": ["░", "▒", "▓", "▤", "▥", "▦"],
            "Pulses Sync": ["⟳", "⟲", "↻", "↺", "▣", " "]
        }

        phrases = [
            "融合路径中 ...", "拓扑重建中 ...", "核心对齐中 ...", "神经接驳中 ...", "通道加载中 ...", "子图展开中 ..."
        ]

        make_row: callable = lambda x: "".join(random.choice(x) for _ in range(36))

        with Live(console=self.console, refresh_per_second=20) as live:
            for i in range(50):
                row1 = f"{prefix} [bold #87CEFA]{make_row(lines['Neural Link'])}[/]"
                row2 = f"{prefix} [bold #00E5EE]{make_row(lines['Tensor Flow'])}[/]"
                row3 = f"{prefix} [bold #FFB6C1]{make_row(lines['Pulses Sync'])}[/]"
                desc = f"{prefix} [dim]{random.choice(phrases)}[/]"
                live.update(Text.from_markup(f"{row1}\n{row2}\n{row3}\n{desc}"))
                await asyncio.sleep(0.12)

            # 完成提示
            done = f"{prefix} [bold #39FF14]▣ Model Core Connected."
            live.update(Text.from_markup(done))

        self.console.print(
            f"\n[bold #7CFC00]>>> Sequence compiler engaged. {const.DESC} intelligence online. <<<\n"
        )

    async def boot_process_sequence(self, workers: int = 5) -> None:
        """
        三段式多进程启动动画，构建→同步→注入，模拟完整并行计算体系唤醒。
        """
        if self.design_level != const.SHOW_LEVEL:
            return None

        self.console.print(
            f"\n[bold #00F5FF]▶ {const.DESC} spawning computer nodes ...\n"
        )

        prefix = f"[bold #FF8C00][{const.DESC}::Boot][/]"
        phases = [
            {"symbols": ["○", "◌"], "prompt": "初始化计算核心..."},
            {"symbols": ["◍", "◎", "◈"], "prompt": "节点握手与同步..."},
            {"symbols": ["▣", "✶", "▶"], "prompt": "注入子任务通道..."}
        ]

        current_state = ["○"] * workers  # 初始化状态

        def render(state: list[str], prompt: str) -> "Text":
            lines = [
                f"{prefix} [#AAAAAA]P{index:02d}[/] [bold #00E5EE]{x}[/]"
                for index, x in enumerate(state, start=1)
            ]
            lines.append(f"{prefix} [dim]{prompt}[/]")
            return Text.from_markup("\n".join(lines))

        with Live(console=self.console, refresh_per_second=20) as live:
            for phase in phases:
                for _ in range(6):
                    for i in range(workers):
                        if random.random() < 0.8:
                            current_state[i] = random.choice(phase["symbols"])
                    live.update(render(current_state, phase["prompt"]))
                    await asyncio.sleep(0.12)

            # 最终定格为完成状态
            for i in range(workers):
                current_state[i] = "▣"
            live.update(render(current_state, "所有任务模块已就绪"))
            await asyncio.sleep(0.5)

        self.console.print(
            f"\n[bold #7CFC00]>>> ✔ Core nodes connected. Task scheduling ready. <<<\n"
        )

    async def boot_process_matrix(self, workers: int = 5) -> None:
        """
        多进程构建动画，节奏控制，模拟进程同步、状态切换与联通构建。
        """
        if self.design_level != const.SHOW_LEVEL:
            return None

        self.console.print(
            f"\n[bold #00F5FF]▶ {const.DESC} spawning computer nodes ...\n"
        )

        prefix = f"[bold #FF8C00][{const.DESC}::Boot][/]"

        status_chain = ["○", "◍", "◎", "◈", "◉", "▣"]
        prompts = [
            "调度通道握手中...", "核心状态响应中...", "传输缓冲同步中...",
            "多核接入中...", "激活控制权...", "连接子任务流..."
        ]

        current_state, delay = [0] * workers, 0.12  # 每个进程当前的状态索引

        def render() -> "Text":
            lines = []
            for index, x in enumerate(range(workers), start=1):
                tag = f"[#AAAAAA]P{index:02d}[/]"
                symbol = status_chain[min(current_state[x], len(status_chain) - 1)]
                color = "#00E5EE" if symbol != "▣" else "#39FF14"
                lines.append(f"{prefix} {tag} [bold {color}]{symbol}[/]")
            prompt = f"{prefix} [dim]{random.choice(prompts)}[/]"
            return Text.from_markup("\n".join(lines + [prompt]))

        with Live(console=self.console, refresh_per_second=20) as live:
            for step in range(workers + 4):
                # 随机推进部分节点状态
                for i in range(workers):
                    if random.random() < 0.7 and current_state[i] < len(status_chain) - 1:
                        current_state[i] += 1
                live.update(render())
                await asyncio.sleep(delay)

            # 最终完成所有为 ▣
            for i in range(workers):
                current_state[i] = len(status_chain) - 1
                live.update(render())
                await asyncio.sleep(delay / 2)

        self.console.print(
            f"\n[bold #7CFC00]>>> ✔ Core nodes connected. Task scheduling ready. <<<\n"
        )

    async def model_manifest(self) -> None:
        """
        神经网格显影动画。
        """
        if self.design_level != const.SHOW_LEVEL:
            return None

        self.console.print(
            f"\n[bold #00F5FF]▶ {const.DESC} initializing neural pipelines ...\n"
        )

        # 神经构造字符集
        charset = [
            "⧉", "Σ", "ψ", "∇", "⊙", "⨁", "⌬", "∴", "░", "▒", "▓", "█"
        ]
        colors = [
            "#A8A8A8", "#B2B2B2", "#BCBCBC", "#C6C6C6", "#D0D0D0", "#DADADA", "#E4E4E4", "#EEEEEE"
        ]
        logo_color = random.choice(
            ["#FFD7AF", "#FFD75F", "#FFAFD7", "#FF87AF", "#FF5FAF"]
        )

        rows = random.randint(4, 6)
        cols = rows * 3

        def render_frame(current_row: int) -> "Text":
            lines = []
            for idx in range(current_row):
                row = glyph_grid[idx]
                # 加入轻微“漂移”偏移量
                indent = " " * random.randint(8, 12)
                lines.append(f"[bold {random.choice(colors)}]{indent}{' '.join(row)}[/]")
            return Text.from_markup("\n".join(lines))

        # 动画绘制过程
        with Live(console=self.console, refresh_per_second=30) as live:
            for _ in range(3):
                # 构造字符画网格
                glyph_grid = [
                    [random.choice(charset) for _ in range(cols)] for _ in range(rows)
                ]
                letters = list(const.DESC.upper())
                inserted_positions = set()

                while letters:
                    i = random.randint(0, rows - 1)
                    j = random.randint(0, cols - 1)
                    if (i, j) not in inserted_positions:
                        glyph_grid[i][j] = f"[bold {logo_color}]{letters.pop(0)}[/]"
                        inserted_positions.add((i, j))

                for frame in range(1, rows + 1):
                    live.update(render_frame(frame))
                    await asyncio.sleep(0.12)

            await asyncio.sleep(0.5)
        self.console.print(
            f"\n[bold #7CFC00]>>> {const.DESC} model instantiated. Core synthesis complete. <<<\n"
        )

    async def batch_runner_task_grid(self) -> None:
        """
        任务调度网格动画。
        """
        if self.design_level != const.SHOW_LEVEL:
            return None

        self.console.print(
            f"\n[bold #00F5FF]▶ {const.DESC} scheduling batch tasks ...\n"
        )

        def render_grid() -> str:
            padding = " " * offset
            return "\n".join(
                padding + " ".join(f"[{color}]{char}[/]" for char, color in row)
                for row in grid
            )

        def render_focus() -> str:
            padding = " " * offset
            return "\n".join(
                padding + " ".join(
                    f"[bold reverse #FFFF33]{positions[(r, c)]}[/]" if (r, c) in positions
                    else f"[dim]{char}[/]"
                    for c, (char, _) in enumerate(row)
                )
                for r, row in enumerate(grid)
            )

        rows = random.randint(4, 6)
        cols = rows * 2
        offset = 10

        # 空字符与样式池
        empty_char = "□"
        fill_chars = ["■", "▣", "▤", "▥", "▦", "▧", "▨", "▩"]

        # 冷色调颜色池
        colors = ["#00FFAA", "#00F5FF", "#1CE1D3", "#7CFC00", "#00FFFF", "#66FFCC"]

        # 初始化网格为灰色空格
        grid = [[(empty_char, "#444444") for _ in range(cols)] for _ in range(rows)]

        # 洗牌：调度顺序随机化
        coordinates = [(r, c) for r in range(rows) for c in range(cols)]
        random.shuffle(coordinates)

        letters = list(const.DESC.upper())
        positions = {}
        slots = random.sample(coordinates, len(letters))  # 随机 6 个格子

        for pos, letter in zip(slots, letters):
            positions[pos] = letter

        with Live(console=self.console, refresh_per_second=30) as live:
            for r, c in coordinates:
                if (r, c) in positions:
                    choice_char = f"[bold #FFD700]{positions[(r, c)]}[/]"  # 金色字母
                else:
                    choice_char = random.choice(fill_chars)
                grid[r][c] = (
                    choice_char, random.choice(colors)
                )
                live.update(Text.from_markup(render_grid()))
                await asyncio.sleep(0.08)

            # 收束帧，高亮字母
            live.update(Text.from_markup(render_focus()))
            await asyncio.sleep(1.5)

        self.console.print(
            f"\n[bold #7CFC00]>>> Task graph finalized. Ready to dispatch. <<<\n"
        )

    async def channel_animation(self) -> None:
        """
        多通道色带流动画。
        """
        if self.design_level != const.SHOW_LEVEL:
            return None

        self.console.print(
            f"\n[bold #00F5FF]▶ {const.DESC} preparing multi-channel pipeline ...\n"
        )

        # 动态波形池（动画阶段）
        stream = random.choice(
            ["▁▂▃▄▅▆▇█", "⎺⎻⎼⎽⎼⎻⎺", "░▒▓█▓▒░", "◜◝◞◟", "⋅∙•◦●"]
        )
        # 冷色调渐变色
        gradient = ["#87CEFA", "#00CED1", "#20B2AA", "#00FFAA", "#7CFC00", "#ADFF2F"]

        width = random.randint(28, 36)
        cycles = random.randint(1, 3)
        padding = 6
        channels = random.randint(1, 3)
        fade_frames = random.randint(8, 12)  # 前几帧淡入效果

        # 每个通道可以选择方向
        channel_directions = [random.choice([1, -1]) for _ in range(channels)]

        # 构造一帧所有通道的渲染文本
        def build_frame(offset: int) -> "Text":
            lines = []
            for ch in range(channels):
                line = " " * padding
                direction = channel_directions[ch]
                for i in range(width):
                    idx = (i * direction + offset) % len(gradient)
                    color = gradient[idx]
                    char = stream[(i + offset) % len(stream)]
                    # 淡入效果
                    if offset < fade_frames:
                        line += f"[dim {color}]{char}[/]"
                    else:
                        line += f"[bold {color}]{char}[/]"
                lines.append(line)
            content = "\n".join(lines)
            return Text.from_markup(content)

        with Live(console=self.console, refresh_per_second=30) as live:
            for frame in range(width * cycles):
                live.update(build_frame(frame))
                await asyncio.sleep(0.05)

        self.console.print(
            f"\n[bold #7CFC00]>>> {const.DESC} sample channels synchronized successfully. <<<\n"
        )

    async def wave_converge_animation(self) -> None:
        """
        镜像波纹汇聚动画。
        """
        if self.design_level != const.SHOW_LEVEL:
            return None

        self.console.print(
            f"\n[bold #00F5FF]▶ {const.DESC} spinning up autonomous loop engine ...\n"
        )

        width = 41
        padding = 4
        wave_chars = ["~", "≈", "≋", "≂"]

        gradient_sets = {
            "cyan_to_blue": [
                "#00FFFF", "#33CCFF", "#3399FF", "#3366FF", "#3333FF", "#222288", "#111144"
            ],
            "green_to_teal": [
                "#66FF66", "#33FF99", "#00FFCC", "#00CCCC", "#009999", "#006666", "#003333"
            ],
            "purple_to_pink": [
                "#FF66FF", "#CC66FF", "#9966FF", "#6633FF", "#6633CC", "#661199", "#330066"
            ],
            "orange_to_red": [
                "#FFCC66", "#FF9966", "#FF6666", "#FF3333", "#CC3333", "#992222", "#661111"
            ],
        }

        colors = [
            "#00FFFF",  # 明亮蓝绿
            "#00FF88",  # 青绿色
            "#7CFC00",  # 草绿色
            "#FFD700",  # 金黄
            "#FF69B4",  # 粉红
            "#FF4500",  # 橘红
            "#FF6347",  # 番茄红
            "#BA55D3",  # 紫罗兰
            "#00CED1",  # 深青
            "#ADD8E6",  # 淡蓝
        ]

        # 颜色梯度（左→中）+（中→右）
        _, gradient = random.choice(list(gradient_sets.items()))

        def render_wave_line_colored(step: int, wave_char: str) -> "Text":
            text = Text(" " * padding)
            wave_width = step

            # 左侧波纹（渐变）
            for i in range(wave_width):
                color_l = gradient[i % len(gradient)]
                text.append(wave_char, style=color_l)

            # 中间空隙
            middle = width - wave_width * 2
            text.append(" " * middle)

            # 右侧波纹（镜像渐变）
            for j in reversed(range(wave_width)):
                color_r = gradient[j % len(gradient)]
                text.append(wave_char, style=color_r)

            return text

        def render_frame(step: int) -> "Text":
            wave_char = wave_chars[step % len(wave_chars)]
            lines = [
                render_wave_line_colored(step, wave_char),
                render_wave_line_colored(step, wave_char),
                render_wave_line_colored(step, wave_char),
            ]
            return Text("\n").join(lines)

        max_step = (width - padding * 2) // 2 + padding
        final_symbol = f"{{{{ ⊕ {const.DESC} Engine ⊕ }}}}"

        async def flash_logo() -> typing.Coroutine | None:
            # 打字效果：从左到右逐字符打印
            for i in range(1, len(final_symbol) + 1):
                partial = final_symbol[:i]
                centered = " " * padding + partial.center(width)
                live.update(Text.from_markup("\n" + f"[bold {random.choice(colors)}]{centered}[/]" + "\n"))
                await asyncio.sleep(0.08)

            await asyncio.sleep(0.3)

            # 闪烁效果
            full_line = " " * padding + final_symbol.center(width)
            for _ in range(3):
                live.update(
                    Text.from_markup("\n" * 1 + "\n" + "\n")
                )  # 隐藏
                await asyncio.sleep(0.08)
                live.update(
                    Text.from_markup(
                        "\n".join(
                            ["", f"[bold {random.choice(colors)}]{full_line}[/]", ""]
                        )
                    )
                )  # 显示
                await asyncio.sleep(0.08)

        with Live(console=self.console, refresh_per_second=30) as live:
            for _ in range(2):
                for frame in range(max_step):
                    live.update(render_frame(frame))
                    await asyncio.sleep(0.05)

            # 最终帧
            await flash_logo()
            await asyncio.sleep(0.5)

        self.console.print(
            f"\n[bold #7CFC00]>>> {const.DESC} loop engine stabilized. Watching environment. <<<\n"
        )

    async def pixel_bloom(self) -> None:
        if self.design_level != const.SHOW_LEVEL:
            return None

        self.console.print(
            f"\n[bold #00F5FF]▶ {const.DESC} preparing canvas pipeline ...\n"
        )

        width, height, padding = 30, 5, " " * 8
        logo = list(const.DESC)
        symbols = ["⧉", "Σ", "ψ", "∇", "⊙", "⨁", "⌬", "░", "▒", "▓", "█"]
        fade_symbols = ["█", "▓", "▒", "░", "·", "·"]

        dim_backgrounds = random.choice([
            "#D8E8E8",  # 云灰青
            "#E6F0F2",  # 冰蓝白
            "#F0EAE2",  # 柔奶白
            "#F5F5DC",  # 象牙米
            "#EAEAEA",  # 雾面银灰
            "#FFEFD5",  # 浅金杏
            "#E0F7FA",  # 雾水蓝
            "#FFF8DC",  # 奶油色
            "#F5F0FF",  # 淡紫灰
            "#F0FFF0"  # 茶绿白
        ])

        colors = random.choice(
            [
                "#FFD700",  # 金色光芒 · 稳定核心感
                "#00FFAA",  # 霓虹青绿 · 智能科技感
                "#FF6EC7",  # 樱桃霓光 · AI 个性感
                "#87CEFA",  # 冰湖蓝 · 安静扩散感
                "#FF5F87",  # 激光粉 · 动感跃起
                "#7CFC00",  # 荧光绿 · 脉冲活性
                "#39FF14",  # 翠绿光 · 生长视觉
                "#00FFFF",  # 纯蓝光 · 信号纯净感
                "#FFAF00",  # 琥珀橙 · 稳定能量核心
                "#BA55D3",  # 紫电波 · 深度神经模型
                "#00CED1",  # 蓝绿中和 · 编译感
                "#FF00FF"  # 电子紫 · 数字能量场
            ]
        )

        center_r, center_c = height // 2, width // 2
        grid = [[" " for _ in range(width)] for _ in range(height)]

        def render() -> "Text":
            return Text.from_markup(
                "\n".join(
                    padding + " ".join(
                        cell if cell.startswith("[") else f"[dim {dim_backgrounds}]{cell}[/]" for cell in row
                    ) for row in grid
                )
            )

        in_bounds: callable = lambda x, y: 0 <= x < height and 0 <= y < width

        with Live(console=self.console, refresh_per_second=30) as live:
            max_radius = max(center_r, center_c)

            # 爆破绽放
            for layer in range(max_radius + 1):
                for dr in range(-layer, layer + 1):
                    for dc in range(-layer, layer + 1):
                        r, c = center_r + dr, center_c + dc
                        if in_bounds(r, c) and grid[r][c] == " ":
                            grid[r][c] = random.choice(symbols)
                live.update(render())
                await asyncio.sleep(0.04)

            # 渐淡
            for fade in fade_symbols:
                for r in range(height):
                    for c in range(width):
                        if not grid[r][c].startswith("["):
                            grid[r][c] = fade
                live.update(render())
                await asyncio.sleep(0.03)

            # 植入 LOGO
            start_c = center_c - len(logo) // 2
            for i, ch in enumerate(logo):
                if in_bounds(center_r, start_c + i):
                    grid[center_r][start_c + i] = f"[bold {colors}]{ch}[/]"
            live.update(render())

            # LOGO 闪烁 3 次
            for _ in range(5):
                await asyncio.sleep(0.15)
                for i in range(len(logo)):
                    grid[center_r][start_c + i] = " "
                live.update(render())
                await asyncio.sleep(0.1)
                for i, ch in enumerate(logo):
                    grid[center_r][start_c + i] = f"[bold {colors}]{ch}[/]"
                live.update(render())

            await asyncio.sleep(0.2)

        self.console.print(
            f"\n[bold #7CFC00]>>> {const.DESC} pixel glyph matrix stabilized. <<<"
        )

    def content_pose(self, rlt, avg, dur, org, vd_start, vd_close, vd_limit, video_temp, frate) -> None:
        """
        根据日志等级展示当前视频处理过程中的关键帧率与时长信息。
        """
        if self.design_level != const.SHOW_LEVEL:
            return None

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

        self.console.print(table_info)
        self.console.print(table_clip)

    def assort_frame(self, begin_fr, final_fr, stage_cs) -> None:
        """
        根据日志等级输出帧片段处理的起止帧号及耗时统计。
        """
        if self.design_level != const.SHOW_LEVEL:
            return None

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

        self.console.print(table)


if __name__ == '__main__':
    pass
