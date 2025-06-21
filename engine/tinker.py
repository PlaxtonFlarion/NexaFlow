#   _____ _       _
#  |_   _(_)_ __ | | _____ _ __
#    | | | | '_ \| |/ / _ \ '__|
#    | | | | | | |   <  __/ |
#    |_| |_|_| |_|_|\_\___|_|
#
# ==== Notes: License ====
# Copyright (c) 2024  Framix :: 画帧秀
# This file is licensed under the Framix :: 画帧秀 License. See the LICENSE.md file for more details.

import os
import re
import sys
import json
import random
import shutil
import typing
import aiofiles
from loguru import logger
from rich.text import Text
from rich.tree import Tree
from rich.console import Console
from rich.logging import (
    LogRecord, RichHandler
)
from engine.terminal import Terminal
from nexacore.design import Design
from nexaflow import const


class _FramixBaseError(Exception):
    """
    Framix 异常体系的基础类。

    作为所有 Framix 异常继承根，用于构建项目中统一的异常结构。
    继承自内建 Exception 类，不包含额外逻辑，仅用于类型标识。
    """
    pass


class FramixError(_FramixBaseError):
    """
    Framix 的通用异常类，用于抛出带有上下文信息的错误。

    支持传入自定义错误消息，实例化后通过 __str__ 方法格式化输出，
    便于日志系统与控制台清晰展示错误来源。

    Parameters
    ----------
    msg : Optional[Any], default=None
        异常的提示内容，可为字符串、异常对象或任意可格式化输出的对象。

    Attributes
    ----------
    msg : Optional[Any]
        存储的异常消息内容，在打印与日志记录中使用。
    """

    def __init__(self, msg: typing.Optional[typing.Any] = None):
        self.msg = msg

    def __str__(self):
        return f"<{const.DESC}Error> {self.msg}"

    __repr__ = __str__


class FileAssist(object):
    """
    文件操作工具类，用于读取与写入 JSON 文件。
    """

    @staticmethod
    async def load_parameters(src: typing.Any) -> typing.Any:
        """
        从指定路径读取 JSON 文件内容并解析为 Python 字典对象。

        Parameters
        ----------
        src : typing.Any
            JSON 文件路径，支持字符串或路径对象。

        Returns
        -------
        typing.Any
            返回解析后的配置对象（通常为 `dict` 类型）。
        """
        async with aiofiles.open(src, "r", encoding=const.CHARSET) as file:
            return json.loads(await file.read())

    @staticmethod
    async def dump_parameters(src: typing.Any, dst: dict) -> None:
        """
        将配置参数以 JSON 格式写入指定路径的文件。

        Parameters
        ----------
        src : typing.Any
            目标文件路径，可为字符串或路径对象，用于存储配置数据。

        dst : dict
            待写入的配置数据字典。
        """
        async with aiofiles.open(src, "w", encoding=const.CHARSET) as file:
            await file.write(
                json.dumps(dst, indent=4, separators=(",", ":"), ensure_ascii=False)
            )


class Active(object):
    """
    日志激活器类，用于配置日志系统的输出通道与格式。

    通过接入自定义的 Rich 控制台处理器（_RichSink），将日志输出格式化为高亮、
    美观、结构化的信息流，适用于 CLI 终端中的即时日志反馈。
    """

    class _RichSink(RichHandler):
        """
        基于 RichHandler 的日志输出接收器，用于自定义控制台美化输出。

        _RichSink 继承自 rich.logging.RichHandler，重载 emit 方法，
        将日志信息通过 rich 控制台格式化输出，适用于实时、美观的日志展示。

        Parameters
        ----------
        console : Console
            rich 提供的 Console 实例，用于渲染日志文本与样式。
        """
        debug_color = [
            "#00CED1",  # 深青色 - 冷静理性
            "#7FFFD4",  # 冰蓝绿 - 轻盈科技
            "#66CDAA",  # 中度绿松石 - 适合背景级别
            "#20B2AA",  # 浅海蓝 - 稳定中间调
            "#5F9EA0",  # 军蓝灰 - 稳重调试色
            "#87CEEB",  # 天蓝 - 清晰非干扰性
            "#4682B4",  # 钢蓝 - 稍微暗一点用于子模块
            "#98FB98",  # 浅绿色 - 绿色无压调试层
            "#B0C4DE",  # 灰蓝色 - 安静辅助信息
            "#AAAAAA",  # 中灰 - 用于淡化无关 debug 流
        ]
        level_style = {
            "DEBUG": f"bold {random.choice(debug_color)}",
            "INFO": "bold #00FF7F",
            "WARNING": "bold #FFD700",
            "ERROR": "bold #FF4500",
            "CRITICAL": "bold #FF1493",
        }

        def __init__(self, console: "Console"):
            super().__init__(
                console=console, rich_tracebacks=True, show_path=False,
                show_time=False, markup=False
            )

        def emit(self, record: "LogRecord") -> None:
            """
            重载日志处理器的输出逻辑，将格式化后的记录打印到指定控制台。
            """
            self.console.print(
                const.PRINT_HEAD, Text(self.format(record), style=self.level_style.get(
                    record.levelname, "bold #ADD8E6"
                ))
            )

    @staticmethod
    def active(log_level: str) -> None:
        """
        配置并激活 Rich 控制台日志处理器。

        移除默认日志处理器，添加自定义 _RichSink 实例，结合 rich.console
        提供彩色输出支持，并通过传入的 log_level 控制日志等级。

        Parameters
        ----------
        log_level : str
            日志等级（如 "INFO", "DEBUG", "WARNING", "ERROR"），不区分大小写。
        """
        logger.remove(0)
        logger.add(
            Active._RichSink(Design.console),
            level=log_level.upper(), format=const.PRINT_FORMAT, diagnose=False
        )


class Entry(object):
    """
    用于构建视频处理任务条目的容器类。

    Entry 用于组织一组与标题关联的视频条目，每个条目包含查询路径和对应的视频路径。
    适用于视频批量分析、字幕序列管理或脚本生成任务中，提供结构化数据记录能力。

    Parameters
    ----------
    title : str
        当前任务条目的标题标识，通常为任务名、类别或唯一名称。

    Attributes
    ----------
    title : str
        条目标题，用于标识该组视频或任务集合。

    sheet : list[dict]
        存储的所有视频记录，每项为包含 query 路径和 video 路径的字典。
    """

    def __init__(self, title: str):
        self.title = title
        self.sheet = []

    def update_video(self, subtitle: str, sequence: str, video_path: str) -> None:
        """
        向 sheet 添加一个新的视频记录，生成对应的 query 路径。
        """
        self.sheet.append({
            "query": os.path.join(subtitle, sequence),
            "video": video_path
        })


class Craft(object):
    """
    Craft 类提供静态方法，用于路径修正与模板文件的异步读取。

    该工具类用于辅助处理文件系统相关的内容清洗与模板加载操作，
    常用于日志路径、配置路径或 HTML 模板的预处理阶段。
    """

    @staticmethod
    async def editor(file_path: str) -> None:
        """
        调用系统默认文本编辑器以异步方式打开指定配置文件。

        Parameters
        ----------
        file_path : str
            要打开的配置文件路径，支持绝对路径或相对路径。

        Notes
        -----
        - 在 Windows 平台：
            - 优先尝试调用 `notepad++`；
            - 若未安装，则退回使用系统默认的 `Notepad`。
        - 在 macOS 平台：
            - 调用系统内建的 `TextEdit`（以阻塞模式 `-W` 打开）。
        - 编辑器调用过程为异步命令执行，不会阻塞主线程；
          适用于配置快速预览或编辑器集成场景。
        """
        if sys.platform == "win32":
            first = ["notepad++"] if shutil.which("notepad++") else ["Notepad"]
        else:
            first = ["open", "-W", "-a", "TextEdit"]

        await Terminal.cmd_line(first + [file_path])

    @staticmethod
    async def revise_path(path: typing.Union[str, "os.PathLike"]) -> str:
        """
        清理路径字符串中的控制字符与特殊不可见符号。

        使用正则表达式匹配 ASCII 控制字符（0x00–0x1F、0x7F–0x9F）以及
        Unicode 中的格式控制符（如空格变体、方向控制符等），将其清除。

        Parameters
        ----------
        path : Union[str, os.PathLike]
            待清理的路径字符串。

        Returns
        -------
        str
            已清除非法字符的路径字符串。
        """
        return re.sub("[\x00-\x1f\x7f-\x9f\u2000-\u20ff\u202a-\u202e]", "", path)

    @staticmethod
    async def achieve(template: typing.Union[str, "os.PathLike"]) -> str:
        """
        异步读取指定模板文件的内容。

        用于加载 HTML、文本等外部文件资源，自动设置字符编码并处理文件不存在异常。

        Parameters
        ----------
        template : Union[str, os.PathLike]
            要读取的模板文件路径。

        Returns
        -------
        str
            文件内容的完整字符串形式。

        Raises
        ------
        FramixError
            若模板文件未找到或无法打开，则抛出该错误。
        """
        try:
            async with aiofiles.open(template, "r", encoding=const.CHARSET) as f:
                template_file = await f.read()
        except FileNotFoundError as e:
            raise FramixError(e)

        return template_file


class Search(object):
    """
    视频资源检索与结构遍历工具类。

    提供从目录结构中递归查找视频集合的能力，自动构建树形展示结构（rich.Tree），
    并将每层结构组织为 Entry 数据对象，适用于批量视频处理任务的数据预加载阶段。
    """

    @staticmethod
    def is_video_file(file: str) -> bool:
        """
        判断文件是否为视频类型（基于扩展名）。
        """
        video_name = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
        return file.lower().endswith(video_name)

    def list_videos_in_directory(self, folder: str) -> list[str]:
        """
        列出指定目录下的所有视频文件路径。
        """
        video_file_list = []
        if os.path.exists(folder):
            with os.scandir(folder) as entries:
                for entry in entries:
                    if entry.is_file() and self.is_video_file(entry.name):
                        video_file_list.append(entry.path)
        return video_file_list

    def find_sequence(self, sequence_path: str) -> list[str]:
        """
        查找指定序列路径下的所有视频文件。
        """
        video_folder_path = os.path.join(sequence_path, "video")
        return self.list_videos_in_directory(video_folder_path)

    def find_subtitle(self, subtitle_path: str, subtitle_tree: "Tree") -> list[tuple[str, str]]:
        """
        遍历字幕目录下的序列结构，并构建层级展示树与数据记录。
        """
        all_videos = []
        with os.scandir(subtitle_path) as sequences:
            for sequence_entry in sequences:
                if sequence_entry.is_dir():
                    videos = self.find_sequence(sequence_entry.path)
                    sequence_tree = subtitle_tree.add(f"📂 Sequence: {sequence_entry.name}", style="bold #F5FFFA")
                    for video in videos:
                        sequence_tree.add(f"🎥 Video Path: {video}", style="bold #E6E6FA")
                        all_videos.append((sequence_entry.name, video))  # 保存序列号和视频路径
        return all_videos

    def find_title(self, title_path: str, title_tree: "Tree") -> "Entry":
        """
        查找标题目录下的所有字幕及其下的视频，返回 Entry 对象。
        """
        entry = Entry(title_path.split(os.path.sep)[-1])
        with os.scandir(title_path) as subtitles:
            for subtitle_entry in subtitles:
                if subtitle_entry.is_dir():
                    subtitle_tree = title_tree.add(f"📁 Subtitle: {subtitle_entry.name}", style="bold #E9967A")
                    videos = self.find_subtitle(subtitle_entry.path, subtitle_tree)
                    for sequence, video in videos:
                        entry.update_video(subtitle_entry.name, sequence, video)
        return entry

    def find_collection(self, collection_path: str, collection_tree: "Tree") -> list["Entry"]:
        """
        解析视频集合目录，构建 Entry 列表并构造标题级展示树。
        """
        entries = []
        with os.scandir(collection_path) as titles:
            for title_entry in titles:
                if title_entry.is_dir():
                    title_tree = collection_tree.add(f"📀 Title: {title_entry.name}", style="bold #FFDAB9")
                    entry = self.find_title(title_entry.path, title_tree)
                    entries.append(entry)
        return entries

    def accelerate(self, base_folder: str) -> typing.Union[list, "FramixError"]:
        """
        快速加载指定目录下的视频集合结构，返回可视化树与视频数据。
        """
        if not os.path.exists(base_folder):
            return FramixError(f"文件夹不存在 {base_folder}")

        root_tree = Tree(
            f"🌐 [bold #FFA54F]Video Library: {base_folder}[/]",
            guide_style="bold #AEEEEE"
        )

        collection_list = []
        with os.scandir(base_folder) as collection:
            for collection_entry in collection:
                if collection_entry.is_dir() and (name := collection_entry.name) == const.R_COLLECTION:
                    title_tree = root_tree.add(f"📂 Collection: {name}", style="bold #FDF5E6")
                    entries = self.find_collection(collection_entry.path, title_tree)
                    collection_list.append(entries)

        if not collection_list:
            return FramixError(f"没有视频文件 {base_folder}")

        Design.console.print(root_tree)
        return collection_list


class Review(object):
    """
    用于封装测试回顾信息的轻量对象类。

    Review 类用于存储任务执行的时间段与耗时等元信息，便于后续输出结构化摘要或日志记录。
    支持通过 material 属性传入任意长度的元组数据，并在字符串表示中提取前 3 个元素进行展示。

    Attributes
    ----------
    material : Union[Any, tuple]
        存储的任意形式的元信息数据，通常为 (start, end, cost, ...) 的结构。
    """
    __material: tuple = tuple()

    def __init__(self, *args, **__):
        self.material = args

    @property
    def material(self) -> typing.Optional[typing.Union[typing.Any, tuple]]:
        return self.__material

    @material.setter
    def material(self, value):
        self.__material = value

    def __str__(self):
        start, end, cost, *_ = self.material
        return f"<Review start={start} end={end} cost={cost}>"

    __repr__ = __str__


if __name__ == '__main__':
    pass
