import os
import re
import typing
import aiofiles
from loguru import logger
from rich.tree import Tree
from rich.console import Console
from rich.logging import RichHandler
from nexaflow import const


class RichSink(RichHandler):

    def __init__(self, console: "Console"):
        super().__init__(console=console, rich_tracebacks=True, show_path=False, show_time=False)

    def emit(self, record):
        log_message = self.format(record)
        self.console.print(log_message)


class Entry(object):

    def __init__(self, title: str):
        self.title = title
        self.sheet = []

    def update_video(self, subtitle, sequence, video_path):
        self.sheet.append({
            "query": os.path.join(subtitle, sequence),
            "video": video_path
        })


class Find(object):

    @staticmethod
    def is_video_file(file: str):
        video_name = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
        return file.lower().endswith(video_name)

    def list_videos_in_directory(self, folder):
        video_file_list = []
        if os.path.exists(folder):
            with os.scandir(folder) as entries:
                for entry in entries:
                    if entry.is_file() and self.is_video_file(entry.name):
                        video_file_list.append(entry.path)
        return video_file_list

    def find_sequence(self, sequence_path: str):
        video_folder_path = os.path.join(sequence_path, "video")
        return self.list_videos_in_directory(video_folder_path)

    def find_subtitle(self, subtitle_path, subtitle_tree):
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

    def find_title(self, title_path: str, title_tree: "Tree"):
        entry = Entry(title_path.split(os.path.sep)[-1])
        with os.scandir(title_path) as subtitles:
            for subtitle_entry in subtitles:
                if subtitle_entry.is_dir():
                    subtitle_tree = title_tree.add(f"📁 Subtitle: {subtitle_entry.name}", style="bold #E9967A")
                    videos = self.find_subtitle(subtitle_entry.path, subtitle_tree)
                    for sequence, video in videos:
                        entry.update_video(subtitle_entry.name, sequence, video)
        return entry

    def find_collection(self, collection_path: str, collection_tree: "Tree"):
        entries = []
        with os.scandir(collection_path) as titles:
            for title_entry in titles:
                if title_entry.is_dir():
                    title_tree = collection_tree.add(f"📀 Title: {title_entry.name}", style="bold #FFDAB9")
                    entry = self.find_title(title_entry.path, title_tree)
                    entries.append(entry)
        return entries

    def accelerate(self, base_folder: str):
        if not os.path.exists(base_folder):
            return FramixAnalyzerError(f"文件夹错误")

        root_tree = Tree(
            f"🌐 [bold #FFA54F]Video Library: {os.path.relpath(base_folder)}[/]",
            guide_style="bold #AEEEEE"
        )
        collection_list = []
        with os.scandir(base_folder) as collection:
            for collection_entry in collection:
                if collection_entry.is_dir() and (name := collection_entry.name) == "Nexa_Collection":
                    title_tree = root_tree.add(f"📂 Collection: {name}", style="bold #FDF5E6")
                    entries = self.find_collection(collection_entry.path, title_tree)
                    collection_list.append(entries)

        if len(collection_list) == 0:
            return FramixAnalyzerError(f"没有任何视频文件")
        return root_tree, collection_list


class Craft(object):

    @staticmethod
    async def revise_path(path: typing.Union[str, "os.PathLike"]) -> str:
        pattern = r"[\x00-\x1f\x7f-\x9f\u2000-\u20ff\u202a-\u202e]"
        return re.sub(pattern, "", path)

    @staticmethod
    async def achieve(
            template: typing.Union[str, "os.PathLike"]
    ) -> typing.Union[str, "Exception"]:

        try:
            async with aiofiles.open(template, "r", encoding=const.CHARSET) as f:
                template_file = await f.read()
        except FileNotFoundError as e:
            return e
        return template_file


class Active(object):

    @staticmethod
    def active(log_level: str):
        logger.remove(0)
        logger.add(
            RichSink(Console()), level=log_level.upper(), format=const.PRINT_FORMAT, diagnose=False
        )


class Review(object):

    __material: tuple = tuple()

    def __init__(self, *args):
        self.material = args

    @property
    def material(self):
        return self.__material

    @material.setter
    def material(self, value):
        self.__material = value

    def __str__(self):
        start, end, cost, *_ = self.material
        return f"<Review start={start} end={end} cost={cost}>"

    __repr__ = __str__


class FramixError(Exception):
    pass


class FramixAnalysisError(FramixError):

    def __init__(self, msg: typing.Any):
        self.msg = msg


class FramixAnalyzerError(FramixError):

    def __init__(self, msg: typing.Any):
        self.msg = msg


class FramixReporterError(FramixError):

    def __init__(self, msg: typing.Any):
        self.msg = msg


if __name__ == '__main__':
    pass
