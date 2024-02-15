import os
import re
import sys
import cv2
import time
import json
import shutil
import random
import asyncio
import aiofiles
import datetime
import numpy as np
from pathlib import Path
from loguru import logger
from rich.prompt import Prompt
from typing import Optional
from frameflow.skills.database import DataBase
from frameflow.skills.show import Show
from frameflow.skills.manage import Manage
from frameflow.skills.parameters import Deploy, Option

operation_system = sys.platform.strip().lower()
work_platform = os.path.basename(os.path.abspath(sys.argv[0])).lower()
exec_platform = [
    "framix.exe", "framix.bin", "framix", "framix.py",
    "framix-rc.exe", "framix-rc.bin", "framix-rc", "framix-rc.py",
]

if work_platform == "framix.exe" or work_platform == "framix-rc.exe":
    _job_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    _universal = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
elif work_platform == "framix.bin" or work_platform == "framix-rc.bin":
    _job_path = os.path.dirname(sys.executable)
    _universal = os.path.dirname(os.path.dirname(sys.executable))
elif work_platform == "framix" or work_platform == "framix-rc":
    _job_path = os.path.dirname(sys.executable)
    _universal = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))))
elif work_platform == "framix.py" or work_platform == "framix-rc.py":
    _job_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _universal = os.path.dirname(os.path.abspath(__file__))
else:
    Show.console.print(
        f"[bold]Application name must be [bold red]framix[/bold red] ...[/bold]"
    )
    Show.simulation_progress(
        f"Exit after 5 seconds ...", 1, 0.05
    )
    sys.exit(1)

_tools_path = os.path.join(_job_path, "archivix", "tools")
_model_path = os.path.join(_job_path, "archivix", "molds")
_main_total_temp = os.path.join(_job_path, "archivix", "pages", "template_main_total.html")
_main_temp = os.path.join(_job_path, "archivix", "pages", "template_main.html")
_view_total_temp = os.path.join(_job_path, "archivix", "pages", "template_view_total.html")
_view_temp = os.path.join(_job_path, "archivix", "pages", "template_view.html")
_alien = os.path.join(_job_path, "archivix", "pages", "template_alien.html")
_initial_report = os.path.join(_universal, "framix.report")
_initial_deploy = os.path.join(_universal, "framix.source")
_initial_option = os.path.join(_universal, "framix.source")

if operation_system == "win32":
    _adb = os.path.join(_tools_path, "win", "platform-tools", "adb.exe")
    _ffmpeg = os.path.join(_tools_path, "win", "ffmpeg", "bin", "ffmpeg.exe")
    _ffprobe = os.path.join(_tools_path, "win", "ffmpeg", "bin", "ffprobe.exe")
    _scrcpy = os.path.join(_tools_path, "win", "scrcpy", "scrcpy.exe")
elif operation_system == "darwin":
    _adb = os.path.join(_tools_path, "mac", "platform-tools", "adb")
    _ffmpeg = os.path.join(_tools_path, "mac", "ffmpeg", "bin", "ffmpeg")
    _ffprobe = os.path.join(_tools_path, "mac", "ffmpeg", "bin", "ffprobe")
    _scrcpy = os.path.join(_tools_path, "mac", "scrcpy", "bin", "scrcpy")
else:
    Show.console.print(
        "[bold]Only compatible with [bold red]Windows[/bold red] and [bold red]macOS[/bold red] platforms ...[/bold]"
    )
    Show.simulation_progress(
        f"Exit after 5 seconds ...", 1, 0.05
    )
    sys.exit(1)

os.environ["PATH"] = os.path.dirname(_adb) + os.path.pathsep + os.environ.get("PATH", "")
os.environ["PATH"] = os.path.dirname(_ffmpeg) + os.path.pathsep + os.environ.get("PATH", "")
os.environ["PATH"] = os.path.dirname(_ffprobe) + os.path.pathsep + os.environ.get("PATH", "")
os.environ["PATH"] = os.path.dirname(_scrcpy) + os.path.pathsep + os.environ.get("PATH", "")

try:
    from nexaflow import toolbox
    from nexaflow.terminal import Terminal
    from nexaflow.skills.report import Report
    from nexaflow.video import VideoObject, VideoFrame
    from nexaflow.cutter.cutter import VideoCutter
    from nexaflow.hook import CompressHook, FrameSaveHook
    from nexaflow.hook import PaintCropHook, PaintOmitHook
    from nexaflow.classifier.base import ClassifierResult
    from nexaflow.classifier.keras_classifier import KerasClassifier
    from nexaflow.classifier.framix_classifier import FramixClassifier
except (RuntimeError, ModuleNotFoundError) as err:
    Show.console.print(f"[bold red]Error: {err}")
    Show.simulation_progress(
        f"Exit after 5 seconds ...", 1, 0.05
    )
    sys.exit(1)


class Parser(object):

    @staticmethod
    def parse_scale(dim_str):
        try:
            float_val = float(dim_str) if dim_str else None
        except ValueError:
            return None
        return round(max(0.1, min(1.0, float_val)), 2) if float_val else None

    @staticmethod
    def parse_sizes(dim_str):
        if isinstance(dim_str, tuple):
            return dim_str
        elif isinstance(dim_str, str):
            match_size_list = re.findall(r"-?\d*\.?\d+", dim_str)
            if len(match_size_list) >= 2:
                converted = []
                for num in match_size_list:
                    try:
                        converted_num = int(num)
                    except ValueError:
                        converted_num = float(num)
                    converted.append(converted_num)
                return tuple(converted[:2])
        return None

    @staticmethod
    def parse_mills(dim_str):
        if isinstance(dim_str, int | float):
            return float(dim_str)
        if isinstance(dim_str, str):
            seconds_pattern = re.compile(r"^\d+(\.\d+)?$")
            full_pattern = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})(\.\d+)?")
            if match := full_pattern.match(dim_str):
                hours, minutes, seconds, milliseconds = match.groups()
                total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
                if milliseconds:
                    total_seconds += float(milliseconds)
                return total_seconds
            elif seconds_pattern.match(dim_str):
                return float(dim_str)
        return None

    @staticmethod
    def parse_stage(dim_str):
        stage_parts = []
        parts = re.split(r"[.,;:\s]+", dim_str)
        match_parts = [part for part in parts if re.match(r"-?\d+(\.\d+)?", part)]
        for number in match_parts:
            try:
                stage_parts.append(int(number))
            except ValueError:
                stage_parts = []
                break
        return tuple(stage_parts[:2]) if len(stage_parts) >= 2 else None

    @staticmethod
    def parse_cmd():
        parser = ArgumentParser(description="Command Line Arguments Framix")

        parser.add_argument('--flick', action='store_true', help='循环分析视频帧')
        parser.add_argument('--paint', action='store_true', help='绘制图片分割线条')
        parser.add_argument('--video', action='append', help='分析视频')
        parser.add_argument('--stack', action='append', help='分析视频文件集合')
        parser.add_argument('--merge', action='append', help='聚合时间戳报告')
        parser.add_argument('--union', action='append', help='聚合视频帧报告')
        parser.add_argument('--train', action='append', help='归类图片文件')
        parser.add_argument('--build', action='append', help='训练模型文件')

        parser.add_argument('--alone', action='store_true', help='独立控制')
        parser.add_argument('--group', action='store_true', help='分组报告')
        parser.add_argument('--quick', action='store_true', help='快速模式')
        parser.add_argument('--basic', action='store_true', help='基础模式')
        parser.add_argument('--keras', action='store_true', help='智能模式')

        parser.add_argument('--boost', action='store_true', help='跳帧模式')
        parser.add_argument('--color', action='store_true', help='彩色模式')
        parser.add_argument('--shape', nargs='?', const=None, type=Parser.parse_sizes, help='图片尺寸')
        parser.add_argument('--scale', nargs='?', const=None, type=Parser.parse_scale, help='缩放比例')
        parser.add_argument('--begin', nargs='?', const=None, type=Parser.parse_stage, help='开始帧')
        parser.add_argument('--final', nargs='?', const=None, type=Parser.parse_stage, help='结束帧')
        parser.add_argument('--start', nargs='?', const=None, type=Parser.parse_mills, help='开始时间')
        parser.add_argument('--close', nargs='?', const=None, type=Parser.parse_mills, help='结束时间')
        parser.add_argument('--limit', nargs='?', const=None, type=Parser.parse_mills, help='持续时间')
        parser.add_argument('--crops', action='append', help='获取区域')
        parser.add_argument('--omits', action='append', help='忽略区域')

        # 调试模式
        parser.add_argument('--debug', action='store_true', help='调试模式')

        return parser.parse_args()


class Missions(object):

    COMPRESS: int | float = 0.4  # 默认压缩率

    def __init__(self, alone: bool, quick: bool, basic: bool, keras: bool, group: bool, *args, **kwargs):
        self.alone, self.quick, self.basic, self.keras, self.group = alone, quick, basic, keras, group
        self.boost, self.color, self.shape, self.scale, *_ = args
        *_, self.start, self.close, self.limit, self.begin, self.final, _, _ = args
        *_, self.crops, self.omits = args

        self.lines = kwargs["lines"]
        self.model_path = kwargs["model_path"]
        self.main_total_temp = kwargs["main_total_temp"]
        self.main_temp = kwargs["main_temp"]
        self.view_total_temp = kwargs["view_total_temp"]
        self.view_temp = kwargs["view_temp"]
        self.alien = kwargs["alien"]
        self.initial_report = kwargs["initial_report"]
        self.initial_deploy = kwargs["initial_deploy"]
        self.initial_option = kwargs["initial_option"]
        self.adb = kwargs["adb"]
        self.ffmpeg = kwargs["ffmpeg"]
        self.ffprobe = kwargs["ffprobe"]
        self.scrcpy = kwargs["scrcpy"]

    @staticmethod
    def only_video(folder: str):

        class Entry(object):

            def __init__(self, title: str, place: str, sheet: list):
                self.title = title
                self.place = place
                self.sheet = sheet

        return [
            Entry(
                Path(root).name, root,
                [os.path.join(root, f) for f in sorted(file) if Path(f).name.split(".")[-1] != "log"]
            )
            for root, _, file in os.walk(folder) if file
        ]

    def video_task(self, video_file: str):
        reporter = Report(total_path=self.initial_report)
        reporter.title = f"Framix_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        reporter.query = time.strftime('%Y%m%d%H%M%S')
        new_video_path = os.path.join(reporter.video_path, os.path.basename(video_file))

        shutil.copy(video_file, new_video_path)

        deploy = Deploy(self.initial_deploy)
        deploy.alone = self.alone
        deploy.group = self.group
        deploy.quick = self.quick
        deploy.basic = self.basic
        deploy.keras = self.keras

        deploy.boost = self.boost
        deploy.color = self.color

        deploy.shape = self.shape if "--shape" in self.lines else deploy.shape
        deploy.scale = self.scale if "--scale" in self.lines else deploy.scale
        deploy.start = self.start if "--start" in self.lines else deploy.start
        deploy.close = self.close if "--close" in self.lines else deploy.close
        deploy.limit = self.limit if "--limit" in self.lines else deploy.limit
        deploy.begin = self.begin if "--begin" in self.lines else deploy.begin
        deploy.final = self.final if "--final" in self.lines else deploy.final

        deploy.crops = self.crops if "--crops" in self.lines else deploy.crops
        deploy.omits = self.omits if "--omits" in self.lines else deploy.omits

        looper = asyncio.get_event_loop()

        if deploy.quick:
            logger.info(f"Framix Analyzer: 快速模式 ...")
            video_filter = [f"fps={deploy.fps}"] if deploy.color else [f"fps={deploy.fps}", "format=gray"]
            if deploy.shape:
                original_shape = looper.run_until_complete(
                    ask_video_larger(self.ffprobe, new_video_path)
                )
                w, h, ratio = looper.run_until_complete(
                    ask_magic_frame(original_shape, deploy.shape)
                )
                logger.debug(f"Image Shape: [W:{w} H{h} Ratio:{ratio}]")
                video_filter.append(f"scale={w}:{h}")
            elif deploy.scale:
                scale = max(0.1, min(1.0, deploy.scale))
                video_filter.append(f"scale=iw*{scale}:ih*{scale}")
                logger.debug(f"Image Scale: {deploy.scale}")
            else:
                video_filter.append(f"scale=iw*{self.COMPRESS}:ih*{self.COMPRESS}")
            logger.info(f"应用过滤器: {video_filter}")

            duration = looper.run_until_complete(
                ask_video_length(self.ffprobe, new_video_path)
            )
            vision_start, vision_close, vision_limit = examine_flip(
                deploy.parse_mills(deploy.start),
                deploy.parse_mills(deploy.close),
                deploy.parse_mills(deploy.limit),
                duration
            )
            vision_start = deploy.parse_times(vision_start)
            vision_close = deploy.parse_times(vision_close)
            vision_limit = deploy.parse_times(vision_limit)
            logger.info(f"视频时长: [{duration}] [{deploy.parse_times(duration)}]")
            logger.info(f"start=[{vision_start}] - close=[{vision_close}] - limit=[{vision_limit}]")

            looper.run_until_complete(
                ask_video_detach(
                    self.ffmpeg, video_filter, new_video_path, reporter.frame_path,
                    start=vision_start, close=vision_close, limit=vision_limit
                )
            )
            result = {
                "total_path": Path(reporter.total_path).name,
                "title": reporter.title,
                "query": reporter.query,
                "stage": {"start": 0, "end": 0, "cost": 0},
                "frame": Path(reporter.frame_path).name
            }
            logger.debug(f"Quick: {result}")
            reporter.load(result)

            looper.run_until_complete(
                reporter.ask_invent_total_report(
                    os.path.dirname(reporter.total_path),
                    get_template(self.view_temp),
                    get_template(self.view_total_temp),
                    deploy.group
                )
            )
            return reporter.total_path

        elif deploy.keras and not deploy.basic:
            logger.info(f"Framix Analyzer: 智能模式 ...")
            kc = KerasClassifier(data_size=deploy.model_size)
            try:
                kc.load_model(self.model_path)
            except ValueError as e:
                logger.error(f"发生 {e}")
                kc = None
        else:
            logger.info(f"Framix Analyzer: 基础模式 ...")
            kc = None

        futures = looper.run_until_complete(
            analyzer(
                new_video_path, deploy, kc, reporter.frame_path, reporter.extra_path,
                ffmpeg=self.ffmpeg, ffprobe=self.ffprobe
            )
        )

        if futures is None:
            return None
        start, end, cost, classifier = futures

        result = {
            "total_path": Path(reporter.total_path).name,
            "title": reporter.title,
            "query": reporter.query,
            "stage": {"start": start, "end": end, "cost": f"{cost:.5f}"},
            "frame": Path(reporter.frame_path).name
        }

        if classifier:
            original_inform = reporter.draw(
                classifier_result=classifier,
                proto_path=reporter.proto_path,
                template_file=get_template(self.alien),
            )
            result["extra"] = Path(reporter.extra_path).name
            result["proto"] = Path(original_inform).name

        logger.debug(f"Restore: {result}")
        reporter.load(result)

        with DataBase(os.path.join(reporter.reset_path, "Framix_Data.db")) as database:
            if classifier:
                column_list = [
                    'total_path', 'title', 'query_path', 'query', 'stage', 'frame_path', 'extra_path', 'proto_path'
                ]
                database.create('stocks', *column_list)
                stage = {'stage': {'start': start, 'end': end, 'cost': cost}}
                database.insert(
                    'stocks', column_list,
                    (reporter.total_path, reporter.title, reporter.query_path, reporter.query, json.dumps(stage),
                     reporter.frame_path, reporter.extra_path, reporter.proto_path)
                )
            else:
                column_list = [
                    'total_path', 'title', 'query_path', 'query', 'stage', 'frame_path'
                ]
                database.create('stocks', *column_list)
                stage = {'stage': {'start': start, 'end': end, 'cost': cost}}
                database.insert(
                    'stocks', column_list,
                    (reporter.total_path, reporter.title, reporter.query_path, reporter.query, json.dumps(stage),
                     reporter.frame_path)
                )

        looper.run_until_complete(
            reporter.ask_create_total_report(
                os.path.dirname(reporter.total_path),
                get_template(self.main_temp),
                get_template(self.main_total_temp),
                deploy.group
            )
        )
        return reporter.total_path

    def video_dir_task(self, folder: str):
        reporter = Report(total_path=self.initial_report)

        deploy = Deploy(self.initial_deploy)
        deploy.alone = self.alone
        deploy.group = self.group
        deploy.quick = self.quick
        deploy.basic = self.basic
        deploy.keras = self.keras

        deploy.boost = self.boost
        deploy.color = self.color

        deploy.shape = self.shape if "--shape" in self.lines else deploy.shape
        deploy.scale = self.scale if "--scale" in self.lines else deploy.scale
        deploy.start = self.start if "--start" in self.lines else deploy.start
        deploy.close = self.close if "--close" in self.lines else deploy.close
        deploy.limit = self.limit if "--limit" in self.lines else deploy.limit
        deploy.begin = self.begin if "--begin" in self.lines else deploy.begin
        deploy.final = self.final if "--final" in self.lines else deploy.final

        deploy.crops = self.crops if "--crops" in self.lines else deploy.crops
        deploy.omits = self.omits if "--omits" in self.lines else deploy.omits

        looper = asyncio.get_event_loop()

        if deploy.quick:
            logger.debug(f"Framix Analyzer: 快速模式 ...")
            for video in self.only_video(folder):
                reporter.title = video.title
                for path in video.sheet:
                    reporter.query = os.path.basename(path).split(".")[0]
                    shutil.copy(path, reporter.video_path)
                    new_video_path = os.path.join(reporter.video_path, os.path.basename(path))

                    video_filter = [f"fps={deploy.fps}"] if deploy.color else [f"fps={deploy.fps}", "format=gray"]
                    if deploy.shape:
                        original_shape = looper.run_until_complete(
                            ask_video_larger(self.ffprobe, new_video_path)
                        )
                        w, h, ratio = looper.run_until_complete(
                            ask_magic_frame(original_shape, deploy.shape)
                        )
                        logger.debug(f"Image Shape: [W:{w} H{h} Ratio:{ratio}]")
                        video_filter.append(f"scale={w}:{h}")
                    elif deploy.scale:
                        scale = max(0.1, min(1.0, deploy.scale))
                        video_filter.append(f"scale=iw*{scale}:ih*{scale}")
                        logger.debug(f"Image Scale: {deploy.scale}")
                    else:
                        video_filter.append(f"scale=iw*{self.COMPRESS}:ih*{self.COMPRESS}")
                    logger.info(f"应用过滤器: {video_filter}")

                    duration = looper.run_until_complete(
                        ask_video_length(self.ffprobe, new_video_path)
                    )
                    vision_start, vision_close, vision_limit = examine_flip(
                        deploy.parse_mills(deploy.start),
                        deploy.parse_mills(deploy.close),
                        deploy.parse_mills(deploy.limit),
                        duration
                    )
                    vision_start = deploy.parse_times(vision_start)
                    vision_close = deploy.parse_times(vision_close)
                    vision_limit = deploy.parse_times(vision_limit)
                    logger.info(f"视频时长: [{duration}] [{deploy.parse_times(duration)}]")
                    logger.info(f"start=[{vision_start}] - close=[{vision_close}] - limit=[{vision_limit}]")

                    looper.run_until_complete(
                        ask_video_detach(
                            self.ffmpeg, video_filter, new_video_path, reporter.frame_path,
                            start=deploy.start, close=deploy.close, limit=deploy.limit
                        )
                    )
                    result = {
                        "total_path": Path(reporter.total_path).name,
                        "title": reporter.title,
                        "query": reporter.query,
                        "stage": {"start": 0, "end": 0, "cost": 0},
                        "frame": Path(reporter.frame_path).name
                    }
                    logger.debug(f"Quick: {result}")
                    reporter.load(result)

            looper.run_until_complete(
                reporter.ask_invent_total_report(
                    os.path.dirname(reporter.total_path),
                    get_template(self.view_temp),
                    get_template(self.view_total_temp),
                    deploy.group
                )
            )
            return reporter.total_path

        elif deploy.keras and not deploy.basic:
            logger.info(f"Framix Analyzer: 智能模式 ...")
            kc = KerasClassifier(data_size=deploy.model_size)
            try:
                kc.load_model(self.model_path)
            except ValueError as e:
                logger.error(f"{e}")
                kc = None
        else:
            logger.info(f"Framix Analyzer: 基础模式 ...")
            kc = None

        for video in self.only_video(folder):
            reporter.title = video.title
            for path in video.sheet:
                reporter.query = os.path.basename(path).split(".")[0]
                shutil.copy(path, reporter.video_path)
                new_video_path = os.path.join(reporter.video_path, os.path.basename(path))

                futures = looper.run_until_complete(
                    analyzer(
                        new_video_path, deploy, kc, reporter.frame_path, reporter.extra_path,
                        ffmpeg=self.ffmpeg, ffprobe=self.ffprobe
                    )
                )
                if futures is None:
                    continue
                start, end, cost, classifier = futures

                result = {
                    "total_path": Path(reporter.total_path).name,
                    "title": reporter.title,
                    "query": reporter.query,
                    "stage": {"start": start, "end": end, "cost": f"{cost:.5f}"},
                    "frame": Path(reporter.frame_path).name
                }

                if classifier:
                    original_inform = reporter.draw(
                        classifier_result=classifier,
                        proto_path=reporter.proto_path,
                        template_file=get_template(self.alien),
                    )
                    result["extra"] = Path(reporter.extra_path).name
                    result["proto"] = Path(original_inform).name

                logger.debug(f"Restore: {result}")
                reporter.load(result)

                with DataBase(os.path.join(reporter.reset_path, "Framix_Data.db")) as database:
                    if classifier:
                        column_list = [
                            'total_path', 'title', 'query_path', 'query', 'stage', 'frame_path', 'extra_path', 'proto_path'
                        ]
                        database.create('stocks', *column_list)
                        stage = {'stage': {'start': start, 'end': end, 'cost': cost}}
                        database.insert(
                            'stocks', column_list,
                            (reporter.total_path, reporter.title, reporter.query_path, reporter.query, json.dumps(stage),
                             reporter.frame_path, reporter.extra_path, reporter.proto_path)
                        )
                    else:
                        column_list = [
                            'total_path', 'title', 'query_path', 'query', 'stage', 'frame_path'
                        ]
                        database.create('stocks', *column_list)
                        stage = {'stage': {'start': start, 'end': end, 'cost': cost}}
                        database.insert(
                            'stocks', column_list,
                            (reporter.total_path, reporter.title, reporter.query_path, reporter.query, json.dumps(stage),
                             reporter.frame_path)
                        )

        looper.run_until_complete(
            reporter.ask_create_total_report(
                os.path.dirname(reporter.total_path),
                get_template(self.main_temp),
                get_template(self.main_total_temp),
                deploy.group
            )
        )
        return reporter.total_path

    def train_model(self, video_file: str):
        if not os.path.isfile(video_file):
            logger.error(f"{video_file} 视频文件未找到 ...")
            return
        logger.info(f"视频文件 {video_file} ...")

        screen = cv2.VideoCapture(video_file)
        if not screen.isOpened():
            logger.error(f"{video_file} 视频文件损坏 ...")
            return
        screen.release()
        logger.info(f"{video_file} 可正常播放 ...")

        reporter = Report(total_path=self.initial_report)
        reporter.title = f"Model_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}"
        if not os.path.exists(reporter.query_path):
            os.makedirs(reporter.query_path, exist_ok=True)

        deploy = Deploy(self.initial_deploy)
        deploy.alone = self.alone
        deploy.group = self.group
        deploy.quick = self.quick
        deploy.basic = self.basic
        deploy.keras = self.keras

        deploy.boost = self.boost
        deploy.color = self.color

        deploy.shape = self.shape if "--shape" in self.lines else deploy.shape
        deploy.scale = self.scale if "--scale" in self.lines else deploy.scale
        deploy.start = self.start if "--start" in self.lines else deploy.start
        deploy.close = self.close if "--close" in self.lines else deploy.close
        deploy.limit = self.limit if "--limit" in self.lines else deploy.limit
        deploy.begin = self.begin if "--begin" in self.lines else deploy.begin
        deploy.final = self.final if "--final" in self.lines else deploy.final

        deploy.crops = self.crops if "--crops" in self.lines else deploy.crops
        deploy.omits = self.omits if "--omits" in self.lines else deploy.omits

        video_temp_file = os.path.join(
            reporter.query_path, f"tmp_fps{deploy.fps}.mp4"
        )

        looper = asyncio.get_event_loop()
        duration = looper.run_until_complete(
            ask_video_length(self.ffprobe, video_file)
        )
        vision_start, vision_close, vision_limit = examine_flip(
            deploy.parse_mills(deploy.start),
            deploy.parse_mills(deploy.close),
            deploy.parse_mills(deploy.limit),
            duration
        )
        vision_start = deploy.parse_times(vision_start)
        vision_close = deploy.parse_times(vision_close)
        vision_limit = deploy.parse_times(vision_limit)
        logger.info(f"视频时长: [{duration}] [{deploy.parse_times(duration)}]")
        logger.info(f"start=[{vision_start}] - close=[{vision_close}] - limit=[{vision_limit}]")

        asyncio.run(
            ask_video_change(
                self.ffmpeg, deploy.fps, video_file, video_temp_file,
                start=vision_start, close=vision_close, limit=vision_limit
            )
        )

        video = VideoObject(video_temp_file)
        video.load_frames(
            silently_load_hued=False,
            not_transform_gray=True
        )

        cutter = VideoCutter()
        res = cutter.cut(
            video=video,
            block=deploy.block
        )
        stable, unstable = res.get_range(
            threshold=deploy.threshold,
            offset=deploy.offset
        )

        if deploy.shape:
            original_shape = looper.run_until_complete(
                ask_video_larger(self.ffprobe, video_file)
            )
            w, h, ratio = looper.run_until_complete(
                ask_magic_frame(original_shape, deploy.shape)
            )
            target_shape = w, h
            target_scale = deploy.scale
            logger.info(f"调整宽高比: {w} x {h}")
        elif deploy.scale:
            target_shape = deploy.shape
            target_scale = max(0.1, min(1.0, deploy.scale))
        else:
            target_shape = deploy.shape
            target_scale = self.COMPRESS

        res.pick_and_save(
            range_list=stable,
            frame_count=20,
            to_dir=reporter.query_path,
            meaningful_name=True,
            not_grey=deploy.color,
            compress_rate=target_scale,
            target_size=target_shape
        )

        os.remove(video_temp_file)

    def build_model(self, src: str):
        if not os.path.isdir(src):
            logger.error("训练模型需要一个分类文件夹 ...")
            return

        real_path, file_list = "", []
        logger.debug(f"搜索文件夹: {src}")
        for root, dirs, files in os.walk(src, topdown=False):
            for name in files:
                file_list.append(os.path.join(root, name))
            for name in dirs:
                if len(name) == 1 and re.search(r"0", name):
                    real_path = os.path.join(root, name)
                    logger.debug(f"分类文件夹: {real_path}")
                    break

        if not real_path or len(file_list) == 0:
            logger.error(f"文件夹未正确分类 ...")
            return

        image_color, image_aisle = "grayscale", 1
        for image_file in os.listdir(real_path):
            image_path = os.path.join(real_path, image_file)
            if not os.path.isfile(image_path):
                logger.error(f"存在无效的图像文件 ...")
                return
            image = cv2.imread(image_path)
            logger.info(f"图像分辨率: {image.shape}")
            if image.ndim == 3:
                if np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(image[:, :, 1], image[:, :, 2]):
                    logger.info("The image is grayscale image, stored in RGB format ...")
                else:
                    logger.info("The image is color image ...")
                    image_color = "rgb"
                    image_aisle = image.ndim
            else:
                logger.info("The image is grayscale image ...")
            break

        final_path = os.path.dirname(real_path)
        new_model_path = os.path.join(final_path, f"Create_Model_{time.strftime('%Y%m%d%H%M%S')}")
        new_model_name = f"Keras_Model_{random.randint(10000, 99999)}.h5"
        fc = FramixClassifier(color=image_color, aisle=image_aisle, data_size=self.shape)
        fc.build(final_path, new_model_path, new_model_name)

    async def combines_main(self, merge: list, group: bool):
        major, total = await asyncio.gather(
            ask_get_template(self.main_temp), ask_get_template(self.main_total_temp),
            return_exceptions=True
        )
        tasks = [
            Report.ask_create_total_report(m, major, total, group) for m in merge
        ]
        error = await asyncio.gather(*tasks)
        for e in error:
            if isinstance(e, Exception):
                logger.error(e)

    async def combines_view(self, merge: list, group: bool):
        views, total = await asyncio.gather(
            ask_get_template(self.view_temp), ask_get_template(self.view_total_temp),
            return_exceptions=True
        )
        tasks = [
            Report.ask_invent_total_report(m, views, total, group) for m in merge
        ]
        error = await asyncio.gather(*tasks)
        for e in error:
            if isinstance(e, Exception):
                logger.error(e)

    async def painting(self, deploy: Deploy):
        import tempfile
        from PIL import Image, ImageDraw, ImageFont

        async def paint_lines(serial):
            image_folder = "/sdcard/Pictures/Shots"
            image = f"{time.strftime('%Y%m%d%H%M%S')}_{random.randint(100, 999)}_" + "Shot.png"
            await Terminal.cmd_line(
                self.adb, "-s", serial, "wait-for-device"
            )
            await Terminal.cmd_line(
                self.adb, "-s", serial, "shell", "mkdir", "-p", image_folder
            )
            await Terminal.cmd_line(
                self.adb, "-s", serial, "shell", "screencap", "-p", f"{image_folder}/{image}"
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                image_save_path = os.path.join(temp_dir, image)
                await Terminal.cmd_line(
                    self.adb, "-s", serial, "pull", f"{image_folder}/{image}", image_save_path
                )

                if deploy.color:
                    old_image = toolbox.imread(image_save_path)
                    new_image = VideoFrame(0, 0, old_image)
                else:
                    old_image = toolbox.imread(image_save_path)
                    old_image = toolbox.turn_grey(old_image)
                    new_image = VideoFrame(0, 0, old_image)

                if len(deploy.crops) > 0:
                    for crop in deploy.crops:
                        if len(crop) == 4 and sum(crop) > 0:
                            x, y, x_size, y_size = crop
                        paint_crop_hook = PaintCropHook((y_size, x_size), (y, x))
                        paint_crop_hook.do(new_image)

                if len(deploy.omits) > 0:
                    for omit in deploy.omits:
                        if len(omit) == 4 and sum(omit) > 0:
                            x, y, x_size, y_size = omit
                            paint_omit_hook = PaintOmitHook((y_size, x_size), (y, x))
                            paint_omit_hook.do(new_image)

                cv2.imencode(".png", new_image.data)[1].tofile(image_save_path)

                image_file = Image.open(image_save_path)
                image_file = image_file.convert("RGB")

                original_w, original_h = image_file.size
                if deploy.shape:
                    twist_w, twist_h, _ = await ask_magic_frame(image_file.size, deploy.shape)
                else:
                    twist_w, twist_h = original_w, original_h

                min_scale, max_scale = 0.1, 1.0
                if deploy.scale:
                    image_scale = max_scale if deploy.shape else max(min_scale, min(max_scale, deploy.scale))
                else:
                    image_scale = max_scale if deploy.shape else self.COMPRESS

                new_w, new_h = int(twist_w * image_scale), int(twist_h * image_scale)
                logger.debug(
                    f"原始尺寸: {(original_w, original_h)} 调整尺寸: {(new_w, new_h)} 缩放比例: {int(image_scale * 100)}%"
                )

                if new_w == new_h:
                    x_line_num, y_line_num = 10, 10
                elif new_w > new_h:
                    x_line_num, y_line_num = 10, 20
                else:
                    x_line_num, y_line_num = 20, 10

                resized = image_file.resize((new_w, new_h))

                draw = ImageDraw.Draw(resized)
                font = ImageFont.load_default()

                if y_line_num > 0:
                    for i in range(1, y_line_num):
                        x_line = int(new_w * (i * (1 / y_line_num)))
                        text = f"{i * int(100 / y_line_num):02}"
                        bbox = draw.textbbox((0, 0), text, font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        y_text_start = 3
                        draw.line(
                            [(x_line, text_width + 5 + y_text_start), (x_line, new_h)],
                            fill=(0, 255, 255), width=1
                        )
                        draw.text(
                            (x_line - text_height // 2, y_text_start),
                            text, fill=(0, 255, 255), font=font
                        )

                if x_line_num > 0:
                    for i in range(1, x_line_num):
                        y_line = int(new_h * (i * (1 / x_line_num)))
                        text = f"{i * int(100 / x_line_num):02}"
                        bbox = draw.textbbox((0, 0), text, font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        x_text_start = 3
                        draw.line(
                            [(text_width + 5 + x_text_start, y_line), (new_w, y_line)],
                            fill=(255, 182, 193), width=1
                        )
                        draw.text(
                            (x_text_start, y_line - text_height // 2),
                            text, fill=(255, 182, 193), font=font
                        )

                resized.show()

            await Terminal.cmd_line(
                self.adb, "-s", serial, "shell", "rm", f"{image_folder}/{image}"
            )
            return resized

        manage = Manage(self.adb)
        device_list = await manage.operate_device()
        tasks = [paint_lines(device.serial) for device in device_list]
        resized_result = await asyncio.gather(*tasks)

        while True:
            action = Prompt.ask(
                f"[bold]保存图片([bold #5fd700]Y[/bold #5fd700]/[bold #ff87af]N[/bold #ff87af])?[/bold]",
                console=Show.console, default="Y"
            )
            if action.strip().upper() == "Y":
                reporter = Report(self.initial_report)
                reporter.title = f"Hooks_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
                for device, resize_img in zip(device_list, resized_result):
                    img_save_path = os.path.join(
                        reporter.query_path, f"hook_{device.serial}_{random.randint(10000, 99999)}.png"
                    )
                    resize_img.save(img_save_path)
                    Show.console.print(f"[bold]保存图片: {[img_save_path]}")
                break
            elif action.strip().upper() == "N":
                break
            else:
                Show.console.print(f"[bold][bold red]没有该选项,请重新输入[/bold red] ...[/bold]\n")

    async def analysis(self, deploy: Deploy):

        device_events = {}
        all_stop_event = asyncio.Event()

        # Time
        async def timepiece(amount, serial, events):
            stop_event_control = events["stop_event"] if deploy.alone else all_stop_event
            while True:
                if events["head_event"].is_set():
                    for i in range(amount):
                        if stop_event_control.is_set() and i != amount:
                            logger.success(f"{serial} 主动停止 ...")
                            logger.warning(f"{serial} 剩余时间 -> 00 秒")
                            return
                        elif events["fail_event"].is_set():
                            logger.error(f"{serial} 意外停止 ...")
                            logger.warning(f"{serial} 剩余时间 -> 00 秒")
                            return
                        if amount - i <= 10:
                            logger.warning(f"{serial} 剩余时间 -> {amount - i:02} 秒 {'----' * (amount - i)}")
                        else:
                            logger.warning(f"{serial} 剩余时间 -> {amount - i:02} 秒 {'----' * 10} ...")
                        await asyncio.sleep(1)
                    logger.warning(f"{serial} 剩余时间 -> 00 秒")
                    return
                elif events["fail_event"].is_set():
                    logger.error(f"{serial} 意外停止 ...")
                    break
                await asyncio.sleep(0.2)

        # Screen Copy
        async def start_record(serial: str, dst: str, events):

            # Stream
            async def input_stream():
                async for line in transports.stdout:
                    logger.info(stream := line.decode(encoding="UTF-8", errors="ignore").strip())
                    if "Recording started" in stream:
                        events["head_event"].set()
                    elif "Recording complete" in stream:
                        stop_event_control.set()
                        events["done_event"].set()
                        break

            # Stream
            async def error_stream():
                async for line in transports.stderr:
                    logger.info(stream := line.decode(encoding="UTF-8", errors="ignore").strip())
                    if "Could not find" in stream or "connection failed" in stream or "Recorder error" in stream:
                        events["fail_event"].set()
                        break

            stop_event_control = events["stop_event"] if deploy.alone else all_stop_event
            temp_video = f"{os.path.join(dst, 'screen')}_{time.strftime('%Y%m%d%H%M%S')}_{random.randint(100, 999)}.mkv"
            cmd = [
                self.scrcpy, "-s", serial, "--no-audio", "--video-bit-rate", "8M", "--max-fps", "60", "--record", temp_video
            ]
            transports = await Terminal.cmd_link(*cmd)
            asyncio.create_task(input_stream())
            asyncio.create_task(error_stream())
            await asyncio.sleep(1)
            return temp_video, transports

        # Screen Copy
        async def stop_record(temp_video, transports, events):
            if operation_system == "win32":
                await Terminal.cmd_line("taskkill", "/im", "scrcpy.exe")
            else:
                transports.terminate()
                await transports.wait()

            for _ in range(10):
                if events["done_event"].is_set():
                    logger.success(f"视频录制成功: {os.path.basename(temp_video)}")
                    return True
                elif events["fail_event"].is_set():
                    logger.error(f"视频录制失败: {os.path.basename(temp_video)}")
                    return False
                await asyncio.sleep(0.2)

        # Video_Balance
        async def video_balance(standard, duration, video_src):
            start_time_point = duration - standard
            end_time_point = duration
            start_time_str = str(datetime.timedelta(seconds=start_time_point))
            end_time_str = str(datetime.timedelta(seconds=end_time_point))

            logger.info(f"{os.path.basename(video_src)} {duration} [{start_time_str} - {end_time_str}]")
            video_dst = os.path.join(
                os.path.dirname(video_src), f"tailor_fps{deploy.fps}_{random.randint(100, 999)}.mp4"
            )

            await ask_video_tailor(
                self.ffmpeg, video_src, video_dst, start=start_time_str, limit=end_time_str
            )
            return video_dst

        # Record
        async def commence():

            # Wait Device Online
            async def wait_for_device(serial):
                logger.info(f"wait-for-device {serial} ...")
                await Terminal.cmd_line(self.adb, "-s", serial, "wait-for-device")

            await asyncio.gather(
                *(wait_for_device(device.serial) for device in device_list)
            )

            todo_list = []

            if deploy.quick or deploy.basic or deploy.keras:
                group_fmt_dirs = reporter.clock()
                for device in device_list:
                    await asyncio.sleep(0.2)
                    device_events[device.serial] = {
                        "head_event": asyncio.Event(), "done_event": asyncio.Event(),
                        "stop_event": asyncio.Event(), "fail_event": asyncio.Event()
                    }
                    reporter.query = os.path.join(group_fmt_dirs, device.serial)
                    temp_video, transports = await start_record(
                        device.serial, reporter.video_path, device_events[device.serial]
                    )
                    todo_list.append(
                        [temp_video, transports, reporter.total_path, reporter.title, reporter.query_path,
                         reporter.query, reporter.frame_path, reporter.extra_path, reporter.proto_path]
                    )

                await asyncio.gather(
                    *(timepiece(timer_mode, serial, events) for serial, events in device_events.items())
                )
                effective_list = await asyncio.gather(
                    *(stop_record(temp_video, transports, events)
                      for (_, events), (temp_video, transports, *_) in zip(device_events.items(), todo_list))
                )
                for idx, effective in enumerate(effective_list):
                    if not effective:
                        todo_list.pop(idx)

            else:
                for device in device_list:
                    await asyncio.sleep(0.2)
                    device_events[device.serial] = {
                        "head_event": asyncio.Event(), "done_event": asyncio.Event(),
                        "stop_event": asyncio.Event(), "fail_event": asyncio.Event()
                    }
                    temp_video, transports = await start_record(
                        device.serial, reporter.query_path, device_events[device.serial]
                    )
                    todo_list.append(
                        [temp_video, transports, reporter.total_path, reporter.title, reporter.query_path,
                         reporter.query_path, reporter.frame_path, reporter.extra_path, reporter.proto_path]
                    )

                await asyncio.gather(
                    *(timepiece(timer_mode, serial, events)
                      for serial, events in device_events.items())
                )
                effective_list = await asyncio.gather(
                    *(stop_record(temp_video, transports, events)
                      for (_, events), (temp_video, transports, *_) in zip(device_events.items(), todo_list))
                )
                for idx, effective in enumerate(effective_list):
                    if not effective:
                        todo_list.pop(idx)

            if not deploy.alone and len(todo_list) > 1:
                duration_list = await asyncio.gather(
                    *(ask_video_length(self.ffprobe, temp_video) for temp_video, *_ in todo_list)
                )
                duration_list = [duration for duration in duration_list if not isinstance(duration, Exception)]
                if len(duration_list) == 0:
                    todo_list.clear()
                    return todo_list

                standard = min(duration_list)
                logger.info(f"标准录制时间: {standard}")
                balance_task = [
                    video_balance(standard, duration, video_src)
                    for duration, (video_src, *_) in zip(duration_list, todo_list)
                ]
                video_dst_list = await asyncio.gather(*balance_task)
                for idx, dst in enumerate(video_dst_list):
                    todo_list[idx][0] = dst

            return todo_list

        # Analysis Mode
        async def analysis_tactics():
            if len(task_list) == 0:
                return False

            # 快速模式
            if deploy.quick:
                logger.debug(f"Framix Analyzer: 快速模式 ...")
                video_filter_list = []
                default_filter = [f"fps={deploy.fps}"] if deploy.color else [f"fps={deploy.fps}", "format=gray"]
                if deploy.shape:
                    original_shape_list = await asyncio.gather(
                        *(ask_video_larger(self.ffprobe, temp_video) for temp_video, *_ in task_list)
                    )
                    final_shape_list = await asyncio.gather(
                        *(ask_magic_frame(original_shape, deploy.shape) for original_shape in original_shape_list)
                    )
                    for final_shape in final_shape_list:
                        video_filter = default_filter.copy()
                        w, h, ratio = final_shape
                        video_filter.append(f"scale={w}:{h}")
                        video_filter_list.append(video_filter)
                elif deploy.scale:
                    for temp_video, *_ in task_list:
                        video_filter = default_filter.copy()
                        scale = max(0.1, min(1.0, deploy.scale))
                        video_filter.append(f"scale=iw*{scale}:ih*{scale}")
                        video_filter_list.append(video_filter)
                else:
                    for temp_video, *_ in task_list:
                        video_filter = default_filter.copy()
                        video_filter.append(f"scale=iw*{self.COMPRESS}:ih*{self.COMPRESS}")
                        video_filter_list.append(video_filter)

                for filters in video_filter_list:
                    logger.info(f"应用过滤器: {filters}")

                duration_list = await asyncio.gather(
                    *(ask_video_length(self.ffprobe, temp_video) for temp_video, *_ in task_list)
                )
                looper = asyncio.get_event_loop()
                duration_result = [looper.run_in_executor(
                    None, examine_flip,
                    deploy.parse_mills(deploy.start), deploy.parse_mills(deploy.close),
                    deploy.parse_mills(deploy.limit), duration
                ) for duration in duration_list]
                duration_result_list = await asyncio.gather(*duration_result)

                all_duration = []
                for (vision_start, vision_close, vision_limit), duration in zip(duration_result_list, duration_list):
                    all_duration.append(
                        (
                            vision_start := deploy.parse_times(vision_start),
                            vision_close := deploy.parse_times(vision_close),
                            vision_limit := deploy.parse_times(vision_limit)
                        )
                    )
                    logger.info(f"视频时长: [{duration}] [{deploy.parse_times(duration)}]")
                    logger.info(f"start=[{vision_start}] - close=[{vision_close}] - limit=[{vision_limit}]")

                await asyncio.gather(
                    *(ask_video_detach(
                        self.ffmpeg, video_filter, temp_video, frame_path,
                        start=vision_start, close=vision_close, limit=vision_limit
                    )
                      for (
                        temp_video, *_, frame_path, _, _
                    ), video_filter, (
                        vision_start, vision_close, vision_limit
                    ) in zip(
                        task_list, video_filter_list, duration_result_list
                    ))
                )
                for *_, total_path, title, query_path, query, frame_path, _, _ in task_list:
                    result = {
                        "total_path": Path(total_path).name,
                        "title": title,
                        "query": query,
                        "stage": {"start": 0, "end": 0, "cost": 0},
                        "frame": Path(frame_path).name
                    }
                    logger.debug(f"Quick: {result}")
                    reporter.load(result)

            # 基础模式 or 智能模式
            elif deploy.basic or deploy.keras:
                futures = await asyncio.gather(
                    *(analyzer(
                        temp_video, deploy, kc, frame_path, extra_path,
                        ffmpeg=self.ffmpeg, ffprobe=self.ffprobe
                    ) for temp_video, *_, frame_path, extra_path, _ in task_list)
                )

                for future, todo in zip(futures, task_list):
                    if future is None:
                        continue

                    start, end, cost, classifier = future
                    *_, total_path, title, query_path, query, frame_path, extra_path, proto_path = todo

                    result = {
                        "total_path": Path(total_path).name,
                        "title": title,
                        "query": query,
                        "stage": {"start": start, "end": end, "cost": f"{cost:.5f}"},
                        "frame": Path(frame_path).name
                    }

                    if classifier:
                        logger.debug(f"Framix Analyzer: 智能模式 ...")
                        template_file = await ask_get_template(self.alien)
                        original_inform = reporter.draw(
                            classifier_result=classifier,
                            proto_path=proto_path,
                            template_file=template_file,
                        )
                        result["extra"] = Path(extra_path).name
                        result["proto"] = Path(original_inform).name
                    else:
                        logger.debug(f"Framix Analyzer: 基础模式 ...")

                    logger.debug(f"Restore: {result}")
                    reporter.load(result)

                    with DataBase(os.path.join(os.path.dirname(total_path), "Nexa_Recovery", "Framix_Data.db")) as database:
                        if classifier:
                            column_list = [
                                'total_path', 'title', 'query_path', 'query', 'stage', 'frame_path', 'extra_path', 'proto_path'
                            ]
                            database.create('stocks', *column_list)
                            stage = {'stage': {'start': start, 'end': end, 'cost': cost}}
                            database.insert(
                                'stocks', column_list,
                                (total_path, title, query_path, query, json.dumps(stage), frame_path, extra_path, proto_path)
                            )
                        else:
                            column_list = [
                                'total_path', 'title', 'query_path', 'query', 'stage', 'frame_path'
                            ]
                            database.create('stocks', *column_list)
                            stage = {'stage': {'start': start, 'end': end, 'cost': cost}}
                            database.insert(
                                'stocks', column_list,
                                (total_path, title, query_path, query, json.dumps(stage), frame_path)
                            )

            # 录制模式
            else:
                logger.debug(f"Framix Analyzer: 录制模式 ...")
                return False

        # Device View
        async def device_mode_view():
            if len(device_list) == 1:
                Show.console.print(f"[bold]<Link> <单设备模式>")
            else:
                Show.console.print(f"[bold]<Link> <多设备模式>")
            for device in device_list:
                Show.console.print(f"[bold #00FFAF]Connect:[/bold #00FFAF] {device}")

        # Initialization ===============================================================================================
        manage = Manage(self.adb)
        device_list = await manage.operate_device()

        reporter = Report(self.initial_report)
        const_title = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        if deploy.quick:
            input_title = "Quick"
        elif deploy.basic:
            input_title = "Basic"
        elif deploy.keras:
            input_title = "Keras"
        else:
            input_title = "Video"
        reporter.title = f"{input_title}_{const_title}"

        if deploy.keras and not deploy.quick and not deploy.basic:
            kc = KerasClassifier(data_size=deploy.model_size)
            try:
                kc.load_model(self.model_path)
            except ValueError as e:
                logger.error(f"{e}")
                kc = None
        else:
            kc = None
        # Initialization ===============================================================================================

        # Loop =========================================================================================================
        timer_mode = 5
        while True:
            try:
                await device_mode_view()
                if action := Prompt.ask(
                        prompt=f"[bold #5FD7FF]<<<按 Enter 开始 [bold #D7FF5F]{timer_mode}[/bold #D7FF5F] 秒>>>[/bold #5FD7FF]",
                        console=Show.console):
                    select = action.strip().lower()
                    if "header" in select:
                        if match := re.search(r"(?<=header\s).*", select):
                            if match.group().strip():
                                src_hd = f"{input_title}_{time.strftime('%Y%m%d_%H%M%S')}"
                                if hd := match.group().strip():
                                    new_hd = f"{src_hd}_{hd}"
                                else:
                                    new_hd = f"{src_hd}_{random.randint(10000, 99999)}"
                                logger.success("新标题设置成功 ...")
                                reporter.title = new_hd
                                continue
                        Show.tips_document()
                        continue
                    elif select == "serial":
                        device_list = await manage.operate_device()
                        continue
                    elif select == "create" or select == "invent":
                        if len(reporter.range_list) > 0:
                            try:
                                reporter.range_list[0]["proto"]
                            except KeyError:
                                logger.debug(f"View Report Combines ...")
                                await self.combines_view([os.path.dirname(reporter.total_path)], deploy.group)
                            else:
                                logger.debug(f"Main Report Combines ...")
                                await self.combines_main([os.path.dirname(reporter.total_path)], deploy.group)
                            finally:
                                break
                        Show.console.print(f"[bold red]没有可以生成的报告 ...\n")
                        continue
                    elif select == "deploy":
                        Show.console.print("[bold yellow]修改 deploy.json 文件后请完全退出编辑器进程再继续操作 ...")
                        deploy.dump_deploy(self.initial_deploy)
                        if operation_system == "win32":
                            await Terminal.cmd_line("Notepad", self.initial_deploy)
                        else:
                            await Terminal.cmd_line("open", "-W", "-a", "TextEdit", self.initial_deploy)
                        deploy.load_deploy(self.initial_deploy)
                        deploy.view_deploy()
                        continue
                    elif select.isdigit():
                        value, lower_bound, upper_bound = int(select), 5, 300
                        if value > 300 or value < 5:
                            Show.console.print(
                                f"[bold #FFFF87]{lower_bound} <= [bold #FFD7AF]Time[/bold #FFD7AF] <= {upper_bound}[/bold #FFFF87]"
                            )
                        timer_mode = max(lower_bound, min(upper_bound, value))
                    else:
                        Show.tips_document()
                        continue
            except ValueError:
                Show.tips_document()
                continue
            else:
                task_list = await commence()
                await analysis_tactics()
                for _, event in device_events.items():
                    if event["fail_event"].is_set():
                        device_list = await manage.operate_device()
                        break
            finally:
                all_stop_event.clear()
                for _, event in device_events.items():
                    event["head_event"].clear()
                    event["done_event"].clear()
                    event["stop_event"].clear()
                    event["fail_event"].clear()
                device_events.clear()
        # Loop =========================================================================================================


def initializer(log_level: str) -> None:
    logger.remove(0)
    log_format = "| <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sys.stderr, format=log_format, level=log_level.upper())


def get_template(template_path: str) -> str | Exception:
    try:
        with open(template_path, mode="r", encoding="utf-8") as f:
            template_file = f.read()
    except FileNotFoundError as e:
        return e
    return template_file


def examine_flip(
        start: int | float, close: int | float, limit: int | float, duration: int | float
) -> tuple[Optional[int | float], Optional[int | float], Optional[int | float]]:
    """
    校验视频剪辑参数
    :param start: 开始时间
    :param close: 结束时间
    :param limit: 持续时间
    :param duration: 视频时间
    :return: 校验结果
    """
    start_point = close_point = limit_point = None
    if start:
        if start == duration:
            return start_point, close_point, limit_point
        start_point = max(0, min(start, duration))

    if close:
        close_point = max(start_point, min(close, duration)) if start else max(0, min(close, duration))
    elif limit:
        limit_point = max(0, min(limit, duration - start)) if start else max(0, min(limit, duration))

    return start_point, close_point, limit_point


async def ask_get_template(template_path: str) -> str | Exception:
    try:
        async with aiofiles.open(template_path, mode="r", encoding="utf-8") as f:
            template_file = await f.read()
    except FileNotFoundError as e:
        return e
    return template_file


async def ask_video_change(ffmpeg, fps: int, src: str, dst: str, **kwargs) -> None:
    """
    转换视频的帧采样率
    :param ffmpeg: ffmpeg 可执行文件
    :param fps: 帧采样率
    :param src: 输入文件
    :param dst: 输出文件
    :param kwargs: [start 开始时间] [close 结束时间] [limit 持续时间]
    :return:
    """
    start = kwargs.get("start", None)
    close = kwargs.get("close", None)
    limit = kwargs.get("limit", None)

    cmd = [ffmpeg]

    if start:
        cmd += ["-ss", start]
    if close:
        cmd += ["-to", close]
    elif limit:
        cmd += ["-t", limit]
    cmd += ["-i", src]
    cmd += ["-vf", f"fps={fps}", "-c:v", "libx264", "-crf", "18", "-c:a", "copy", dst]

    await Terminal.cmd_line(*cmd)


async def ask_video_detach(ffmpeg, video_filter: list, src: str, dst: str, **kwargs) -> None:
    start = kwargs.get("start", None)
    close = kwargs.get("close", None)
    limit = kwargs.get("limit", None)

    cmd = [ffmpeg]

    if start:
        cmd += ["-ss", start]
    if close:
        cmd += ["-to", close]
    elif limit:
        cmd += ["-t", limit]
    cmd += ["-i", src]
    cmd += ["-vf", ",".join(video_filter), f"{os.path.join(dst, 'frame_%05d.png')}"]

    await Terminal.cmd_line(*cmd)


async def ask_video_tailor(ffmpeg, src: str, dst: str, **kwargs) -> None:
    start = kwargs.get("start", None)
    close = kwargs.get("close", None)
    limit = kwargs.get("limit", None)

    cmd = [ffmpeg]

    if start:
        cmd += ["-ss", start]
    if close:
        cmd += ["-to", close]
    elif limit:
        cmd += ["-t", limit]
    cmd += ["-i", src]
    cmd += ["-c", "copy", dst]

    await Terminal.cmd_line(*cmd)


async def ask_video_length(ffprobe, src: str) -> float | Exception:
    cmd = [
        ffprobe, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", "-i", src
    ]
    result = await Terminal.cmd_line(*cmd)
    try:
        fmt_result = float(result.strip())
    except ValueError as e:
        return e
    return fmt_result


async def ask_video_larger(ffprobe, src: str) -> tuple[int | Exception, int | Exception]:
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height", "-of", "default=noprint_wrappers=1", src
    ]
    result = await Terminal.cmd_line(*cmd)
    match_w = re.search(r"(?<=width=)\d+", result)
    match_h = re.search(r"(?<=height=)\d+", result)
    try:
        w, h = int(match_w.group()), int(match_h.group())
    except (AttributeError, ValueError) as e:
        return e
    return w, h


async def ask_magic_frame(
        original_frame_size: tuple, input_frame_size: tuple
) -> tuple[int, int, float]:

    # 计算原始宽高比
    original_w, original_h = original_frame_size
    original_ratio = original_w / original_h

    if original_frame_size == input_frame_size:
        return original_w, original_h, original_ratio

    # 检查并调整传入的最大宽度和高度的限制
    frame_w, frame_h = input_frame_size
    max_w = max(original_w * 0.1, min(frame_w, original_w))
    max_h = max(original_h * 0.1, min(frame_h, original_h))

    # 根据原始宽高比调整宽度和高度
    if max_w / max_h > original_ratio:
        # 如果调整后的宽高比大于原始宽高比，则以高度为基准调整宽度
        adjusted_h = max_h
        adjusted_w = adjusted_h * original_ratio
    else:
        # 否则，以宽度为基准调整高度
        adjusted_w = max_w
        adjusted_h = adjusted_w / original_ratio

    new_w, new_h = int(adjusted_w), int(adjusted_h)
    return new_w, new_h, original_ratio


async def analyzer(
        vision_path: str, deploy: "Deploy", kc: "KerasClassifier", *args, **kwargs
) -> Optional[tuple[float, float, float, Optional[ClassifierResult]]]:

    frame_path, extra_path = args
    ffmpeg = kwargs["ffmpeg"]
    ffprobe = kwargs["ffprobe"]

    async def validate():
        screen_tag, screen_cap = None, None
        if os.path.isfile(vision_path):
            screen = cv2.VideoCapture(vision_path)
            if screen.isOpened():
                screen_tag = os.path.basename(vision_path)
                screen_cap = vision_path
            screen.release()
        elif os.path.isdir(vision_path):
            if len(
                    file_list := [
                        file for file in os.listdir(vision_path) if os.path.isfile(
                            os.path.join(vision_path, file)
                        )
                    ]
            ) > 1 or len(file_list) == 1:
                screen = cv2.VideoCapture(os.path.join(vision_path, file_list[0]))
                if screen.isOpened():
                    screen_tag = os.path.basename(file_list[0])
                    screen_cap = os.path.join(vision_path, file_list[0])
                screen.release()
        return screen_tag, screen_cap

    async def frame_flip():
        change_record = os.path.join(
            os.path.dirname(vision_path),
            f"screen_fps{deploy.fps}_{random.randint(100, 999)}.mp4"
        )

        duration = await ask_video_length(ffprobe, vision_path)
        looper = asyncio.get_event_loop()
        duration_result = looper.run_in_executor(
            None, examine_flip,
            deploy.parse_mills(deploy.start),
            deploy.parse_mills(deploy.close), deploy.parse_mills(deploy.limit),
            duration
        )
        vision_start, vision_close, vision_limit = await duration_result
        vision_start = deploy.parse_times(vision_start)
        vision_close = deploy.parse_times(vision_close)
        vision_limit = deploy.parse_times(vision_limit)
        logger.info(f"视频时长: [{duration}] [{deploy.parse_times(duration)}]")
        logger.info(f"start=[{vision_start}] - close=[{vision_close}] - limit=[{vision_limit}]")

        await ask_video_change(
            ffmpeg, deploy.fps, vision_path, change_record,
            start=vision_start, close=vision_close, limit=vision_limit
        )
        logger.info(f"视频转换完成: {os.path.basename(change_record)}")
        os.remove(vision_path)
        logger.info(f"移除旧的视频: {os.path.basename(vision_path)}")

        if deploy.shape:
            original_shape = await ask_video_larger(ffprobe, change_record)
            w, h, ratio = await ask_magic_frame(original_shape, deploy.shape)
            target_shape = w, h
            target_scale = deploy.scale
            logger.info(f"调整宽高比: {w} x {h}")
        elif deploy.scale:
            target_shape = deploy.shape
            target_scale = max(0.1, min(1.0, deploy.scale))
        else:
            target_shape = deploy.shape
            target_scale = 0.4

        video = VideoObject(change_record)
        task, hued = video.load_frames(
            silently_load_hued=deploy.color,
            not_transform_gray=False,
            shape=target_shape,
            scale=target_scale
        )
        return video, task, hued

    async def frame_flow():
        video, task, hued = await frame_flip()
        cutter = VideoCutter()

        compress_hook = CompressHook(
            compress_rate=1,
            target_size=None,
            not_grey=False
        )
        cutter.add_hook(compress_hook)

        if len(deploy.crops) > 0:
            for crop in deploy.crops:
                x, y, x_size, y_size = crop
                crop_hook = PaintCropHook((y_size, x_size), (y, x))
                cutter.add_hook(crop_hook)

        if len(deploy.omits) > 0:
            for omit in deploy.omits:
                x, y, x_size, y_size = omit
                omit_hook = PaintOmitHook((y_size, x_size), (y, x))
                cutter.add_hook(omit_hook)

        save_hook = FrameSaveHook(extra_path)
        cutter.add_hook(save_hook)

        res = cutter.cut(
            video=video,
            block=deploy.block
        )

        stable, unstable = res.get_range(
            threshold=deploy.threshold,
            offset=deploy.offset
        )

        files = os.listdir(extra_path)
        files.sort(key=lambda n: int(n.split("(")[0]))
        total_images, desired_count = len(files), 12
        # 如果总图片数不超过12张，则无需删除
        if total_images <= desired_count:
            retain_indices = range(total_images)
        else:
            # 计算每隔多少张图片保留一张
            interval = total_images / desired_count
            retain_indices = [int(i * interval) for i in range(desired_count)]
            # 为了确保恰好得到12张，调整最后一个索引
            if len(retain_indices) < desired_count:
                retain_indices.append(total_images - 1)
            elif len(retain_indices) > desired_count:
                retain_indices = retain_indices[:desired_count]
        # 删除未被保留的图片
        for index, file in enumerate(files):
            if index not in retain_indices:
                os.remove(os.path.join(extra_path, file))
        # 为保留下来的图片绘制分割线条
        for draw in os.listdir(extra_path):
            toolbox.draw_line(os.path.join(extra_path, draw))

        classify = kc.classify(
            video=video,
            valid_range=stable,
            keep_data=True
        )

        important_frames = classify.get_important_frame_list()

        pbar = toolbox.show_progress(classify.get_length(), 50, "Faster")
        frames_list = []
        if deploy.boost:
            frames_list.append(previous := important_frames[0])
            pbar.update(1)
            for current in important_frames[1:]:
                frames_list.append(current)
                pbar.update(1)
                frames_diff = current.frame_id - previous.frame_id
                if not previous.is_stable() and not current.is_stable() and frames_diff > 1:
                    for specially in classify.data[previous.frame_id: current.frame_id - 1]:
                        frames_list.append(specially)
                        pbar.update(1)
                previous = current
            pbar.close()
        else:
            for current in classify.data:
                frames_list.append(current)
                pbar.update(1)
            pbar.close()

        if deploy.color:
            video.hued_data = tuple(hued.result())
            logger.info(f"彩色帧已加载: {video.frame_details(video.hued_data)}")
            task.shutdown()
            frames = [video.hued_data[frame.frame_id - 1] for frame in frames_list]
        else:
            frames = [frame for frame in frames_list]

        return classify, frames

    async def frame_flick(classify):
        logger.info(f"阶段划分: {classify.get_ordered_stage_set()}")
        begin_stage, begin_frame = deploy.begin
        final_stage, final_frame = deploy.final
        try:
            start_frame = classify.get_not_stable_stage_range()[begin_stage][begin_frame]
            end_frame = classify.get_not_stable_stage_range()[final_stage][final_frame]
        except AssertionError as e:
            logger.error(f"{e}")
            start_frame = classify.get_important_frame_list()[0]
            end_frame = classify.get_important_frame_list()[-1]
            logger.warning(f"Framix Analyzer recalculate ...")
        except IndexError as e:
            logger.error(f"{e}")
            for i, unstable_stage in enumerate(classify.get_specific_stage_range("-3")):
                Show.console.print(f"[bold]第 {i:02} 个非稳定阶段:")
                Show.console.print(f"[bold]{'=' * 30}")
                for j, frame in enumerate(unstable_stage):
                    Show.console.print(f"[bold]第 {j:05} 帧: {frame}")
                Show.console.print(f"[bold]{'=' * 30}")
                Show.console.print(f"\n")
            start_frame = classify.get_important_frame_list()[0]
            end_frame = classify.get_important_frame_list()[-1]
            logger.warning(f"Framix Analyzer recalculate ...")

        if start_frame == end_frame:
            logger.warning(f"{start_frame} == {end_frame}")
            start_frame = classify.data[0]
            end_frame = classify.data[-1]
            logger.warning(f"Framix Analyzer recalculate ...")

        time_cost = end_frame.timestamp - start_frame.timestamp
        logger.info(
            f"图像分类结果: [开始帧: {start_frame.timestamp:.5f}] [结束帧: {end_frame.timestamp:.5f}] [总耗时: {time_cost:.5f}]"
        )
        return start_frame.frame_id, end_frame.frame_id, time_cost

    async def frame_forge(frame):
        try:
            short_timestamp = format(round(frame.timestamp, 5), ".5f")
            pic_name = f"{frame.frame_id}_{short_timestamp}.png"
            pic_path = os.path.join(frame_path, pic_name)
            _, codec = cv2.imencode(".png", frame.data)
            async with aiofiles.open(pic_path, "wb") as f:
                await f.write(codec.tobytes())
        except Exception as e:
            return e

    async def analytics_basic():
        video, task, hued = await frame_flip()

        if deploy.color:
            video.hued_data = tuple(hued.result())
            logger.info(f"彩色帧已加载: {video.frame_details(video.hued_data)}")
            task.shutdown()
            frames = [i for i in video.hued_data]
        else:
            frames = [i for i in video.grey_data]

        if operation_system == "win32":
            logger.debug(f"运行环境: {operation_system}")
            forge_result = await asyncio.gather(
                *(frame_forge(frame) for frame in frames),
                return_exceptions=True
            )
        else:
            logger.debug(f"运行环境: {operation_system}")
            tasks = [
                [frame_forge(frame) for frame in chunk]
                for chunk in
                [frames[i:i + 100] for i in range(0, len(frames), 100)]
            ]
            forge_list = []
            for task in tasks:
                task_result = await asyncio.gather(*task, return_exceptions=True)
                forge_list.extend(task_result)
            forge_result = tuple(forge_list)

        for result in forge_result:
            if isinstance(result, Exception):
                logger.error(f"Error: {result}")

        start_frame = frames[0]
        end_frame = frames[-1]

        time_cost = end_frame.timestamp - start_frame.timestamp
        return (start_frame.frame_id, end_frame.frame_id, time_cost), None

    async def analytics_keras():
        classify, frames = await frame_flow()

        if operation_system == "win32":
            logger.debug(f"运行环境: {operation_system}")
            flick_result, *forge_result = await asyncio.gather(
                frame_flick(classify), *(frame_forge(frame) for frame in frames),
                return_exceptions=True
            )
        else:
            logger.debug(f"运行环境: {operation_system}")
            tasks = [
                [frame_forge(frame) for frame in chunk]
                for chunk in
                [frames[i:i + 100] for i in range(0, len(frames), 100)]
            ]
            flick_task = asyncio.create_task(frame_flick(classify))
            forge_list = []
            for task in tasks:
                task_result = await asyncio.gather(*task, return_exceptions=True)
                forge_list.extend(task_result)
            forge_result = tuple(forge_list)
            flick_result = await flick_task

        for result in forge_result:
            if isinstance(result, Exception):
                logger.error(f"Error: {result}")

        return flick_result, classify

    tag, screen_record = await validate()
    if not tag or not screen_record:
        logger.error(f"{tag} 不是一个标准的mp4视频文件，或视频文件已损坏 ...")
        return None
    logger.info(f"{tag} 可正常播放，准备加载视频 ...")

    (start, end, cost), classifier = await analytics_keras() if kc else await analytics_basic()
    return start, end, cost, classifier


async def ask_main():
    deploy = Deploy(missions.initial_deploy)
    deploy.alone = missions.alone
    deploy.group = missions.group
    deploy.quick = missions.quick
    deploy.basic = missions.basic
    deploy.keras = missions.keras

    deploy.boost = missions.boost
    deploy.color = missions.color

    deploy.shape = missions.shape if "--shape" in missions.lines else deploy.shape
    deploy.scale = missions.scale if "--scale" in missions.lines else deploy.scale
    deploy.start = missions.start if "--start" in missions.lines else deploy.start
    deploy.close = missions.close if "--close" in missions.lines else deploy.close
    deploy.limit = missions.limit if "--limit" in missions.lines else deploy.limit
    deploy.begin = missions.begin if "--begin" in missions.lines else deploy.begin
    deploy.final = missions.final if "--final" in missions.lines else deploy.final

    deploy.crops = missions.crops if "--crops" in missions.lines else deploy.crops
    deploy.omits = missions.omits if "--omits" in missions.lines else deploy.omits

    if cmd_lines.flick:
        await missions.analysis(deploy)
    elif cmd_lines.paint:
        await missions.painting(deploy)
    elif cmd_lines.merge and len(cmd_lines.merge) > 0:
        await missions.combines_main(cmd_lines.merge, missions.group)
    elif cmd_lines.union and len(cmd_lines.union) > 0:
        await missions.combines_view(cmd_lines.union, missions.group)
    else:
        Show.help_document()


async def ask_test():
    pass


if __name__ == '__main__':
    if len(sys.argv) == 1:
        Show.help_document()
        sys.exit(0)

    lines = sys.argv[1:]

    from multiprocessing import Pool, freeze_support

    freeze_support()

    from argparse import ArgumentParser

    cmd_lines = Parser.parse_cmd()
    _level = "DEBUG" if cmd_lines.debug else "INFO"
    initializer(_level)

    # Debug Mode =======================================================================================================
    logger.debug(f"Level: {_level}")
    logger.debug(f"Lines: {lines}")

    logger.debug(f"System: {operation_system}")
    logger.debug(f"Worker: {work_platform}")

    logger.debug(f"Tools: {_tools_path}")
    logger.debug(f"Model: {_model_path}")
    logger.debug(f"Html-Template: {_main_total_temp}")
    logger.debug(f"Html-Template: {_main_temp}")
    logger.debug(f"Html-Template: {_view_total_temp}")
    logger.debug(f"Html-Template: {_view_temp}")
    logger.debug(f"Html-Template: {_alien}")

    logger.debug(f"adb: {_adb}")
    logger.debug(f"ffmpeg: {_ffmpeg}")
    logger.debug(f"ffprobe: {_ffprobe}")
    logger.debug(f"scrcpy: {_scrcpy}")

    for env in os.environ["PATH"].split(os.path.pathsep):
        logger.debug(env)
    # Debug Mode =======================================================================================================

    _alone = cmd_lines.alone
    _quick = cmd_lines.quick
    _basic = cmd_lines.basic
    _keras = cmd_lines.keras

    _group = cmd_lines.group

    _boost = cmd_lines.boost
    _color = cmd_lines.color

    _shape = cmd_lines.shape
    _scale = cmd_lines.scale

    _start = cmd_lines.start
    _close = cmd_lines.close
    _limit = cmd_lines.limit

    _begin = cmd_lines.begin
    _final = cmd_lines.final

    # Debug Mode =======================================================================================================
    cpu = os.cpu_count()
    logger.debug(f"CPU Core: {cpu}")
    # Debug Mode =======================================================================================================

    _crops = []
    if cmd_lines.crops and len(cmd_lines.crops) > 0:
        for hook in cmd_lines.crops:
            if len(match_list := re.findall(r"-?\d*\.?\d+", hook)) == 4:
                valid_list = [float(num) if "." in num else int(num) for num in match_list]
                if sum(valid_list) > 0:
                    _crops.append(tuple(valid_list))
    if len(_crops) >= 2:
        _crops = list(set(_crops))

    _omits = []
    if cmd_lines.omits and len(cmd_lines.omits) > 0:
        for hook in cmd_lines.omits:
            if len(match_list := re.findall(r"-?\d*\.?\d+", hook)) == 4:
                valid_list = [float(num) if "." in num else int(num) for num in match_list]
                if sum(valid_list) > 0:
                    _omits.append(tuple(valid_list))
    if len(_omits) >= 2:
        _omits = list(set(_omits))

    _initial_deploy = os.path.join(_initial_deploy, "deploy.json")
    _initial_option = os.path.join(_initial_option, "option.json")

    option = Option(_initial_option)
    _initial_report = option.total_path if option.total_path else _initial_report
    _model_path = os.path.join(
        _model_path, option.model_name
    ) if option.model_name else os.path.join(
        _model_path, "model.h5"
    )

    # Debug Mode =======================================================================================================
    logger.debug(f"Initial-Report: {_initial_report}")
    logger.debug(f"Initial-Deploy: {_initial_deploy}")
    logger.debug(f"Initial-Option: {_initial_option}")
    logger.debug(f"Model-Path: {_model_path}")
    # Debug Mode =======================================================================================================

    missions = Missions(
        _alone, _quick, _basic, _keras, _group,
        _boost, _color, _shape, _scale, _start, _close, _limit, _begin, _final, _crops, _omits,
        lines=lines, model_path=_model_path,
        main_total_temp=_main_total_temp, main_temp=_main_temp,
        view_total_temp=_view_total_temp, view_temp=_view_temp,
        alien=_alien,
        initial_report=_initial_report, initial_deploy=_initial_deploy, initial_option=_initial_option,
        adb=_adb, ffmpeg=_ffmpeg, ffprobe=_ffprobe, scrcpy=_scrcpy
    )

    if cmd_lines.stack and len(cmd_lines.stack) > 0:
        members = len(cmd_lines.stack)
        if members == 1:
            missions.video_dir_task(cmd_lines.stack[0])
        else:
            processes = members if members <= cpu else cpu
            with Pool(processes=processes, initializer=initializer, initargs=("ERROR",)) as pool:
                results = pool.starmap(missions.video_dir_task, [(i,) for i in cmd_lines.stack])
            template_total = get_template(
                missions.view_total_temp
            ) if missions.quick else get_template(missions.main_total_temp)
            Report.merge_report(results, template_total, missions.quick)
        sys.exit(0)
    elif cmd_lines.video and len(cmd_lines.video) > 0:
        members = len(cmd_lines.video)
        if members == 1:
            missions.video_task(cmd_lines.video[0])
        else:
            processes = members if members <= cpu else cpu
            with Pool(processes=processes, initializer=initializer, initargs=("ERROR",)) as pool:
                results = pool.starmap(missions.video_task, [(i,) for i in cmd_lines.video])
            template_total = get_template(
                missions.view_total_temp
            ) if missions.quick else get_template(missions.main_total_temp)
            Report.merge_report(results, template_total, missions.quick)
        sys.exit(0)
    elif cmd_lines.train and len(cmd_lines.train) > 0:
        members = len(cmd_lines.train)
        if members == 1:
            missions.train_model(cmd_lines.train[0])
        else:
            processes = members if members <= cpu else cpu
            with Pool(processes=processes, initializer=initializer, initargs=("ERROR",)) as pool:
                pool.starmap(missions.train_model, [(i,) for i in cmd_lines.train])
        sys.exit(0)
    elif cmd_lines.build and len(cmd_lines.build) > 0:
        members = len(cmd_lines.build)
        if members == 1:
            missions.build_model(cmd_lines.build[0])
        else:
            processes = members if members <= cpu else cpu
            with Pool(processes=processes, initializer=initializer, initargs=("ERROR",)) as pool:
                pool.starmap(missions.build_model, [(i,) for i in cmd_lines.build])
        sys.exit(0)
    else:
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(ask_main())
            sys.exit(0)
        except KeyboardInterrupt:
            sys.exit(0)
