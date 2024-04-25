__all__ = []

import os
import sys
import shutil
from loguru import logger
from frameflow.skills.show import Show
from nexaflow import const

_platform = sys.platform.strip().lower()
_software = os.path.basename(os.path.abspath(sys.argv[0])).strip().lower()
_sys_symbol = os.sep
_env_symbol = os.path.pathsep

if _software == f"{const.NAME}.exe":
    _workable = os.path.dirname(os.path.abspath(sys.argv[0]))
    _feasible = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
elif _software == f"{const.NAME}.bin":
    _workable = os.path.dirname(sys.executable)
    _feasible = os.path.dirname(os.path.dirname(sys.executable))
elif _software == f"{const.NAME}":
    _workable = os.path.dirname(sys.executable)
    _feasible = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))))
elif _software == f"{const.NAME}.py":
    _workable = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _feasible = os.path.dirname(os.path.abspath(__file__))
else:
    logger.error(f"Software compatible with [bold red]{const.NAME}[/bold red] ...")
    Show.simulation_progress(f"Exit after 5 seconds ...", 1, 0.05)
    sys.exit(Show.fail())

_turbo = os.path.join(_workable, "archivix", "tools")

if _platform == "win32":
    _adb = os.path.join(_turbo, "win", "platform-tools", "adb.exe")
    _fmp = os.path.join(_turbo, "win", "ffmpeg", "bin", "ffmpeg.exe")
    _fpb = os.path.join(_turbo, "win", "ffmpeg", "bin", "ffprobe.exe")
    _scc = os.path.join(_turbo, "win", "scrcpy", "scrcpy.exe")
elif _platform == "darwin":
    _adb = os.path.join(_turbo, "mac", "platform-tools", "adb")
    _fmp = os.path.join(_turbo, "mac", "ffmpeg", "bin", "ffmpeg")
    _fpb = os.path.join(_turbo, "mac", "ffmpeg", "bin", "ffprobe")
    _scc = os.path.join(_turbo, "mac", "scrcpy", "bin", "scrcpy")
else:
    logger.error(f"{const.NAME} compatible with [bold red]Win & Mac[/bold red] ...")
    Show.simulation_progress(f"Exit after 5 seconds ...", 1, 0.05)
    sys.exit(Show.fail())

for _tls in (_tools := [_adb, _fmp, _fpb, _scc]):
    os.environ["PATH"] = os.path.dirname(_tls) + _env_symbol + os.environ.get("PATH", "")

for _tls in _tools:
    if not shutil.which((_tls_name := os.path.basename(_tls))):
        logger.error(f"{const.NAME} missing files [bold red]{_tls_name}[/bold red] ...")
        Show.simulation_progress(f"Exit after 5 seconds ...", 1, 0.05)
        sys.exit(Show.fail())

_atom_total_temp = os.path.join(_workable, "archivix", "pages", "template_atom_total.html")
_main_share_temp = os.path.join(_workable, "archivix", "pages", "template_main_share.html")
_main_total_temp = os.path.join(_workable, "archivix", "pages", "template_main_total.html")
_view_share_temp = os.path.join(_workable, "archivix", "pages", "template_view_share.html")
_view_total_temp = os.path.join(_workable, "archivix", "pages", "template_view_total.html")

for _tmp in (_temps := [_atom_total_temp, _main_share_temp, _main_total_temp, _view_share_temp, _view_total_temp]):
    if not os.path.isfile(_tmp):
        _tmp_name = os.path.basename(_tmp)
        logger.error(f"{const.NAME} missing files [bold red]{_tmp_name}[/bold red] ...")
        Show.simulation_progress(f"Exit after 5 seconds ...", 1, 0.05)
        sys.exit(Show.fail())

_initial_source = os.path.join(_feasible, f"{const.NAME}.source")

_total_place = os.path.join(_feasible, f"{const.NAME}.report")
_model_place = os.path.join(_workable, "archivix", "molds", "Keras_Gray_W256_H256_00000.h5")

if len(sys.argv) == 1:
    Show.help_document()
    sys.exit(Show.done())

_lines = sys.argv[1:]

try:
    import re
    import cv2
    import json
    import time
    import numpy
    import random
    import typing
    import signal
    import inspect
    import asyncio
    import aiofiles
    import datetime
    from pathlib import Path
    from rich.prompt import Prompt
    from engine.manage import Manage
    from engine.switch import Switch
    from engine.terminal import Terminal
    from engine.active import Active, Review, RichSink, FramixReporterError
    from frameflow.skills.config import Option
    from frameflow.skills.config import Deploy
    from frameflow.skills.config import Script
    from frameflow.skills.insert import Insert
    from frameflow.skills.parser import Parser
    from nexaflow import toolbox
    from nexaflow.report import Report
    from nexaflow.video import VideoObject, VideoFrame
    from nexaflow.cutter.cutter import VideoCutter
    from nexaflow.hook import FrameSizeHook, FrameSaveHook
    from nexaflow.hook import PaintCropHook, PaintOmitHook
    from nexaflow.classifier.keras_classifier import KerasStruct
except (ImportError, ModuleNotFoundError):
    Show.console.print_exception()
    sys.exit(Show.fail())


class Missions(object):

    def __init__(self, *args, **kwargs):
        self.flick, self.carry, self.fully, self.quick, self.basic, self.keras, self.alone, self.whist, self.group, *_ = args
        self.atom_total_temp = kwargs["atom_total_temp"]
        self.main_share_temp = kwargs["main_share_temp"]
        self.main_total_temp = kwargs["main_total_temp"]
        self.view_share_temp = kwargs["view_share_temp"]
        self.view_total_temp = kwargs["view_total_temp"]
        self.initial_option = kwargs["initial_option"]
        self.initial_deploy = kwargs["initial_deploy"]
        self.initial_script = kwargs["initial_script"]
        self.total_place = kwargs["total_place"]
        self.model_place = kwargs["model_place"]
        self.model_shape = kwargs["model_shape"]
        self.model_aisle = kwargs["model_aisle"]
        self.adb = kwargs["adb"]
        self.fmp = kwargs["fmp"]
        self.fpb = kwargs["fpb"]
        self.scc = kwargs["scc"]

    @staticmethod
    def accelerate(folder: str):

        class Entry(object):

            def __init__(self, title: str, place: str, sheet: list):
                self.title, self.place, self.sheet = title, place, sheet

        entry_list = [
            Entry(os.path.basename(root), root, [
                os.path.join(root, i) for i in sorted(file) if os.path.basename(i) != "nexaflow.log"
            ]) for root, _, file in os.walk(folder) if file
        ]
        return entry_list

    @staticmethod
    def enforce(r: "Report", c: "KerasStruct", start: int, end: int, cost: float):
        with Insert(os.path.join(r.reset_path, f"{const.NAME}_data.db")) as database:
            if c:
                column_list = [
                    'total_path', 'title', 'query_path', 'query', 'stage', 'frame_path', 'extra_path', 'proto_path'
                ]
                database.create('stocks', *column_list)
                stage = {'stage': {'start': start, 'end': end, 'cost': cost}}
                database.insert(
                    'stocks', column_list,
                    (r.total_path, r.title, r.query_path, r.query,
                     json.dumps(stage), r.frame_path, r.extra_path, r.proto_path)
                )
            else:
                column_list = [
                    'total_path', 'title', 'query_path', 'query', 'stage', 'frame_path'
                ]
                database.create('stocks', *column_list)
                stage = {'stage': {'start': start, 'end': end, 'cost': cost}}
                database.insert(
                    'stocks', column_list,
                    (r.total_path, r.title, r.query_path, r.query, json.dumps(stage), r.frame_path)
                )

    # """Child Process"""
    def amazing(self, vision: typing.Union[str, os.PathLike], deploy: typing.Optional["Deploy"], *args, **kwargs):
        attack = self.total_place, self.model_place, self.model_shape, self.model_aisle
        charge = _platform, self.fmp, self.fpb
        alynex = Alynex(*attack, *charge)
        loop = asyncio.get_event_loop()
        loop_complete = loop.run_until_complete(
            alynex.ask_analyzer(vision, deploy, *args, **kwargs)
        )
        return loop_complete

    # """Child Process"""
    def video_file_task(self, video_file: str, deploy: "Deploy"):
        reporter = Report(self.total_place)
        reporter.title = f"{const.DESC}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        reporter.query = time.strftime('%Y%m%d%H%M%S')
        new_video_path = os.path.join(reporter.video_path, os.path.basename(video_file))

        shutil.copy(video_file, new_video_path)

        loop = asyncio.get_event_loop()

        if self.quick:
            logger.info(f"★★★ 快速模式 ★★★")
            const_filter = [f"fps={deploy.frate}"] if deploy.color else [f"fps={deploy.frate}", "format=gray"]
            if deploy.shape:
                original_shape = loop.run_until_complete(
                    Switch.ask_video_larger(self.fpb, new_video_path)
                )
                w, h, ratio = loop.run_until_complete(
                    Switch.ask_magic_frame(original_shape, deploy.shape)
                )
                video_filter_list = const_filter + [f"scale={w}:{h}"]
                logger.debug(f"Image Shape: [W:{w} H{h} Ratio:{ratio}]")
            elif deploy.scale:
                scale = max(0.1, min(1.0, deploy.scale))
                video_filter_list = const_filter + [f"scale=iw*{scale}:ih*{scale}"]
                logger.debug(f"Image Scale: {deploy.scale}")
            else:
                scale = const.COMPRESS
                video_filter_list = const_filter + [f"scale=iw*{scale}:ih*{scale}"]

            logger.info(f"应用过滤器: {video_filter_list}")

            duration = loop.run_until_complete(
                Switch.ask_video_length(self.fpb, new_video_path)
            )
            vision_start, vision_close, vision_limit = loop.run_until_complete(
                Switch.ask_magic_point(
                    Parser.parse_mills(deploy.start),
                    Parser.parse_mills(deploy.close),
                    Parser.parse_mills(deploy.limit),
                    duration
                )
            )
            vision_start = Parser.parse_times(vision_start)
            vision_close = Parser.parse_times(vision_close)
            vision_limit = Parser.parse_times(vision_limit)
            logger.info(f"视频时长: [{duration}] [{Parser.parse_times(duration)}]")
            logger.info(f"视频剪辑: start=[{vision_start}] close=[{vision_close}] limit=[{vision_limit}]")

            loop.run_until_complete(
                Switch.ask_video_detach(
                    self.fmp, video_filter_list, new_video_path, reporter.frame_path,
                    start=vision_start, close=vision_close, limit=vision_limit
                )
            )

            result = {
                "total": os.path.basename(reporter.total_path),
                "title": reporter.title,
                "query": reporter.query,
                "stage": {"start": 0, "end": 0, "cost": 0},
                "frame": os.path.basename(reporter.frame_path),
                "style": "quick"
            }
            Show.console.print_json(data=result)
            logger.debug(f"Quicker: {json.dumps(result, ensure_ascii=False)}")
            loop.run_until_complete(reporter.load(result))

            logger.info(f"正在生成汇总报告 ...")
            try:
                total_html = loop.run_until_complete(
                    reporter.ask_create_total_report(
                        os.path.dirname(reporter.total_path),
                        self.group,
                        loop.run_until_complete(achieve(self.view_share_temp)),
                        loop.run_until_complete(achieve(self.view_total_temp))
                    )
                )
            except FramixReporterError:
                return Show.console.print_exception()

            logger.info(f"成功生成汇总报告 {os.path.relpath(total_html)}")
            return reporter.total_path

        elif self.keras and not self.basic:
            attack = self.total_place, self.model_place, self.model_shape, self.model_aisle
        else:
            attack = self.total_place, None, None, None

        charge = _platform, self.fmp, self.fpb

        alynex = Alynex(*attack, *charge)
        logger.info(f"★★★ {'智能模式' if alynex.kc else '基础模式'} ★★★")

        futures = loop.run_until_complete(
            alynex.ask_analyzer(
                new_video_path, deploy, reporter.frame_path, reporter.extra_path
            )
        )

        if futures is None:
            return None
        start, end, cost, struct = futures.material

        result = {
            "total": os.path.basename(reporter.total_path),
            "title": reporter.title,
            "query": reporter.query,
            "stage": {"start": start, "end": end, "cost": f"{cost:.5f}"},
            "frame": os.path.basename(reporter.frame_path)
        }

        if struct:
            if isinstance(
                    tmp := loop.run_until_complete(achieve(self.atom_total_temp)), Exception
            ):
                return logger.error(f"[bold red]{tmp}[/bold red]")
            logger.info(f"模版引擎正在渲染 ...")
            original_inform = loop.run_until_complete(
                reporter.ask_draw(struct, reporter.proto_path, tmp)
            )
            logger.info(f"模版引擎渲染完毕 {os.path.relpath(original_inform)}")
            result["extra"] = os.path.basename(reporter.extra_path)
            result["proto"] = os.path.basename(original_inform)
            result["style"] = "keras"
        else:
            result["style"] = "basic"

        Show.console.print_json(data=result)
        logger.debug(f"Restore: {json.dumps(result, ensure_ascii=False)}")
        loop.run_until_complete(reporter.load(result))

        self.enforce(reporter, struct, start, end, cost)

        logger.info(f"正在生成汇总报告 ...")
        try:
            total_html = loop.run_until_complete(
                reporter.ask_create_total_report(
                    os.path.dirname(reporter.total_path),
                    self.group,
                    loop.run_until_complete(achieve(self.main_share_temp)),
                    loop.run_until_complete(achieve(self.main_total_temp))
                )
            )
        except FramixReporterError:
            return Show.console.print_exception()

        logger.info(f"成功生成汇总报告 {os.path.relpath(total_html)}")
        return reporter.total_path

    # """Child Process"""
    def video_data_task(self, video_data: str, deploy: "Deploy"):
        reporter = Report(self.total_place)

        loop = asyncio.get_event_loop()

        if self.quick:
            logger.info(f"★★★ 快速模式 ★★★")
            for video in self.accelerate(video_data):
                reporter.title = video.title
                for path in video.sheet:
                    reporter.query = os.path.basename(path).split(".")[0]
                    shutil.copy(path, reporter.video_path)
                    new_video_path = os.path.join(reporter.video_path, os.path.basename(path))

                    const_filter = [f"fps={deploy.frate}"] if deploy.color else [f"fps={deploy.frate}", "format=gray"]
                    if deploy.shape:
                        original_shape = loop.run_until_complete(
                            Switch.ask_video_larger(self.fpb, new_video_path)
                        )
                        w, h, ratio = loop.run_until_complete(
                            Switch.ask_magic_frame(original_shape, deploy.shape)
                        )
                        video_filter_list = const_filter + [f"scale={w}:{h}"]
                        logger.debug(f"Image Shape: [W:{w} H{h} Ratio:{ratio}]")
                    elif deploy.scale:
                        scale = max(0.1, min(1.0, deploy.scale))
                        video_filter_list = const_filter + [f"scale=iw*{scale}:ih*{scale}"]
                        logger.debug(f"Image Scale: {deploy.scale}")
                    else:
                        scale = const.COMPRESS
                        video_filter_list = const_filter + [f"scale=iw*{scale}:ih*{scale}"]

                    logger.info(f"应用过滤器: {video_filter_list}")

                    duration = loop.run_until_complete(
                        Switch.ask_video_length(self.fpb, new_video_path)
                    )
                    vision_start, vision_close, vision_limit = loop.run_until_complete(
                        Switch.ask_magic_point(
                            Parser.parse_mills(deploy.start),
                            Parser.parse_mills(deploy.close),
                            Parser.parse_mills(deploy.limit),
                            duration
                        )
                    )
                    vision_start = Parser.parse_times(vision_start)
                    vision_close = Parser.parse_times(vision_close)
                    vision_limit = Parser.parse_times(vision_limit)
                    logger.info(f"视频时长: [{duration}] [{Parser.parse_times(duration)}]")
                    logger.info(f"视频剪辑: start=[{vision_start}] close=[{vision_close}] limit=[{vision_limit}]")

                    loop.run_until_complete(
                        Switch.ask_video_detach(
                            self.fmp, video_filter_list, new_video_path, reporter.frame_path,
                            start=deploy.start, close=deploy.close, limit=deploy.limit
                        )
                    )

                    result = {
                        "total": os.path.basename(reporter.total_path),
                        "title": reporter.title,
                        "query": reporter.query,
                        "stage": {"start": 0, "end": 0, "cost": 0},
                        "frame": os.path.basename(reporter.frame_path),
                        "style": "quick"
                    }
                    Show.console.print_json(data=result)
                    logger.debug(f"Quicker: {json.dumps(result, ensure_ascii=False)}")
                    loop.run_until_complete(reporter.load(result))

            logger.info(f"正在生成汇总报告 ...")
            try:
                total_html = loop.run_until_complete(
                    reporter.ask_create_total_report(
                        os.path.dirname(reporter.total_path),
                        self.group,
                        loop.run_until_complete(achieve(self.view_share_temp)),
                        loop.run_until_complete(achieve(self.view_total_temp))
                    )
                )
            except FramixReporterError:
                return Show.console.print_exception()

            logger.info(f"成功生成汇总报告 {os.path.relpath(total_html)}")
            return reporter.total_path

        elif self.keras and not self.basic:
            attack = self.total_place, self.model_place, self.model_shape, self.model_aisle
        else:
            attack = self.total_place, None, None, None

        charge = _platform, self.fmp, self.fpb

        alynex = Alynex(*attack, *charge)
        logger.info(f"★★★ {'智能模式' if alynex.kc else '基础模式'} ★★★")

        for video in self.accelerate(video_data):
            reporter.title = video.title
            for path in video.sheet:
                reporter.query = os.path.basename(path).split(".")[0]
                shutil.copy(path, reporter.video_path)
                new_video_path = os.path.join(reporter.video_path, os.path.basename(path))

                futures = loop.run_until_complete(
                    alynex.ask_analyzer(
                        new_video_path, deploy, reporter.frame_path, reporter.extra_path
                    )
                )
                if futures is None:
                    continue
                start, end, cost, struct = futures.material

                result = {
                    "total": os.path.basename(reporter.total_path),
                    "title": reporter.title,
                    "query": reporter.query,
                    "stage": {"start": start, "end": end, "cost": f"{cost:.5f}"},
                    "frame": os.path.basename(reporter.frame_path)
                }

                if struct:
                    if isinstance(
                            tmp := loop.run_until_complete(achieve(self.atom_total_temp)), Exception
                    ):
                        return logger.error(f"[bold red]{tmp}[/bold red]")
                    logger.info(f"模版引擎正在渲染 ...")
                    original_inform = loop.run_until_complete(
                        reporter.ask_draw(struct, reporter.proto_path, tmp)
                    )
                    logger.info(f"模版引擎渲染完毕 {os.path.relpath(original_inform)}")
                    result["extra"] = os.path.basename(reporter.extra_path)
                    result["proto"] = os.path.basename(original_inform)
                    result["style"] = "keras"
                else:
                    result["style"] = "basic"

                Show.console.print_json(data=result)
                logger.debug(f"Restore: {json.dumps(result, ensure_ascii=False)}")
                loop.run_until_complete(reporter.load(result))

                self.enforce(reporter, struct, start, end, cost)

        logger.info(f"正在生成汇总报告 ...")
        try:
            total_html = loop.run_until_complete(
                reporter.ask_create_total_report(
                    os.path.dirname(reporter.total_path),
                    self.group,
                    loop.run_until_complete(achieve(self.main_share_temp)),
                    loop.run_until_complete(achieve(self.main_total_temp))
                )
            )
        except FramixReporterError:
            return Show.console.print_exception()

        logger.info(f"成功生成汇总报告 {os.path.relpath(total_html)}")
        return reporter.total_path

    # """Child Process"""
    def train_model(self, video_file: str, deploy: "Deploy"):
        if not os.path.isfile(video_file):
            return logger.error(f"{video_file} 视频文件未找到 ...")
        logger.info(f"视频文件 {video_file} ...")

        screen = cv2.VideoCapture(video_file)
        if not screen.isOpened():
            return logger.error(f"{video_file} 视频文件损坏 ...")
        screen.release()
        logger.info(f"{video_file} 可正常播放 ...")

        reporter = Report(self.total_place)
        reporter.title = f"Model_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}"
        if not os.path.exists(reporter.query_path):
            os.makedirs(reporter.query_path, exist_ok=True)

        video_temp_file = os.path.join(
            reporter.query_path, f"tmp_fps{deploy.frate}.mp4"
        )

        loop = asyncio.get_event_loop()
        duration = loop.run_until_complete(
            Switch.ask_video_length(self.fpb, video_file)
        )
        vision_start, vision_close, vision_limit = loop.run_until_complete(
            Switch.ask_magic_point(
                Parser.parse_mills(deploy.start),
                Parser.parse_mills(deploy.close),
                Parser.parse_mills(deploy.limit),
                duration
            )
        )
        vision_start = Parser.parse_times(vision_start)
        vision_close = Parser.parse_times(vision_close)
        vision_limit = Parser.parse_times(vision_limit)
        logger.info(f"视频时长: [{duration}] [{Parser.parse_times(duration)}]")
        logger.info(f"视频剪辑: start=[{vision_start}] close=[{vision_close}] limit=[{vision_limit}]")

        asyncio.run(
            Switch.ask_video_change(
                self.fmp, deploy.frate, video_file, video_temp_file,
                start=vision_start, close=vision_close, limit=vision_limit
            )
        )

        video = VideoObject(video_temp_file)
        logger.info(f"视频帧长度: {video.frame_count} 分辨率: {video.frame_size}")
        logger.info(f"加载到内存: {video.name}")
        video_load_time = time.time()
        video.load_frames(
            load_hued=False, none_gray=True
        )
        logger.info(f"灰度帧已加载: {video.frame_details(video.grey_data)}")
        logger.info(f"视频加载耗时: {time.time() - video_load_time:.2f} 秒")

        cutter = VideoCutter()

        logger.info(f"压缩视频: {video.name}")
        logger.info(f"视频帧数: {video.frame_count} 帧片段数: {video.frame_count - 1} 帧分辨率: {video.frame_size}")
        cut_start_time = time.time()
        cut_range = cutter.cut(video=video, block=deploy.block)
        logger.info(f"压缩完成: {video.name}")
        logger.info(f"压缩耗时: {time.time() - cut_start_time:.2f} 秒")

        stable, unstable = cut_range.get_range(
            threshold=deploy.thres, offset=deploy.shift
        )

        if deploy.shape:
            original_shape = loop.run_until_complete(
                Switch.ask_video_larger(self.fpb, video_file)
            )
            w, h, ratio = loop.run_until_complete(
                Switch.ask_magic_frame(original_shape, deploy.shape)
            )
            target_shape = w, h
            target_scale = deploy.scale
            logger.info(f"调整宽高比: {w} x {h}")
        elif deploy.scale:
            target_shape = deploy.shape
            target_scale = max(0.1, min(1.0, deploy.scale))
        else:
            target_shape = deploy.shape
            target_scale = const.COMPRESS

        cut_range.pick_and_save(
            range_list=stable,
            frame_count=20,
            to_dir=reporter.query_path,
            meaningful_name=True,
            not_grey=deploy.color,
            compress_rate=target_scale,
            target_size=target_shape
        )

        os.remove(video_temp_file)

    # """Child Process"""
    def build_model(self, video_data: str, deploy: "Deploy"):
        if not os.path.isdir(video_data):
            return logger.error(f"编译模型需要一个已经分类的文件夹 ...")

        real_path, file_list = "", []
        logger.debug(f"搜索文件夹: {video_data}")
        for root, dirs, files in os.walk(video_data, topdown=False):
            for name in files:
                file_list.append(os.path.join(root, name))
            for name in dirs:
                if len(name) == 1 and re.search(r"0", name):
                    real_path = os.path.join(root, name)
                    logger.debug(f"分类文件夹: {real_path}")
                    break

        if not real_path or len(file_list) == 0:
            return logger.error(f"文件夹未正确分类 ...")

        image, image_color, image_aisle = None, "grayscale", 1
        for image_file in os.listdir(real_path):
            image_path = os.path.join(real_path, image_file)
            if not os.path.isfile(image_path):
                return logger.error(f"存在无效的图像文件 ...")
            image = cv2.imread(image_path)
            logger.info(f"图像分辨率: {image.shape}")
            if image.ndim == 3:
                if numpy.array_equal(image[:, :, 0], image[:, :, 1]) and numpy.array_equal(image[:, :, 1], image[:, :, 2]):
                    logger.info(f"The image is grayscale image, stored in RGB format ...")
                else:
                    logger.info(f"The image is color image ...")
                    image_color, image_aisle = "rgb", image.ndim
            else:
                logger.info(f"The image is grayscale image ...")
            break

        final_path = os.path.dirname(real_path)
        new_model_path = os.path.join(
            final_path, f"Create_Model_{time.strftime('%Y%m%d%H%M%S')}", f"{random.randint(100, 999)}"
        )

        image_shape = deploy.shape if deploy.shape else (image.shape if image.shape else self.model_shape)
        w, h, *_ = image_shape if image_shape else const.MODEL_SHAPE
        name = f"Gray" if image_aisle == 1 else f"Hued"
        new_model_name = f"Keras_{name}_W{w}_H{h}_{random.randint(10000, 99999)}.h5"

        kc = KerasStruct(color=image_color, aisle=image_aisle, data_size=image_shape)
        kc.build(final_path, new_model_path, new_model_name)

    async def combines_main(self, merge: list):
        major, total = await asyncio.gather(
            achieve(self.main_share_temp), achieve(self.main_total_temp),
            return_exceptions=True
        )
        logger.info(f"正在生成汇总报告 ...")
        state_list = await asyncio.gather(
            *(Report.ask_create_total_report(m, self.group, major, total) for m in merge)
        )
        for state in state_list:
            if isinstance(state, Exception):
                logger.error(f"[bold #FFC0CB]{state}[/bold #FFC0CB]")
            logger.info(f"成功生成汇总报告 {os.path.relpath(state)}")

    async def combines_view(self, merge: list):
        views, total = await asyncio.gather(
            achieve(self.view_share_temp), achieve(self.view_total_temp),
            return_exceptions=True
        )
        logger.info(f"正在生成汇总报告 ...")
        state_list = await asyncio.gather(
            *(Report.ask_create_total_report(m, self.group, views, total) for m in merge)
        )
        for state in state_list:
            if isinstance(state, Exception):
                logger.error(f"[bold #FFC0CB]{state}[/bold #FFC0CB]")
            logger.info(f"成功生成汇总报告 {os.path.relpath(state)}")

    async def painting(self, *args, **__):

        import tempfile
        import PIL.Image
        import PIL.ImageDraw
        import PIL.ImageFont

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
                        if sum(crop.values()) > 0:
                            x, y, x_size, y_size = crop["x"], crop["y"], crop["x_size"], crop["y_size"]
                            paint_crop_hook = PaintCropHook((y_size, x_size), (y, x))
                            paint_crop_hook.do(new_image)

                if len(deploy.omits) > 0:
                    for omit in deploy.omits:
                        if sum(omit.values()) > 0:
                            x, y, x_size, y_size = omit["x"], omit["y"], omit["x_size"], omit["y_size"]
                            paint_omit_hook = PaintOmitHook((y_size, x_size), (y, x))
                            paint_omit_hook.do(new_image)

                cv2.imencode(".png", new_image.data)[1].tofile(image_save_path)

                image_file = PIL.Image.open(image_save_path)
                image_file = image_file.convert("RGB")

                original_w, original_h = image_file.size
                if deploy.shape:
                    twist_w, twist_h, _ = await Switch.ask_magic_frame(image_file.size, deploy.shape)
                else:
                    twist_w, twist_h = original_w, original_h

                min_scale, max_scale = 0.1, 1.0
                if deploy.scale:
                    image_scale = max_scale if deploy.shape else max(min_scale, min(max_scale, deploy.scale))
                else:
                    image_scale = max_scale if deploy.shape else const.COMPRESS

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

                draw = PIL.ImageDraw.Draw(resized)
                font = PIL.ImageFont.load_default()

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

        cmd_lines, platform, deploy, level, power, loop, *_ = args

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
                reporter = Report(self.total_place)
                reporter.title = f"Hooks_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
                for device, resize_img in zip(device_list, resized_result):
                    img_save_path = os.path.join(
                        reporter.query_path, f"hook_{device.serial}_{random.randint(10000, 99999)}.png"
                    )
                    resize_img.save(img_save_path)
                    logger.info(f"保存图片: {os.path.relpath(img_save_path)}")
                break
            elif action.strip().upper() == "N":
                break
            else:
                logger.warning(f"[bold red]没有该选项,请重新输入[/bold red] ...\n")

    async def analysis(self, *args, **__):

        async def commence():

            # Wait Device Online
            async def wait_for_device(serial):
                logger.info(f"wait-for-device {serial} ...")
                await Terminal.cmd_line(self.adb, "-s", serial, "wait-for-device")

            await asyncio.gather(
                *(wait_for_device(device.serial) for device in device_list)
            )

            todo_list = []

            fmt_dir = time.strftime("%Y%m%d%H%M%S") if any((self.quick, self.basic, self.keras)) else None
            for device in device_list:
                await asyncio.sleep(0.2)
                record.device_events[device.serial] = {
                    "head_event": asyncio.Event(), "done_event": asyncio.Event(),
                    "stop_event": asyncio.Event(), "fail_event": asyncio.Event()
                }
                if fmt_dir:
                    reporter.query = os.path.join(fmt_dir, device.serial)
                video_temp, transports = await record.start_record(
                    device.serial, reporter.video_path, record.device_events[device.serial]
                )
                todo_list.append(
                    [video_temp, transports, reporter.total_path, reporter.title, reporter.query_path,
                     reporter.query, reporter.frame_path, reporter.extra_path, reporter.proto_path]
                )
            return todo_list

        async def analysis_tactics():
            if len(task_list) == 0:
                return None

            if self.quick:
                logger.info(f"★★★ 快速模式 ★★★")
                const_filter = [f"fps={deploy.frate}"] if deploy.color else [f"fps={deploy.frate}", "format=gray"]
                if deploy.shape:
                    original_shape_list = await asyncio.gather(
                        *(Switch.ask_video_larger(self.fpb, video_temp) for video_temp, *_ in task_list)
                    )
                    final_shape_list = await asyncio.gather(
                        *(Switch.ask_magic_frame(original_shape, deploy.shape) for original_shape in original_shape_list)
                    )
                    video_filter_list = [
                        const_filter + [f"scale={w}:{h}"] for w, h, ratio in final_shape_list
                    ]
                elif deploy.scale:
                    scale = max(0.1, min(1.0, deploy.scale))
                    video_filter_list = [
                        const_filter + [f"scale=iw*{scale}:ih*{scale}"] for video_temp, *_ in task_list
                    ]
                else:
                    scale = const.COMPRESS
                    video_filter_list = [
                        const_filter + [f"scale=iw*{scale}:ih*{scale}"] for video_temp, *_ in task_list
                    ]

                for flt in video_filter_list:
                    logger.info(f"应用过滤器: {flt}")

                duration_list = await asyncio.gather(
                    *(Switch.ask_video_length(self.fpb, video_temp) for video_temp, *_ in task_list)
                )
                duration_result_list = await asyncio.gather(
                    *(Switch.ask_magic_point(
                        Parser.parse_mills(deploy.start),
                        Parser.parse_mills(deploy.close),
                        Parser.parse_mills(deploy.limit),
                        duration
                    ) for duration in duration_list)
                )

                all_duration = []
                for (vision_start, vision_close, vision_limit), duration in zip(duration_result_list, duration_list):
                    all_duration.append(
                        (
                            vision_start := Parser.parse_times(vision_start),
                            vision_close := Parser.parse_times(vision_close),
                            vision_limit := Parser.parse_times(vision_limit)
                        )
                    )
                    logger.info(f"视频时长: [{duration}] [{Parser.parse_times(duration)}]")
                    logger.info(f"视频剪辑: start=[{vision_start}] close=[{vision_close}] limit=[{vision_limit}]")

                await asyncio.gather(
                    *(Switch.ask_video_detach(
                        self.fmp, video_filter, video_temp, frame_path,
                        start=vision_start, close=vision_close, limit=vision_limit
                    ) for (
                        video_temp, *_, frame_path, _, _), video_filter, (
                        vision_start, vision_close, vision_limit
                    ) in zip(task_list, video_filter_list, duration_result_list))
                )

                for *_, total_path, title, query_path, query, frame_path, _, _ in task_list:
                    result = {
                        "total": os.path.basename(total_path),
                        "title": title,
                        "query": query,
                        "stage": {"start": 0, "end": 0, "cost": 0},
                        "frame": os.path.basename(frame_path),
                        "style": "quick"
                    }
                    Show.console.print_json(data=result)
                    logger.debug(f"Quicker: {json.dumps(result, ensure_ascii=False)}")
                    await reporter.load(result)

            elif self.basic or self.keras:
                logger.info(f"★★★ {'智能模式' if alynex.kc else '基础模式'} ★★★")

                if len(task_list) == 1:
                    futures = await asyncio.gather(
                        *(alynex.ask_analyzer(
                            video_temp, deploy, frame_path, extra_path
                        ) for video_temp, *_, frame_path, extra_path, _ in task_list)
                    )
                else:
                    with ProcessPoolExecutor(power, None, Active.active, ("ERROR",)) as exe:
                        task = [
                            loop.run_in_executor(
                                exe, self.amazing, video_temp, deploy, frame_path, extra_path
                            ) for video_temp, *_, frame_path, extra_path, _ in task_list
                        ]
                        futures = await asyncio.gather(*task)

                for future, todo in zip(futures, task_list):
                    if future is None:
                        continue

                    start, end, cost, struct = future.material
                    *_, total_path, title, query_path, query, frame_path, extra_path, proto_path = todo

                    result = {
                        "total": os.path.basename(total_path),
                        "title": title,
                        "query": query,
                        "stage": {"start": start, "end": end, "cost": f"{cost:.5f}"},
                        "frame": os.path.basename(frame_path)
                    }

                    if struct:
                        if isinstance(
                                tmp := await achieve(self.atom_total_temp), Exception
                        ):
                            return logger.error(f"[bold red]{tmp}[/bold red]")
                        logger.info(f"模版引擎正在渲染 ...")
                        original_inform = await reporter.ask_draw(struct, proto_path, tmp)
                        logger.info(f"模版引擎渲染完毕 {os.path.relpath(original_inform)}")
                        result["extra"] = os.path.basename(extra_path)
                        result["proto"] = os.path.basename(original_inform)
                        result["style"] = "keras"
                    else:
                        result["style"] = "basic"

                    Show.console.print_json(data=result)
                    logger.debug(f"Restore: {json.dumps(result, ensure_ascii=False)}")
                    await reporter.load(result)

                    self.enforce(reporter, struct, start, end, cost)

            else:
                return logger.info(f"★★★ 录制模式 ★★★")

        async def device_mode_view():
            logger.info(f"[bold]<Link> <{'单设备模式' if len(device_list) == 1 else '多设备模式'}>")
            for device in device_list:
                logger.info(f"[bold #00FFAF]Connect:[/bold #00FFAF] {device}")

        async def all_time():
            await asyncio.gather(
                *(record.timing_less(serial, events, timer_mode)
                  for serial, events in record.device_events.items())
            )

        async def all_stop():
            effective_list = await asyncio.gather(
                *(record.close_record(video_temp, transports, events)
                  for (_, events), (video_temp, transports, *_) in zip(record.device_events.items(), task_list))
            )
            for (idx, (effective, video_name)), _ in zip(enumerate(effective_list), task_list):
                if effective.startswith("视频录制失败"):
                    task_list.pop(idx)
                logger.info(f"{effective}: {video_name} ...")

        async def all_over():

            async def balance(duration, video_src):
                start_time_point = duration - standard
                end_time_point = duration
                start_time_str = str(datetime.timedelta(seconds=start_time_point))
                end_time_str = str(datetime.timedelta(seconds=end_time_point))

                logger.info(f"{os.path.basename(video_src)} {duration} [{start_time_str} - {end_time_str}]")
                video_dst = os.path.join(
                    os.path.dirname(video_src), f"tailor_fps{deploy.frate}_{random.randint(100, 999)}.mp4"
                )

                await Switch.ask_video_tailor(
                    self.fmp, video_src, video_dst, start=start_time_str, limit=end_time_str
                )
                os.remove(video_src)
                logger.info(f"移除旧的视频 {Path(video_src).name}")
                return video_dst

            if len(task_list) == 0:
                return logger.warning(f"没有有效任务 ...")

            if self.alone:
                return logger.warning(f"独立控制模式不会平衡视频录制时间 ...")

            duration_list = await asyncio.gather(
                *(Switch.ask_video_length(self.fpb, video_temp) for video_temp, *_ in task_list)
            )
            duration_list = [duration for duration in duration_list if not isinstance(duration, Exception)]
            if len(duration_list) == 0:
                return task_list.clear()

            logger.info(f"标准录制时间: {(standard := min(duration_list))}")
            balance_task = [
                balance(duration, video_src)
                for duration, (video_src, *_) in zip(duration_list, task_list)
            ]
            video_dst_list = await asyncio.gather(*balance_task)
            for idx, dst in enumerate(video_dst_list):
                task_list[idx][0] = dst

        async def combines():
            if len(reporter.range_list) == 0:
                return None
            report = getattr(self, "combines_view" if self.quick else "combines_main")
            return await report([os.path.dirname(reporter.total_path)])

        async def load_commands(script: typing.Union[str, "os.PathLike"]):
            try:
                async with aiofiles.open(script, "r", encoding=const.CHARSET) as f:
                    file_list = json.loads(await f.read())["command"]
                    exec_dict = {
                        file_key: {
                            **({"parser": cmds["parser"]} if cmds.get("parser") else {}),
                            **({"header": cmds["header"]} if cmds.get("header") else {}),
                            **({"looper": cmds["looper"]} if cmds.get("looper") else {}),
                            **({"prefix": [c for c in cmds.get("prefix", []) if c["cmds"]]} if any(
                                c["cmds"] for c in cmds.get("prefix", [])) else {}),
                            **({"action": [c for c in cmds.get("action", []) if c["cmds"]]} if any(
                                c["cmds"] for c in cmds.get("action", [])) else {}),
                            **({"suffix": [c for c in cmds.get("suffix", []) if c["cmds"]]} if any(
                                c["cmds"] for c in cmds.get("suffix", [])) else {}),
                        } for file_dict in file_list for file_key, cmds in file_dict.items()
                        if any(c["cmds"] for c in cmds.get("action", []))
                    }
            except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                return e
            return exec_dict

        async def loop_commands():
            for device_action in device_action_list:
                if device_cmds := device_action.get("cmds", []):
                    device_args = device_action.get("args", [])
                    device_args += [[]] * (len(device_cmds) - len(device_args))
                    yield list(zip(device_cmds, device_args))

        async def exec_function():
            async for cmd_arg_pairs in loop_commands():
                exec_tasks = []
                for device in device_list:
                    for exec_func, exec_args in cmd_arg_pairs:
                        if exec_func == "audio_player":
                            await dynamically(exec_func, exec_args, player)
                        else:
                            exec_tasks.append(
                                asyncio.create_task(dynamically(exec_func, exec_args, device))
                            )

                exec_status_list = await asyncio.gather(*exec_tasks, return_exceptions=True)

                for status in exec_status_list:
                    if isinstance(status, Exception):
                        logger.error(f"[bold #FFC0CB]{status}[/bold #FFC0CB]")

        async def dynamically(exec_func: str, exec_args: list, bean: type[typing.Union["Device", "Player"]]):
            if not (callable(function := getattr(bean, exec_func, None))):
                return logger.error(f"No callable [bold #FFC0CB]{exec_func}[/bold #FFC0CB] ...")

            logger.info(f"{getattr(bean, 'serial', bean.__class__.__name__)} {function.__name__} {exec_args}")
            try:
                if inspect.iscoroutinefunction(function):
                    await function(*exec_args)
                else:
                    await asyncio.to_thread(function, *exec_args)
            except Exception as e:
                return e

        # Initialization ===============================================================================================
        cmd_lines, platform, deploy, level, power, loop, *_ = args

        manage_ = Manage(self.adb)
        device_list = await manage_.operate_device()

        titles_ = {"quick": "Quick", "basic": "Basic", "keras": "Keras"}
        input_title_ = next((title for key, title in titles_.items() if getattr(self, key)), "Video")
        reporter = Report(self.total_place)

        if self.keras and not self.quick and not self.basic:
            attack_ = self.total_place, self.model_place, self.model_shape, self.model_aisle
        else:
            attack_ = self.total_place, None, None, None

        charge_ = platform, self.fmp, self.fpb

        alynex = Alynex(*attack_, *charge_)
        Show.load_animation(cmd_lines)

        from engine.device import Device
        from engine.medias import Record, Player
        record = Record(self.scc, platform, alone=self.alone, whist=self.whist)
        player = Player()

        # Initialization ===============================================================================================

        # Flick Loop ===================================================================================================
        if self.flick:
            const_title_ = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
            reporter.title = f"{input_title_}_{const_title_}"
            timer_mode = 5
            while True:
                try:
                    await device_mode_view()
                    start_tips_ = f"<<<按 Enter 开始 [bold #D7FF5F]{timer_mode}[/bold #D7FF5F] 秒>>>"
                    if action_ := Prompt.ask(prompt=f"[bold #5FD7FF]{start_tips_}[/bold #5FD7FF]", console=Show.console):
                        if (select_ := action_.strip().lower()) == "serial":
                            device_list = await manage_.operate_device()
                            continue
                        elif "header" in select_:
                            if match_ := re.search(r"(?<=header\s).*", select_):
                                if hd_ := match_.group().strip():
                                    src_hd_, a_, b_ = f"{input_title_}_{time.strftime('%Y%m%d_%H%M%S')}", 10000, 99999
                                    logger.success("[bold green]新标题设置成功[/bold green] ...")
                                    reporter.title = f"{src_hd_}_{hd_}" if hd_ else f"{src_hd_}_{random.randint(a_, b_)}"
                                    continue
                            raise ValueError
                        elif select_ in ["invent", "create"]:
                            await combines()
                            break
                        elif select_ == "deploy":
                            logger.warning("修改 [bold yellow]deploy.json[/bold yellow] 文件后请完全退出编辑器进程再继续操作 ...")
                            deploy.dump_deploy(self.initial_deploy)
                            first_ = ["Notepad"] if platform == "win32" else ["open", "-W", "-a", "TextEdit"]
                            first_.append(self.initial_deploy)
                            await Terminal.cmd_line(*first_)
                            deploy.load_deploy(self.initial_deploy)
                            deploy.view_deploy()
                            continue
                        elif select_.isdigit():
                            timer_value_, lower_bound_, upper_bound_ = int(select_), 5, 300
                            if timer_value_ > 300 or timer_value_ < 5:
                                bound_tips_ = f"{lower_bound_} <= [bold #FFD7AF]Time[/bold #FFD7AF] <= {upper_bound_}"
                                logger.info(f"[bold #FFFF87]{bound_tips_}[/bold #FFFF87]")
                            timer_mode = max(lower_bound_, min(upper_bound_, timer_value_))
                        else:
                            raise ValueError
                except ValueError:
                    Show.tips_document()
                    continue
                else:
                    task_list = await commence()
                    await all_time()
                    await all_stop()
                    await all_over()
                    await analysis_tactics()
                    check_ = await record.event_check()
                    device_list = await manage_.operate_device() if check_ else device_list
                finally:
                    await record.clean_check()
        # Flick Loop ===================================================================================================

        # Other Loop ===================================================================================================
        elif self.carry or self.fully:

            if self.carry:
                if isinstance(script_data_ := await load_commands(self.initial_script), Exception):
                    if isinstance(script_data_, FileNotFoundError):
                        Script.dump_script(self.initial_script)
                    return Show.console.print_exception()
                try:
                    script_storage_ = [{carry_: script_data_[carry_] for carry_ in list(set(self.carry))}]
                except KeyError as e_:
                    return logger.error(f"[bold #FFC0CB]{e_}[/bold #FFC0CB]")

            else:
                load_script_data_ = await asyncio.gather(
                    *(load_commands(fully_) for fully_ in self.fully), return_exceptions=True
                )
                for script_data_ in load_script_data_:
                    if isinstance(script_data_, Exception):
                        if isinstance(script_data_, FileNotFoundError):
                            Script.dump_script(self.initial_script)
                        return Show.console.print_exception()
                script_storage_ = [script_data_ for script_data_ in load_script_data_]

            await device_mode_view()
            for script_dict_ in script_storage_:
                for script_key_, script_value_ in script_dict_.items():
                    logger.info(f"Exec: {script_key_}")

                    try:
                        looper_ = int(looper_) if (looper_ := script_value_.get("looper", None)) else 1
                    except ValueError as e_:
                        logger.error(f"[bold #FFC0CB]{e_}[/bold #FFC0CB]")
                        logger.error(f"重置循环次数: {(looper_ := 1)}")

                    header_ = header_ if type(
                        header_ := script_value_.get("header", None)
                    ) is list else ([header_] if type(header_) is str else [time.strftime("%Y%m%d%H%M%S")])

                    for hd_ in header_:
                        reporter.title = f"{script_key_.replace(' ', '').strip()}_{input_title_}_{hd_}"
                        for _ in range(looper_):
                            try:

                                # prefix
                                if device_action_list := script_value_.get("prefix", None):
                                    await exec_function()

                                task_start_time_, task_list = time.time(), await commence()

                                # action
                                if device_action_list := script_value_.get("action", None):
                                    await exec_function()

                                if task_time_ := time.time() - task_start_time_ < 5:
                                    await asyncio.sleep(5 - task_time_)
                                await all_stop()

                                # suffix
                                suffix_task_ = None
                                if device_action_list := script_value_.get("suffix", None):
                                    suffix_task_ = asyncio.create_task(exec_function(), name="suffix")

                                await all_over()

                                await analysis_tactics()
                                check_ = await record.event_check()
                                device_list = await manage_.operate_device() if check_ else device_list
                                await suffix_task_ if suffix_task_ else None
                            finally:
                                await record.clean_check()

            return await combines()
        # Other Loop ===================================================================================================

        return None


class Alynex(object):

    __kc: typing.Optional["KerasStruct"] = None

    def __init__(
            self,
            total_place: typing.Optional[typing.Union[str, "os.PathLike"]],
            model_place: typing.Optional[typing.Union[str, "os.PathLike"]],
            model_shape: typing.Optional[tuple],
            model_aisle: typing.Optional[int],
            *args,
            **__
    ):

        if model_place and model_shape and model_aisle:
            try:
                self.kc = KerasStruct(data_size=model_shape, aisle=model_aisle)
                self.kc.load_model(model_place)
            except ValueError:
                Show.console.print_exception()
                self.kc = None

        self.total_place = total_place
        self.model_place = model_place
        self.model_shape = model_shape
        self.model_aisle = model_aisle
        self.oss, self.fmp, self.fpb, *_ = args

    @property
    def kc(self) -> typing.Optional["KerasStruct"]:
        return self.__kc

    @kc.setter
    def kc(self, value):
        self.__kc = value

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def ask_analyzer(
            self, vision: typing.Union[str, "os.PathLike"], deploy: "Deploy" = None, *args, **kwargs
    ) -> typing.Optional["Review"]:

        frame_path, extra_path, *_ = args

        boost = deploy.boost if deploy else kwargs.get("boost", const.BOOST)
        color = deploy.color if deploy else kwargs.get("color", const.COLOR)

        shape = deploy.shape if deploy else kwargs.get("shape", const.SHAPE)
        scale = deploy.scale if deploy else kwargs.get("scale", const.SCALE)
        start = deploy.start if deploy else kwargs.get("start", const.START)
        close = deploy.close if deploy else kwargs.get("close", const.CLOSE)
        limit = deploy.limit if deploy else kwargs.get("limit", const.LIMIT)

        begin = deploy.begin if deploy else kwargs.get("begin", const.BEGIN)
        final = deploy.final if deploy else kwargs.get("final", const.FINAL)

        crops = deploy.crops if deploy else kwargs.get("crops", const.CROPS)
        omits = deploy.omits if deploy else kwargs.get("omits", const.OMITS)

        frate = deploy.frate if deploy else kwargs.get("frate", const.FRATE)
        thres = deploy.thres if deploy else kwargs.get("thres", const.THRES)
        shift = deploy.shift if deploy else kwargs.get("shift", const.SHIFT)
        block = deploy.block if deploy else kwargs.get("block", const.BLOCK)

        async def frame_check():
            target_screen = None
            if os.path.isfile(vision):
                screen = cv2.VideoCapture(vision)
                if screen.isOpened():
                    target_screen = vision
                screen.release()
            elif os.path.isdir(vision):
                file_list = [
                    file for file in os.listdir(vision) if os.path.isfile(os.path.join(vision, file))
                ]
                if len(file_list) >= 1:
                    screen = cv2.VideoCapture(open_file := os.path.join(vision, file_list[0]))
                    if screen.isOpened():
                        target_screen = open_file
                    screen.release()
            return target_screen

        async def frame_forge(frame):
            try:
                (_, codec), pic_path = cv2.imencode(".png", frame.data), os.path.join(
                    frame_path, f"{frame.frame_id}_{format(round(frame.timestamp, 5), '.5f')}.png"
                )
                async with aiofiles.open(pic_path, "wb") as f:
                    await f.write(codec.tobytes())
            except Exception as e:
                return e

        async def frame_flick():
            logger.info(f"阶段划分: {struct.get_ordered_stage_set()}")
            begin_stage_index, begin_frame_index = begin
            final_stage_index, final_frame_index = final

            try:
                begin_frame = struct.get_not_stable_stage_range()[begin_stage_index][begin_frame_index]
                final_frame = struct.get_not_stable_stage_range()[final_stage_index][final_frame_index]
            except (AssertionError, IndexError) as e:
                logger.error(f"[bold #FFC0CB]{e}[/bold #FFC0CB]")
                logger.warning(f"[bold #FFFACD]Analyzer recalculate[/bold #FFFACD] ...")
                begin_frame = struct.get_important_frame_list()[0]
                final_frame = struct.get_important_frame_list()[-1]

            if final_frame.frame_id <= begin_frame.frame_id:
                logger.warning(f"{final_frame} <= {begin_frame}")
                logger.warning(f"[bold #FFFACD]Analyzer recalculate[/bold #FFFACD] ...")
                begin_frame, end_frame = struct.data[0], struct.data[-1]

            time_cost = final_frame.timestamp - begin_frame.timestamp
            logger.info(
                f"分类结果: [开始帧: {begin_frame.timestamp:.5f}] [结束帧: {final_frame.timestamp:.5f}] [总耗时: {time_cost:.5f}]"
            )
            return begin_frame.frame_id, final_frame.frame_id, time_cost

        async def frame_flip():
            target_vision = os.path.join(
                os.path.dirname(vision), f"screen_fps{frate}_{random.randint(100, 999)}.mp4"
            )

            duration = await Switch.ask_video_length(self.fpb, vision)
            vision_start, vision_close, vision_limit = await Switch.ask_magic_point(
                Parser.parse_mills(start),
                Parser.parse_mills(close),
                Parser.parse_mills(limit),
                duration
            )
            vision_start = Parser.parse_times(vision_start)
            vision_close = Parser.parse_times(vision_close)
            vision_limit = Parser.parse_times(vision_limit)
            logger.info(f"视频时长: [{duration}] [{Parser.parse_times(duration)}]")
            logger.info(f"视频帧率: [{frate}]")
            logger.info(f"视频剪辑: start=[{vision_start}] close=[{vision_close}] limit=[{vision_limit}]")

            await Switch.ask_video_change(
                self.fmp, frate, vision, target_vision,
                start=vision_start, close=vision_close, limit=vision_limit
            )
            logger.info(f"视频转换完成: {os.path.basename(target_vision)}")
            os.remove(vision)
            logger.info(f"移除旧的视频: {os.path.basename(vision)}")

            if shape:
                original_shape = await Switch.ask_video_larger(self.fpb, target_vision)
                w, h, ratio = await Switch.ask_magic_frame(original_shape, shape)
                target_shape = w, h
                target_scale = scale
                logger.info(f"调整宽高比: {w} x {h}")
            elif scale:
                target_shape = shape
                target_scale = max(0.1, min(1.0, scale))
            else:
                target_shape = shape
                target_scale = 0.4

            return target_vision, target_shape, target_scale

        async def color_load():
            video.hued_data = tuple(hued_future.result())
            logger.info(f"彩色帧已加载: {video.frame_details(video.hued_data)}")
            hued_thread.shutdown()

        async def frame_hold():
            if struct is None:
                if color:
                    await color_load()
                    return [i for i in video.hued_data]
                return [i for i in video.grey_data]

            frames_list = []
            important_frames = struct.get_important_frame_list()
            pbar = toolbox.show_progress(struct.get_length(), 50)
            if boost:
                frames_list.append(previous := important_frames[0])
                pbar.update(1)
                for current in important_frames[1:]:
                    frames_list.append(current)
                    pbar.update(1)
                    frames_diff = current.frame_id - previous.frame_id
                    if not previous.is_stable() and not current.is_stable() and frames_diff > 1:
                        for specially in struct.data[previous.frame_id: current.frame_id - 1]:
                            frames_list.append(specially)
                            pbar.update(1)
                    previous = current
                pbar.close()
                logger.info(f"获取关键帧: {len(frames_list)}")
            else:
                for current in struct.data:
                    frames_list.append(current)
                    pbar.update(1)
                pbar.close()
                logger.info(f"获取全部帧: {len(frames_list)}")

            if color:
                await color_load()
                return [video.hued_data[frame.frame_id - 1] for frame in frames_list]
            return [frame for frame in frames_list]

        async def frame_flow():

            cutter = VideoCutter()

            size_hook = FrameSizeHook(1.0, None, True)
            cutter.add_hook(size_hook)
            logger.info(
                f"加载视频帧处理单元: {size_hook.__class__.__name__} "
                f"{[size_hook.compress_rate, size_hook.target_size, size_hook.not_grey]}"
            )

            if len(crop_list := crops) > 0 and sum([j for i in crop_list for j in i.values()]) > 0:
                for crop in crop_list:
                    x, y, x_size, y_size = crop.values()
                    crop_hook = PaintCropHook((y_size, x_size), (y, x))
                    cutter.add_hook(crop_hook)
                    logger.info(
                        f"加载视频帧处理单元: {crop_hook.__class__.__name__} "
                        f"{x, y, x_size, y_size}"
                    )

            if len(omit_list := omits) > 0 and sum([j for i in omit_list for j in i.values()]) > 0:
                for omit in omit_list:
                    x, y, x_size, y_size = omit.values()
                    omit_hook = PaintOmitHook((y_size, x_size), (y, x))
                    cutter.add_hook(omit_hook)
                    logger.info(
                        f"加载视频帧处理单元: {omit_hook.__class__.__name__} "
                        f"{x, y, x_size, y_size}"
                    )

            save_hook = FrameSaveHook(extra_path)
            cutter.add_hook(save_hook)
            logger.info(
                f"加载视频帧处理单元: {save_hook.__class__.__name__} "
                f"{[os.path.basename(extra_path)]}"
            )

            logger.info(f"压缩视频: {video.name}")
            logger.info(f"视频帧数: {video.frame_count} 片段数: {video.frame_count - 1} 分辨率: {video.frame_size}")
            cut_start_time = time.time()
            cut_range = cutter.cut(video=video, block=block)
            logger.info(f"压缩完成: {video.name}")
            logger.info(f"压缩耗时: {time.time() - cut_start_time:.2f} 秒")

            stable, unstable = cut_range.get_range(threshold=thres, offset=shift)

            file_list = os.listdir(extra_path)
            file_list.sort(key=lambda n: int(n.split("(")[0]))
            total_images, desired_count = len(file_list), 12

            if total_images <= desired_count:
                retain_indices = range(total_images)
            else:
                retain_indices = [int(i * (total_images / desired_count)) for i in range(desired_count)]
                if len(retain_indices) < desired_count:
                    retain_indices.append(total_images - 1)
                elif len(retain_indices) > desired_count:
                    retain_indices = retain_indices[:desired_count]

            for index, file in enumerate(file_list):
                if index not in retain_indices:
                    os.remove(os.path.join(extra_path, file))

            for draw in os.listdir(extra_path):
                toolbox.draw_line(os.path.join(extra_path, draw))

            struct_start_time = time.time()
            try:
                struct_data = self.kc.classify(
                    video=video, valid_range=stable, keep_data=True
                )
            except AssertionError as e:
                return logger.warning(f"{e}")

            logger.info(f"分类耗时: {time.time() - struct_start_time:.2f} 秒")
            return struct_data

        async def analytics_basic():
            if self.oss == "win32":
                forge_result = await asyncio.gather(
                    *(frame_forge(frame) for frame in frames), return_exceptions=True
                )
            else:
                forge_tasks = [
                    [frame_forge(frame) for frame in chunk] for chunk in
                    [frames[i:i + 100] for i in range(0, len(frames), 100)]
                ]
                forge_list = []
                for ft in forge_tasks:
                    ft_result = await asyncio.gather(*ft, return_exceptions=True)
                    forge_list.extend(ft_result)
                forge_result = tuple(forge_list)

            for result in forge_result:
                if isinstance(result, Exception):
                    logger.error(f"{result}")

            begin_frame, final_frame = frames[0], frames[-1]
            time_cost = final_frame.timestamp - begin_frame.timestamp
            return begin_frame.frame_id, final_frame.frame_id, time_cost, None

        async def analytics_keras():
            if self.oss == "win32":
                flick_result, *forge_result = await asyncio.gather(
                    frame_flick(), *(frame_forge(frame) for frame in frames), return_exceptions=True
                )
            else:
                forge_tasks = [
                    [frame_forge(frame) for frame in chunk] for chunk in
                    [frames[i:i + 100] for i in range(0, len(frames), 100)]
                ]
                flick_task = asyncio.create_task(frame_flick())
                forge_list = []
                for ft in forge_tasks:
                    ft_result = await asyncio.gather(*ft, return_exceptions=True)
                    forge_list.extend(ft_result)
                forge_result = tuple(forge_list)
                flick_result = await flick_task

            for result in forge_result:
                if isinstance(result, Exception):
                    logger.error(f"{result}")

            begin_frame_id, final_frame_id, time_cost = flick_result
            return begin_frame_id, final_frame_id, time_cost, struct

        # Start
        if (target_record := await frame_check()) is None:
            return logger.warning(f"视频文件损坏: {os.path.basename(vision)}")
        logger.info(f"开始加载视频: {os.path.basename(target_record)}")

        movie, shape, scale = await frame_flip()
        video_load_time = time.time()
        video = VideoObject(movie)
        logger.info(f"视频帧长度: {video.frame_count} 分辨率: {video.frame_size}")
        logger.info(f"加载到内存: {video.name}")
        hued_thread, hued_future = video.load_frames(
            load_hued=color, none_gray=False, shape=shape, scale=scale
        )
        logger.info(f"灰度帧已加载: {video.frame_details(video.grey_data)}")
        logger.info(f"视频加载耗时: {time.time() - video_load_time:.2f} 秒")

        struct = await frame_flow() if self.kc else None
        frames = await frame_hold()

        if struct:
            return Review(*(await analytics_keras()))
        return Review(*(await analytics_basic()))


async def achieve(template: typing.Union[str, os.PathLike]) -> str | Exception:
    try:
        async with aiofiles.open(template, "r", encoding=const.CHARSET) as f:
            template_file = await f.read()
    except FileNotFoundError as e:
        return e
    return template_file


async def arithmetic(*args, **kwargs) -> None:

    async def initialization(transfer):
        proc = members if (members := len(transfer)) <= power else power
        rank = "ERROR" if members > 1 else level
        return proc, None, Active.active, (rank,)

    async def multiple_merge(transfer):
        if len(transfer) <= 1:
            return None
        template_total = await achieve(
            missions.view_total_temp if missions.quick else missions.main_total_temp
        )
        logger.info(f"正在合并汇总报告 ...")
        try:
            report_html = await Report.ask_merge_report(results, template_total)
        except FramixReporterError:
            return Show.console.print_exception()
        logger.info(f"合并汇总报告完成 {os.path.relpath(report_html)}")

    missions, platform, cmd_lines, deploy, level, power, loop, *_ = args
    _ = kwargs["total_place"]
    _ = kwargs["model_place"]
    _ = kwargs["model_shape"]
    _ = kwargs["model_aisle"]
    _ = kwargs["fmp"]
    _ = kwargs["fpb"]

    # --video ==========================================================================================================
    if video_list := cmd_lines.video:
        # Start Child Process
        Show.load_animation(cmd_lines)
        with ProcessPoolExecutor(*(await initialization(video_list))) as exe:
            results = await asyncio.gather(
                *(loop.run_in_executor(exe, missions.video_file_task, i, deploy) for i in video_list)
            )
        await multiple_merge(video_list)
        sys.exit(Show.done())

    # --stack ==========================================================================================================
    elif stack_list := cmd_lines.stack:
        # Start Child Process
        Show.load_animation(cmd_lines)
        with ProcessPoolExecutor(*(await initialization(stack_list))) as exe:
            results = await asyncio.gather(
                *(loop.run_in_executor(exe, missions.video_data_task, i, deploy) for i in stack_list)
            )
        await multiple_merge(stack_list)
        sys.exit(Show.done())

    # --train ==========================================================================================================
    elif train_list := cmd_lines.train:
        # Start Child Process
        with ProcessPoolExecutor(*(await initialization(train_list))) as exe:
            results = await asyncio.gather(
                *(loop.run_in_executor(exe, missions.train_model, i, deploy) for i in train_list)
            )
        sys.exit(Show.done())

    # --build ==========================================================================================================
    elif build_list := cmd_lines.build:
        # Start Child Process
        with ProcessPoolExecutor(*(await initialization(build_list))) as exe:
            results = await asyncio.gather(
                *(loop.run_in_executor(exe, missions.build_model, i, deploy) for i in build_list)
            )
        sys.exit(Show.done())

    return None


async def scheduling(*args, **kwargs) -> None:
    missions, platform, cmd_lines, deploy, level, power, loop, *_ = args
    _ = kwargs["total_place"]
    _ = kwargs["model_place"]
    _ = kwargs["model_shape"]
    _ = kwargs["model_aisle"]
    _ = kwargs["fmp"]
    _ = kwargs["fpb"]

    # --flick --carry --fully ==========================================================================================
    if cmd_lines.flick or cmd_lines.carry or cmd_lines.fully:
        await missions.analysis(cmd_lines, platform, deploy, level, power, loop)
    # --paint ==========================================================================================================
    elif cmd_lines.paint:
        await missions.painting(cmd_lines, platform, deploy, level, power, loop)
    # --union ==========================================================================================================
    elif cmd_lines.union:
        await missions.combines_view(cmd_lines.union)
    # --merge ==========================================================================================================
    elif cmd_lines.merge:
        await missions.combines_main(cmd_lines.merge)
    else:
        Show.help_document()


async def main(*args, **kwargs):
    await arithmetic(*args, **kwargs)
    await scheduling(*args, **kwargs)


if __name__ == '__main__':
    _cmd_lines = Parser.parse_cmd()

    Active.active(_level := "DEBUG" if _cmd_lines.debug else "INFO")

    logger.debug(f"操作系统: {_platform}")
    logger.debug(f"应用名称: {_software}")
    logger.debug(f"系统路径: {_sys_symbol}")
    logger.debug(f"环境变量: {_env_symbol}")
    logger.debug(f"工具路径: {_turbo}")
    logger.debug(f"命令参数: {_lines}")
    logger.debug(f"日志等级: {_level}\n")

    logger.debug(f"* 环境 * {'=' * 30}")
    for _env in os.environ["PATH"].split(_env_symbol):
        logger.debug(f"{_env}")
    logger.debug(f"* 环境 * {'=' * 30}\n")

    logger.debug(f"* 工具 * {'=' * 30}")
    for _tls in _tools:
        logger.debug(f"{os.path.basename(_tls):7}: {_tls}")
    logger.debug(f"* 工具 * {'=' * 30}\n")

    logger.debug(f"* 模版 * {'=' * 30}")
    for _tmp in _temps:
        logger.debug(f"Html-Template: {_tmp}")
    logger.debug(f"* 模版 * {'=' * 30}\n")

    _main_loop = asyncio.get_event_loop()

    _flick = _cmd_lines.flick
    _carry = _cmd_lines.carry
    _fully = _cmd_lines.fully

    _quick = _cmd_lines.quick
    _basic = _cmd_lines.basic
    _keras = _cmd_lines.keras

    _alone = _cmd_lines.alone
    _whist = _cmd_lines.whist

    _group = _cmd_lines.group

    _initial_option = os.path.join(_initial_source, "option.json")
    _initial_deploy = os.path.join(_initial_source, "deploy.json")
    _initial_script = os.path.join(_initial_source, "script.json")
    logger.debug(f"配置文件路径: {_initial_option}")
    logger.debug(f"部署文件路径: {_initial_deploy}")
    logger.debug(f"脚本文件路径: {_initial_script}")

    _option = Option(_initial_option)
    _total_place = _option.total_place or _total_place
    _model_place = _option.model_place or _model_place
    _model_shape = _option.model_shape or const.MODEL_SHAPE
    _model_aisle = _option.model_aisle or const.MODEL_AISLE
    logger.debug(f"报告文件路径: {_total_place}")
    logger.debug(f"模型文件路径: {_model_place}")
    logger.debug(f"模型文件尺寸: 宽 {_model_shape[0]} 高 {_model_shape[1]}")
    logger.debug(f"模型文件色彩: {'灰度' if _model_aisle == 1 else '彩色'}模型")

    logger.debug(f"处理器核心数: {(_power := os.cpu_count())}")

    _deploy = Deploy(_initial_deploy)
    for _attr, _attribute in _deploy.deploys.items():
        if any(_line.startswith(f"--{_attr}") for _line in _lines):
            setattr(_deploy, _attr, getattr(_cmd_lines, _attr))
            logger.info(f"Set <{_attr}> = {_attribute} -> {getattr(_deploy, _attr)}")

    _missions = Missions(
        _flick, _carry, _fully, _quick, _basic, _keras, _alone, _whist, _group,
        atom_total_temp=_atom_total_temp,
        main_share_temp=_main_share_temp,
        main_total_temp=_main_total_temp,
        view_share_temp=_view_share_temp,
        view_total_temp=_view_total_temp,
        initial_option=_initial_option,
        initial_deploy=_initial_deploy,
        initial_script=_initial_script,
        total_place=_total_place,
        model_place=_model_place,
        model_shape=_model_shape,
        model_aisle=_model_aisle,
        adb=_adb,
        fmp=_fmp,
        fpb=_fpb,
        scc=_scc,
    )

    from concurrent.futures import ProcessPoolExecutor

    # Main Process =====================================================================================================
    try:
        _main_loop.run_until_complete(
            main(
                _missions, _platform, _cmd_lines, _deploy, _level, _power, _main_loop,
                total_place=_total_place,
                model_place=_model_place,
                model_shape=_model_shape,
                model_aisle=_model_aisle,
                fmp=_fmp,
                fpb=_fpb
            )
        )
    except KeyboardInterrupt:
        sys.exit(Show.exit())
    except (OSError, RuntimeError, TypeError, ValueError, AttributeError):
        Show.console.print_exception()
        sys.exit(Show.fail())
    else:
        sys.exit(Show.done())
    # Main Process =====================================================================================================
