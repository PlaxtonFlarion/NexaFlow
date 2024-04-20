__all__ = []

import os
import sys
import shutil
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
    Show.console.print(f"[bold]software compatible with [bold red]{const.NAME}[/bold red] ...[/bold]")
    Show.simulation_progress(f"Exit after 5 seconds ...", 1, 0.05)
    sys.exit(Show.abnormal_exit())

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
    Show.console.print(f"[bold]{const.NAME} compatible with [bold red]Win & Mac[/bold red] ...[/bold]")
    Show.simulation_progress(f"Exit after 5 seconds ...", 1, 0.05)
    sys.exit(Show.abnormal_exit())

for _tls in (_tools := [_adb, _fmp, _fpb, _scc]):
    os.environ["PATH"] = os.path.dirname(_tls) + _env_symbol + os.environ.get("PATH", "")

for _tls in _tools:
    if not shutil.which((_tls_name := os.path.basename(_tls))):
        Show.console.print(f"[bold]{const.NAME} missing files [bold red]{_tls_name}[/bold red] ...[/bold]")
        Show.simulation_progress(f"Exit after 5 seconds ...", 1, 0.05)
        sys.exit(Show.abnormal_exit())

_atom_total_temp = os.path.join(_workable, "archivix", "pages", "template_atom_total.html")
_main_share_temp = os.path.join(_workable, "archivix", "pages", "template_main_share.html")
_main_total_temp = os.path.join(_workable, "archivix", "pages", "template_main_total.html")
_view_share_temp = os.path.join(_workable, "archivix", "pages", "template_view_share.html")
_view_total_temp = os.path.join(_workable, "archivix", "pages", "template_view_total.html")

for _tmp in (_temps := [_atom_total_temp, _main_share_temp, _main_total_temp, _view_share_temp, _view_total_temp]):
    if not os.path.isfile(_tmp):
        _tmp_name = os.path.basename(_tmp)
        Show.console.print(f"[bold]{const.NAME} missing files [bold red]{_tmp_name}[/bold red] ...[/bold]")
        Show.simulation_progress(f"Exit after 5 seconds ...", 1, 0.05)
        sys.exit(Show.abnormal_exit())

_initial_source = os.path.join(_feasible, f"{const.NAME}.source")

_total_place = os.path.join(_feasible, f"{const.NAME}.report")
_model_place = os.path.join(_workable, "archivix", "molds", "Keras_Gray_W256_H256_00000.h5")
_model_shape = const.MODEL_SHAPE
_model_aisle = const.MODEL_AISLE

if len(sys.argv) == 1:
    Show.help_document()
    sys.exit(Show.normal_exit())

_attrs = [
    "boost", "color", "shape", "scale",
    "start", "close", "limit", "begin", "final",
    "frate", "thres", "shift", "block", "crops", "omits"
]
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
    from loguru import logger
    from rich.prompt import Prompt
# frameflow & nexaflow =================================================================================================
    from engine.manage import Manage
    from engine.switch import Switch
    from engine.terminal import Terminal
    from engine.active import Active, Review
    from frameflow.skills.config import Option
    from frameflow.skills.config import Deploy
    from frameflow.skills.config import Script
    from frameflow.skills.datagram import DataGram
    from frameflow.skills.parser import Parser
    from nexaflow import toolbox
    from nexaflow.report import Report
    from nexaflow.video import VideoObject, VideoFrame
    from nexaflow.cutter.cutter import VideoCutter
    from nexaflow.hook import CompressHook, FrameSaveHook
    from nexaflow.hook import PaintCropHook, PaintOmitHook
    from nexaflow.classifier.keras_classifier import KerasStruct
except (ImportError, RuntimeError, ModuleNotFoundError) as _e:
    Show.console.print(f"[bold red]Error: {_e}")
    Show.simulation_progress(f"Exit after 5 seconds ...", 1, 0.05)
    sys.exit(Show.abnormal_exit())


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
        with DataGram(os.path.join(r.reset_path, f"{const.NAME}_data.db")) as database:
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
            logger.info(f"start=[{vision_start}] - close=[{vision_close}] - limit=[{vision_limit}]")

            loop.run_until_complete(
                Switch.ask_video_detach(
                    self.fmp, video_filter_list, new_video_path, reporter.frame_path,
                    start=vision_start, close=vision_close, limit=vision_limit
                )
            )

            header = str((
                os.path.basename(reporter.total_path), reporter.title,
                Path(reporter.query).name if self.group and len(Path(reporter.query).parts) == 2 else ""
            ))
            result = {
                header: {
                    "query": reporter.query,
                    "stage": {"start": 0, "end": 0, "cost": 0},
                    "frame": os.path.basename(reporter.frame_path),
                    "style": "quick",
                }
            }
            logger.debug(f"Quicker: {json.dumps(result, ensure_ascii=False)}")
            loop.run_until_complete(reporter.load(result))

            loop.run_until_complete(
                reporter.ask_create_total_report(
                    os.path.dirname(reporter.total_path),
                    self.group,
                    loop.run_until_complete(achieve(self.view_share_temp)),
                    loop.run_until_complete(achieve(self.view_total_temp))
                )
            )
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
        start, end, cost, struct = futures.data

        header = str((
            os.path.basename(reporter.total_path), reporter.title,
            Path(reporter.query).name if self.group and len(Path(reporter.query).parts) == 2 else ""
        ))
        result = {
            header: {
                "query": reporter.query,
                "stage": {"start": start, "end": end, "cost": f"{cost:.5f}"},
                "frame": os.path.basename(reporter.frame_path)
            }
        }

        if struct:
            if isinstance(
                    tmp := loop.run_until_complete(achieve(self.atom_total_temp)), Exception
            ):
                return Show.console.print(f"[bold red]{tmp}")
            logger.info(f"模版引擎正在渲染 ...")
            original_inform = loop.run_until_complete(
                reporter.ask_draw(struct, reporter.proto_path, tmp)
            )
            logger.info(f"模版引擎渲染完毕 ...")
            result[header]["extra"] = os.path.basename(reporter.extra_path)
            result[header]["proto"] = os.path.basename(original_inform)
            result[header]["style"] = "keras"
        else:
            result[header]["style"] = "basic"

        logger.debug(f"Restore: {json.dumps(result, ensure_ascii=False)}")
        loop.run_until_complete(reporter.load(result))

        self.enforce(reporter, struct, start, end, cost)

        loop.run_until_complete(
            reporter.ask_create_total_report(
                os.path.dirname(reporter.total_path),
                self.group,
                loop.run_until_complete(achieve(self.main_share_temp)),
                loop.run_until_complete(achieve(self.main_total_temp))
            )
        )
        return reporter.total_path

    # """Child Process"""
    def video_data_task(self, video_data: str, deploy: "Deploy"):
        reporter = Report(self.total_place)

        loop = asyncio.get_event_loop()

        if self.quick:
            logger.debug(f"★★★ 快速模式 ★★★")
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
                    logger.info(f"start=[{vision_start}] - close=[{vision_close}] - limit=[{vision_limit}]")

                    loop.run_until_complete(
                        Switch.ask_video_detach(
                            self.fmp, video_filter_list, new_video_path, reporter.frame_path,
                            start=deploy.start, close=deploy.close, limit=deploy.limit
                        )
                    )

                    header = str((
                        os.path.basename(reporter.total_path), reporter.title,
                        Path(reporter.query).name if self.group and len(Path(reporter.query).parts) == 2 else ""
                    ))
                    result = {
                        header: {
                            "query": reporter.query,
                            "stage": {"start": 0, "end": 0, "cost": 0},
                            "frame": os.path.basename(reporter.frame_path),
                            "style": "quick"
                        }
                    }
                    logger.debug(f"Quicker: {json.dumps(result, ensure_ascii=False)}")
                    loop.run_until_complete(reporter.load(result))

            loop.run_until_complete(
                reporter.ask_create_total_report(
                    os.path.dirname(reporter.total_path),
                    self.group,
                    loop.run_until_complete(achieve(self.view_share_temp)),
                    loop.run_until_complete(achieve(self.view_total_temp))
                )
            )
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
                start, end, cost, struct = futures.data

                header = str((
                    os.path.basename(reporter.total_path), reporter.title,
                    Path(reporter.query).name if self.group and len(Path(reporter.query).parts) == 2 else ""
                ))
                result = {
                    header: {
                        "query": reporter.query,
                        "stage": {"start": start, "end": end, "cost": f"{cost:.5f}"},
                        "frame": os.path.basename(reporter.frame_path)
                    }
                }

                if struct:
                    if isinstance(
                            tmp := loop.run_until_complete(achieve(self.atom_total_temp)), Exception
                    ):
                        return Show.console.print(f"[bold red]{tmp}")
                    logger.info(f"模版引擎正在渲染 ...")
                    original_inform = loop.run_until_complete(
                        reporter.ask_draw(struct, reporter.proto_path, tmp)
                    )
                    logger.info(f"模版引擎渲染完毕 ...")
                    result[header]["extra"] = os.path.basename(reporter.extra_path)
                    result[header]["proto"] = os.path.basename(original_inform)
                    result[header]["style"] = "keras"
                else:
                    result[header]["style"] = "basic"

                logger.debug(f"Restore: {json.dumps(result, ensure_ascii=False)}")
                loop.run_until_complete(reporter.load(result))

                self.enforce(reporter, struct, start, end, cost)

        loop.run_until_complete(
            reporter.ask_create_total_report(
                os.path.dirname(reporter.total_path),
                self.group,
                loop.run_until_complete(achieve(self.main_share_temp)),
                loop.run_until_complete(achieve(self.main_total_temp))
            )
        )
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
        logger.info(f"start=[{vision_start}] - close=[{vision_close}] - limit=[{vision_limit}]")

        asyncio.run(
            Switch.ask_video_change(
                self.fmp, deploy.frate, video_file, video_temp_file,
                start=vision_start, close=vision_close, limit=vision_limit
            )
        )

        video = VideoObject(video_temp_file)
        video.load_frames(
            load_hued=False, none_gray=True
        )

        cutter = VideoCutter()
        res = cutter.cut(
            video=video, block=deploy.block
        )
        stable, unstable = res.get_range(
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
        state_list = await asyncio.gather(
            *(Report.ask_create_total_report(m, self.group, major, total) for m in merge)
        )
        for state in state_list:
            if isinstance(state, Exception):
                logger.error(state)

    async def combines_view(self, merge: list):
        views, total = await asyncio.gather(
            achieve(self.view_share_temp), achieve(self.view_total_temp),
            return_exceptions=True
        )
        state_list = await asyncio.gather(
            *(Report.ask_create_total_report(m, self.group, views, total) for m in merge)
        )
        for state in state_list:
            if isinstance(state, Exception):
                logger.error(state)

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
                    Show.console.print(f"[bold]保存图片: {[img_save_path]}")
                break
            elif action.strip().upper() == "N":
                break
            else:
                Show.console.print(f"[bold][bold red]没有该选项,请重新输入[/bold red] ...[/bold]\n")

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

            fmt_dir = time.strftime("%Y%m%d%H%M%S") if self.quick or self.basic or self.keras else None
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
                logger.debug(f"★★★ 快速模式 ★★★")
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
                    logger.info(f"start=[{vision_start}] - close=[{vision_close}] - limit=[{vision_limit}]")

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
                    header = str((
                        os.path.basename(reporter.total_path), reporter.title,
                        Path(reporter.query).name if self.group and len(Path(reporter.query).parts) == 2 else ""
                    ))
                    result = {
                        header: {
                            "query": reporter.query,
                            "stage": {"start": 0, "end": 0, "cost": 0},
                            "frame": os.path.basename(reporter.frame_path),
                            "style": "quick"
                        }
                    }
                    logger.debug(f"Quicker: {json.dumps(result, ensure_ascii=False)}")
                    await reporter.load(result)

            elif self.basic or self.keras:
                logger.debug(f"★★★ {'智能模式' if alynex.kc else '基础模式'} ★★★")

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

                    start, end, cost, struct = future.data
                    *_, total_path, title, query_path, query, frame_path, extra_path, proto_path = todo

                    header = str((
                        os.path.basename(total_path), title,
                        Path(query).name if self.group and len(Path(query).parts) == 2 else ""
                    ))
                    result = {
                        header: {
                            "query": query,
                            "stage": {"start": start, "end": end, "cost": f"{cost:.5f}"},
                            "frame": os.path.basename(frame_path)
                        }
                    }

                    if struct:
                        if isinstance(
                                tmp := await achieve(self.atom_total_temp), Exception
                        ):
                            return Show.console.print(f"[bold red]{tmp}")
                        logger.info(f"模版引擎正在渲染 ...")
                        original_inform = await reporter.ask_draw(struct, proto_path, tmp)
                        logger.info(f"模版引擎渲染完毕 ...")
                        result[header]["extra"] = os.path.basename(extra_path)
                        result[header]["proto"] = os.path.basename(original_inform)
                        result[header]["style"] = "keras"
                    else:
                        result[header]["style"] = "basic"

                    logger.debug(f"Restore: {json.dumps(result, ensure_ascii=False)}")
                    await reporter.load(result)

                    self.enforce(reporter, struct, start, end, cost)

            else:
                return logger.debug(f"{const.DESC} Analyzer: 录制模式 ...")

        async def device_mode_view():
            Show.console.print(f"[bold]<Link> <{'单设备模式' if len(device_list) == 1 else '多设备模式'}>")
            for device in device_list:
                Show.console.print(f"[bold #00FFAF]Connect:[/bold #00FFAF] {device}")

        async def all_time(style):
            if style == "less":
                await asyncio.gather(
                    *(record.timing_less(serial, events, timer_mode) for serial, events in record.device_events.items())
                )
            elif style == "many":
                await asyncio.gather(
                    *(record.timing_many(serial, events) for serial, events in record.device_events.items())
                )

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

            effective_list = await asyncio.gather(
                *(record.close_record(video_temp, transports, events)
                  for (_, events), (video_temp, transports, *_) in zip(record.device_events.items(), task_list))
            )
            for (idx, (effective, video_name)), _ in zip(enumerate(effective_list), task_list):
                if effective.startswith("视频录制失败"):
                    task_list.pop(idx)
                logger.info(f"{effective}: {video_name} ...")

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

        async def combines_report():
            if len(reporter.range_list) == 0:
                return False
            combines = getattr(self, "combines_view" if self.quick else "combines_main")
            await combines([os.path.dirname(reporter.total_path)])
            return True

        async def load_commands(script):
            try:
                async with aiofiles.open(script, "r", encoding="utf-8") as f:
                    file = await f.read()
                    exec_dict = {
                        cmds["name"]: {
                            "loop": cmds["loop"], "actions": cmds["actions"]
                        } for cmds in json.loads(file)["commands"]
                    }
            except FileNotFoundError as e:
                Script.dump_script(script)
                return e
            except (KeyError, json.JSONDecodeError) as e:
                return e
            return exec_dict

        async def exec_commands():

            async def is_function(func_name, *func_args):
                for func_arg in func_args:
                    if callable(is_func := getattr(func_arg, func_name, None)):
                        return is_func
                return None

            for device_action in device_action_list:
                if not (device_cmds := device_action["command"]):
                    logger.error(f"No order found {device_cmds} ...")
                    continue

                for device_func in (device_func_list := await asyncio.gather(
                        *(is_function(device_cmds, device, player) for device in device_list)
                )):
                    if device_func is None:
                        logger.error(f"There is no such command {device_cmds} ...")
                        break
                    if device_func.__name__ == "audio_player":
                        device_func_list = [device_func_list[0]]
                        break

                if not (method_args := device_action.get("args", None)):
                    continue

                yield [dynamically(device_func, method_args, None if len(device_func_list) == 1 else device.serial)
                       for device_func, device in zip(device_func_list, device_list) if device_func]

        async def dynamically(function, arg_list, device_sn=None):
            logger.info(
                f"{device_sn if device_sn else 'Device'} {function.__name__} {arg_list}"
            )
            try:
                if inspect.iscoroutinefunction(function):
                    await function(*arg_list)
                else:
                    await asyncio.to_thread(function, *arg_list)
            except Exception as e:
                logger.error(f"{e}")

        # Initialization ===============================================================================================
        cmd_lines, platform, deploy, level, power, loop, *_ = args

        manage = Manage(self.adb)
        device_list = await manage.operate_device()

        titles = {"quick": "Quick", "basic": "Basic", "keras": "Keras"}
        input_title = next((title for key, title in titles.items() if getattr(self, key)), "Video")
        reporter = Report(self.total_place)

        if self.keras and not self.quick and not self.basic:
            attack = self.total_place, self.model_place, self.model_shape, self.model_aisle
        else:
            attack = self.total_place, None, None, None

        charge = platform, self.fmp, self.fpb

        alynex = Alynex(*attack, *charge)
        Show.load_animation(cmd_lines)

        from engine.record import Record, Player
        record = Record(platform, self.scc, self.alone, self.whist)
        player = Player()

        # Initialization ===============================================================================================

        # Flick Loop ===================================================================================================
        if self.flick:
            const_title = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
            reporter.title = f"{input_title}_{const_title}"
            timer_mode = 5
            while True:
                try:
                    await device_mode_view()
                    start_tips = f"<<<按 Enter 开始 [bold #D7FF5F]{timer_mode}[/bold #D7FF5F] 秒>>>"
                    if action := Prompt.ask(prompt=f"[bold #5FD7FF]{start_tips}[/bold #5FD7FF]", console=Show.console):
                        if (select := action.strip().lower()) == "serial":
                            device_list = await manage.operate_device()
                            continue
                        elif "header" in select:
                            if match := re.search(r"(?<=header\s).*", select):
                                if hd := match.group().strip():
                                    src_hd, a, b = f"{input_title}_{time.strftime('%Y%m%d_%H%M%S')}", 10000, 99999
                                    logger.success("新标题设置成功 ...")
                                    reporter.title = f"{src_hd}_{hd}" if hd else f"{src_hd}_{random.randint(a, b)}"
                                    continue
                            raise ValueError
                        elif select in ["invent", "create"]:
                            if await combines_report():
                                break
                            Show.console.print(f"[bold red]没有可以生成的报告 ...\n")
                            continue
                        elif select == "deploy":
                            Show.console.print("[bold yellow]修改 deploy.json 文件后请完全退出编辑器进程再继续操作 ...")
                            deploy.dump_deploy(self.initial_deploy)
                            first = ["Notepad"] if platform == "win32" else ["open", "-W", "-a", "TextEdit"]
                            first.append(self.initial_deploy)
                            await Terminal.cmd_line(*first)
                            deploy.load_deploy(self.initial_deploy)
                            deploy.view_deploy()
                            continue
                        elif select.isdigit():
                            timer_value, lower_bound, upper_bound = int(select), 5, 300
                            if timer_value > 300 or timer_value < 5:
                                bound_tips = f"{lower_bound} <= [bold #FFD7AF]Time[/bold #FFD7AF] <= {upper_bound}"
                                Show.console.print(f"[bold #FFFF87]{bound_tips}[/bold #FFFF87]")
                            timer_mode = max(lower_bound, min(upper_bound, timer_value))
                        else:
                            raise ValueError
                except ValueError:
                    Show.tips_document()
                    continue
                else:
                    task_list = await commence()
                    await all_time("less")
                    await all_over()
                    await analysis_tactics()
                    check = await record.event_check()
                    device_list = await manage.operate_device() if check else device_list
                finally:
                    await record.clean_check()
        # Flick Loop ===================================================================================================

        # Other Loop ===================================================================================================
        elif self.carry or self.fully:

            if self.carry:
                if isinstance(script_data := await load_commands(self.initial_script), Exception):
                    return logger.error(f"{script_data}")
                try:
                    script_storage = [{carry: script_data[carry] for carry in list(set(self.carry))}]
                except KeyError as err:
                    return logger.error(f"{err}")

            else:
                load_result = await asyncio.gather(
                    *(load_commands(fully) for fully in self.fully), return_exceptions=True
                )
                if len(script_data := [i for i in load_result if not isinstance(i, Exception)]) == 0:
                    return logger.error(f"缺少有效的脚本文件 {' '.join(self.fully)}")
                script_storage = script_data

            await device_mode_view()
            for script_dict in script_storage:
                for key, value in script_dict.items():
                    reporter.title = f"{key.replace(' ', '')}_{input_title}"
                    logger.info(f"Exec: {key}")
                    for _ in range(value["loop"]):
                        try:
                            task_list = await commence()

                            if device_action_list := value.get("actions", None):
                                async for exec_func_list in exec_commands():
                                    if len(exec_func_list) == 0:
                                        continue
                                    for exec_func in await asyncio.gather(*exec_func_list, return_exceptions=True):
                                        if isinstance(exec_func, Exception):
                                            logger.error(f"{exec_func}")

                            await all_time("many")
                            await all_over()
                            await analysis_tactics()
                            check = await record.event_check()
                            device_list = await manage.operate_device() if check else device_list
                        finally:
                            await record.clean_check()

            if await combines_report():
                return True
            return Show.console.print(f"[bold red]没有可以生成的报告 ...\n")
        # Other Loop ===================================================================================================

        return None


class Alynex(object):

    __kc: typing.Optional["KerasStruct"] = None

    def __init__(
            self,
            total_place: typing.Optional[typing.Union[str, os.PathLike]],
            model_place: typing.Optional[typing.Union[str, os.PathLike]],
            model_shape: typing.Optional[tuple],
            model_aisle: typing.Optional[int],
            *args,
            **__
    ):

        if model_place and model_shape and model_aisle:
            try:
                self.kc = KerasStruct(data_size=model_shape, aisle=model_aisle)
                self.kc.load_model(model_place)
            except ValueError as err:
                logger.error(f"{err}")
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
            self, vision: typing.Union[str, os.PathLike], deploy: "Deploy" = None, *args, **kwargs
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
                    target_screen = Path(vision)
                screen.release()
            elif os.path.isdir(vision):
                file_list = [
                    file for file in os.listdir(vision) if os.path.isfile(os.path.join(vision, file))
                ]
                if len(file_list) >= 1:
                    screen = cv2.VideoCapture(open_file := os.path.join(vision, file_list[0]))
                    if screen.isOpened():
                        target_screen = Path(open_file)
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
            except AssertionError as e:
                logger.warning(f"{e}")
                begin_frame = struct.get_important_frame_list()[0]
                final_frame = struct.get_important_frame_list()[-1]
                logger.warning(f"{const.DESC} Analyzer recalculate ...")
            except IndexError as e:
                logger.warning(f"{e}")
                for i, unstable_stage in enumerate(struct.get_specific_stage_range("-3")):
                    Show.console.print(f"[bold]第 {i:02} 个非稳定阶段")
                    Show.console.print(f"[bold]{'=' * 30}")
                    for j, frame in enumerate(unstable_stage):
                        Show.console.print(f"[bold]第 {j:05} 帧: {frame}")
                    Show.console.print(f"[bold]{'=' * 30}\n")
                begin_frame = struct.get_important_frame_list()[0]
                final_frame = struct.get_important_frame_list()[-1]
                logger.warning(f"{const.DESC} Analyzer recalculate ...")

            if final_frame.frame_id <= begin_frame.frame_id:
                logger.warning(f"{final_frame} <= {begin_frame}")
                begin_frame, end_frame = struct.data[0], struct.data[-1]
                logger.warning(f"{const.DESC} Analyzer recalculate ...")

            time_cost = final_frame.timestamp - begin_frame.timestamp
            logger.info(
                f"图像分类结果: [开始帧: {begin_frame.timestamp:.5f}] [结束帧: {final_frame.timestamp:.5f}] [总耗时: {time_cost:.5f}]"
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
            logger.info(f"start=[{vision_start}] - close=[{vision_close}] - limit=[{vision_limit}]")

            await Switch.ask_video_change(
                self.fmp, frate, vision, target_vision,
                start=vision_start, close=vision_close, limit=vision_limit
            )
            logger.info(f"视频转换完成: {Path(target_vision).name}")
            os.remove(vision)
            logger.info(f"移除旧的视频: {Path(vision).name}")

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

        async def frame_hold():
            if struct is None:
                if color:
                    video.hued_data = tuple(hued_data.result())
                    logger.info(f"彩色帧已加载: {video.frame_details(video.hued_data)}")
                    hued_task.shutdown()
                    return [i for i in video.hued_data]
                return [i for i in video.grey_data]

            important_frames = struct.get_important_frame_list()
            pbar = toolbox.show_progress(struct.get_length(), 50, "Faster")
            frames_list = []
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
            else:
                for current in struct.data:
                    frames_list.append(current)
                    pbar.update(1)
                pbar.close()

            if color:
                video.hued_data = tuple(hued_data.result())
                logger.info(f"彩色帧已加载: {video.frame_details(video.hued_data)}")
                hued_task.shutdown()
                return [video.hued_data[frame.frame_id - 1] for frame in frames_list]
            return [frame for frame in frames_list]

        async def frame_flow():

            cutter = VideoCutter()

            if len(crop_list := crops) > 0 and sum([j for i in crop_list for j in i.values()]) > 0:
                for crop in crop_list:
                    x, y, x_size, y_size = crop.values()
                    crop_hook = PaintCropHook((y_size, x_size), (y, x))
                    cutter.add_hook(crop_hook)
                    logger.debug(f"{crop_hook.__class__.__name__}: {x, y, x_size, y_size}")

            if len(omit_list := omits) > 0 and sum([j for i in omit_list for j in i.values()]) > 0:
                for omit in omit_list:
                    x, y, x_size, y_size = omit.values()
                    omit_hook = PaintOmitHook((y_size, x_size), (y, x))
                    cutter.add_hook(omit_hook)
                    logger.debug(f"{omit_hook.__class__.__name__}: {x, y, x_size, y_size}")

            save_hook = FrameSaveHook(extra_path)
            cutter.add_hook(save_hook)

            cut_range = cutter.cut(video=video, block=block)

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

            try:
                return self.kc.classify(video=video, valid_range=stable, keep_data=True)
            except AssertionError as e:
                return logger.warning(f"{e}")

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
                    logger.error(f"Error: {result}")

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
                    logger.error(f"Error: {result}")

            begin_frame_id, final_frame_id, time_cost = flick_result
            return begin_frame_id, final_frame_id, time_cost, struct

        if (target_record := await frame_check()) is None:
            return logger.error(f"{vision} 不是一个标准的视频文件或视频文件已损坏 ...")
        logger.info(f"{target_record.name} 可正常播放，准备加载视频 ...")

        movie, shape, scale = await frame_flip()
        video = VideoObject(movie)
        hued_task, hued_data = video.load_frames(
            load_hued=color, none_gray=False, shape=shape, scale=scale
        )

        struct = await frame_flow() if self.kc else None
        frames = await frame_hold()

        if struct:
            return Review(*(await analytics_keras()))
        return Review(*(await analytics_basic()))


async def achieve(template_path: str) -> str | Exception:
    try:
        async with aiofiles.open(template_path, "r", encoding="utf-8") as f:
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
        await Report.ask_merge_report(results, template_total)

    missions, platform, cmd_lines, deploy, level, power, loop, *_ = args
    total_place = kwargs["total_place"]
    model_place = kwargs["model_place"]
    model_shape = kwargs["model_shape"]
    model_aisle = kwargs["model_aisle"]
    adb = kwargs["adb"]
    fmp = kwargs["fmp"]
    fpb = kwargs["fpb"]
    scc = kwargs["scc"]

    # --video ==========================================================================================================
    if video_list := cmd_lines.video:
        # Start Child Process
        Show.load_animation(cmd_lines)
        with ProcessPoolExecutor(*(await initialization(video_list))) as exe:
            results = await asyncio.gather(
                *(loop.run_in_executor(exe, missions.video_file_task, i, deploy) for i in video_list)
            )
        await multiple_merge(video_list)
        sys.exit(Show.normal_exit())

    # --stack ==========================================================================================================
    elif stack_list := cmd_lines.stack:
        # Start Child Process
        Show.load_animation(cmd_lines)
        with ProcessPoolExecutor(*(await initialization(stack_list))) as exe:
            results = await asyncio.gather(
                *(loop.run_in_executor(exe, missions.video_data_task, i, deploy) for i in stack_list)
            )
        await multiple_merge(stack_list)
        sys.exit(Show.normal_exit())

    # --train ==========================================================================================================
    elif train_list := cmd_lines.train:
        # Start Child Process
        with ProcessPoolExecutor(*(await initialization(train_list))) as exe:
            results = await asyncio.gather(
                *(loop.run_in_executor(exe, missions.train_model, i, deploy) for i in train_list)
            )
        sys.exit(Show.normal_exit())

    # --build ==========================================================================================================
    elif build_list := cmd_lines.build:
        # Start Child Process
        with ProcessPoolExecutor(*(await initialization(build_list))) as exe:
            results = await asyncio.gather(
                *(loop.run_in_executor(exe, missions.build_model, i, deploy) for i in build_list)
            )
        sys.exit(Show.normal_exit())

    return None


async def scheduling(*args, **kwargs) -> None:
    missions, platform, cmd_lines, deploy, level, power, loop, *_ = args
    total_place = kwargs["total_place"]
    model_place = kwargs["model_place"]
    model_shape = kwargs["model_shape"]
    model_aisle = kwargs["model_aisle"]
    adb = kwargs["adb"]
    fmp = kwargs["fmp"]
    fpb = kwargs["fpb"]
    scc = kwargs["scc"]

    # --flick --carry --fully ==========================================================================================
    if cmd_lines.flick or cmd_lines.carry or cmd_lines.fully:
        await missions.analysis(cmd_lines, deploy, level, power, loop)
    # --paint ==========================================================================================================
    elif cmd_lines.paint:
        await missions.painting(cmd_lines, deploy, level, power, loop)
    # --union ==========================================================================================================
    elif cmd_lines.union:
        await missions.combines_view(cmd_lines.union)
    # --merge ==========================================================================================================
    elif cmd_lines.merge:
        await missions.combines_main(cmd_lines.merge)
    else:
        Show.help_document()


if __name__ == '__main__':
    _cmd_lines = Parser.parse_cmd()

    Active.active(_level := "DEBUG" if _cmd_lines.debug else "INFO")

    # Debug Mode =======================================================================================================
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
    # Debug Mode =======================================================================================================

    _flick = _cmd_lines.flick
    _carry = _cmd_lines.carry
    _fully = _cmd_lines.fully

    _quick = _cmd_lines.quick
    _basic = _cmd_lines.basic
    _keras = _cmd_lines.keras

    _alone = _cmd_lines.alone
    _whist = _cmd_lines.whist

    _group = _cmd_lines.group

    _boost = _cmd_lines.boost
    _color = _cmd_lines.color
    _shape = _cmd_lines.shape
    _scale = _cmd_lines.scale
    _start = _cmd_lines.start
    _close = _cmd_lines.close
    _limit = _cmd_lines.limit
    _begin = _cmd_lines.begin
    _final = _cmd_lines.final
    _frate = _cmd_lines.frate
    _thres = _cmd_lines.thres
    _shift = _cmd_lines.shift
    _block = _cmd_lines.block

    _crops = []
    if _cmd_lines.crops:
        for _hook in _cmd_lines.crops:
            if len(match_list := re.findall(r"-?\d*\.?\d+", _hook)) == 4:
                _valid_list = [
                    float(num) if "." in num else int(num) for num in match_list
                ]
                if len(_valid_list) == 4 and sum(_valid_list) > 0:
                    _valid_dict = {
                        _k: _v for _k, _v in zip(["x", "y", "x_size", "y_size"], _valid_list)
                    }
                    _crops.append(_valid_dict)
    _unique_crops = {tuple(_i.items()) for _i in _crops}
    _crops = [dict(_i) for _i in _unique_crops]

    _omits = []
    if _cmd_lines.omits:
        for _hook in _cmd_lines.omits:
            if len(match_list := re.findall(r"-?\d*\.?\d+", _hook)) == 4:
                _valid_list = [
                    float(num) if "." in num else int(num) for num in match_list
                ]
                if len(_valid_list) == 4 and sum(_valid_list) > 0:
                    _valid_dict = {
                        _k: _v for _k, _v in zip(["x", "y", "x_size", "y_size"], _valid_list)
                    }
                    _omits.append(_valid_dict)
    _unique_omits = {tuple(_i.items()) for _i in _omits}
    _omits = [dict(_i) for _i in _unique_omits]

    _initial_option = os.path.join(_initial_source, "option.json")
    _initial_deploy = os.path.join(_initial_source, "deploy.json")
    _initial_script = os.path.join(_initial_source, "script.json")
    logger.debug(f"配置文件路径: {_initial_option}")
    logger.debug(f"部署文件路径: {_initial_deploy}")
    logger.debug(f"脚本文件路径: {_initial_script}")

    _option = Option(_initial_option)
    _total_place = _option.total_place or _total_place
    _model_place = _option.model_place or _model_place
    _model_shape = _option.model_shape or _model_shape
    _model_aisle = _option.model_aisle or _model_aisle
    logger.debug(f"报告文件路径: {_total_place}")
    logger.debug(f"模型文件路径: {_model_place}")
    logger.debug(f"模型文件尺寸: 宽 {_model_shape[0]} 高 {_model_shape[1]}")
    logger.debug(f"模型文件色彩: {'灰度' if _model_aisle == 1 else '彩色'}模型")

    logger.debug(f"处理器核心数: {(_power := os.cpu_count())}")

    _deploy = Deploy(_initial_deploy)
    for _attr in _attrs:
        if any(_line.startswith(f"--{_attr}") for _line in _lines):
            logger.debug(f"Set {_attr} = {(_attribute := getattr(_cmd_lines, _attr))}")
            setattr(_deploy, _attr, _attribute)

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

    _main_loop = asyncio.get_event_loop()

    # Main Process =====================================================================================================
    try:
        _main_loop.run_until_complete(
            arithmetic(
                _missions, _platform, _cmd_lines, _deploy, _level, _power, _main_loop,
                total_place=_total_place,
                model_place=_model_place,
                model_shape=_model_shape,
                model_aisle=_model_aisle,
                adb=_adb,
                fmp=_fmp,
                fpb=_fpb,
                scc=_scc,
            )
        )
        _main_loop.run_until_complete(
            scheduling(
                _missions, _platform, _cmd_lines, _deploy, _level, _power, _main_loop,
                total_place=_total_place,
                model_place=_model_place,
                model_shape=_model_shape,
                model_aisle=_model_aisle,
                adb=_adb,
                fmp=_fmp,
                fpb=_fpb,
                scc=_scc,
            )
        )
    except KeyboardInterrupt:
        _main_loop.close()
        sys.exit(Show.normal_exit())
    finally:
        _main_loop.close()
        sys.exit(Show.normal_exit())
    # Main Process =====================================================================================================
