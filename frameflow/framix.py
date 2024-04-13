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
    sys.exit(1)

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
    sys.exit(1)

for _tls in (_tools := [_adb, _fmp, _fpb, _scc]):
    os.environ["PATH"] = os.path.dirname(_tls) + _env_symbol + os.environ.get("PATH", "")

for _tls in _tools:
    if not shutil.which((_tls_name := os.path.basename(_tls))):
        Show.console.print(f"[bold]{const.NAME} missing files [bold red]{_tls_name}[/bold red] ...[/bold]")
        Show.simulation_progress(f"Exit after 5 seconds ...", 1, 0.05)
        sys.exit(1)

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
        sys.exit(1)

_initial_source = os.path.join(_feasible, f"{const.NAME}.source")

_total_place = os.path.join(_feasible, f"{const.NAME}.report")
_model_place = os.path.join(_workable, "archivix", "molds", "Keras_Gray_W256_H256_00000.h5")
_model_shape = const.MODEL_SHAPE
_model_aisle = const.MODEL_AISLE

if len(sys.argv) == 1:
    Show.help_document()
    sys.exit(0)

_attrs = [
    "alone", "group", "boost", "color", "shape", "scale",
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
    import inspect
    import asyncio
    import aiofiles
    import datetime
    from pathlib import Path
    from loguru import logger
    from rich.prompt import Prompt
# frameflow & nexaflow =================================================================================================
    from engine.switch import Switch
    from engine.terminal import Terminal
    from engine.activate import Active, Review
    from frameflow.skills.manage import Manage
    from frameflow.skills.parser import Parser
    from frameflow.skills.configure import Option
    from frameflow.skills.configure import Deploy
    from frameflow.skills.configure import Script
    from frameflow.skills.database import DataBase
    from nexaflow import toolbox
    from nexaflow.report import Report
    from nexaflow.video import VideoObject, VideoFrame
    from nexaflow.cutter.cutter import VideoCutter
    from nexaflow.hook import CompressHook, FrameSaveHook
    from nexaflow.hook import PaintCropHook, PaintOmitHook
    from nexaflow.classifier.keras_classifier import KerasClassifier
    from nexaflow.classifier.framix_classifier import FramixClassifier
except (RuntimeError, ModuleNotFoundError) as _error:
    Show.console.print(f"[bold red]Error: {_error}")
    Show.simulation_progress(f"Exit after 5 seconds ...", 1, 0.05)
    sys.exit(1)


class Mission(object):

    def __init__(self, *args, **kwargs):
        self.carry, self.fully, self.quick, self.basic, self.keras, *_ = args
        _, _, _, _, _, self.alone, self.group, self.boost, self.color, self.shape, self.scale, *_ = args
        *_, self.start, self.close, self.limit, self.begin, self.final, _, _, _, _, _, _ = args
        *_, self.frate, self.thres, self.shift, self.block, self.crops, self.omits = args

        self.attrs = kwargs["attrs"]
        self.lines = kwargs["lines"]
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

    @staticmethod
    def enforce(r, c, start: int, end: int, cost: float):
        with DataBase(os.path.join(r.reset_path, f"{const.NAME}_data.db")) as database:
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

    @staticmethod
    def amazing(vision, deploy, kc, *args):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_complete = loop.run_until_complete(
            Core.ask_analyzer(vision, deploy, kc, *args)
        )
        loop.close()
        return loop_complete

    def video_task(self, video_file: str):
        reporter = Report(self.total_place)
        reporter.title = f"{const.DESC}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        reporter.query = time.strftime('%Y%m%d%H%M%S')
        new_video_path = os.path.join(reporter.video_path, os.path.basename(video_file))

        shutil.copy(video_file, new_video_path)

        deploy = Deploy(self.initial_deploy)
        for attr in self.attrs:
            if any(line.startswith(f"--{attr}") for line in self.lines):
                logger.debug(f"Set {attr} = {(attribute := getattr(self, attr))}")
                setattr(deploy, attr, attribute)

        loop = asyncio.get_event_loop()

        if self.quick:
            logger.info(f"{const.DESC} Analyzer: 快速模式 ...")
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
            result = {
                "total_path": Path(reporter.total_path).name,
                "title": reporter.title,
                "query": reporter.query,
                "stage": {"start": 0, "end": 0, "cost": 0},
                "frame": Path(reporter.frame_path).name
            }
            logger.debug(f"Quick: {result}")
            reporter.load(result)

            loop.run_until_complete(
                reporter.ask_invent_total_report(
                    os.path.dirname(reporter.total_path),
                    loop.run_until_complete(achieve(self.view_share_temp)),
                    loop.run_until_complete(achieve(self.view_total_temp)),
                    deploy.group
                )
            )
            return reporter.total_path

        elif self.keras and not self.basic:
            logger.info(f"{const.DESC} Analyzer: 智能模式 ...")
            kc = KerasClassifier(data_size=self.model_shape, aisle=self.model_aisle)
            try:
                kc.load_model(self.model_place)
            except ValueError as err:
                logger.error(f"{err}")
                kc = None
        else:
            logger.info(f"{const.DESC} Analyzer: 基础模式 ...")
            kc = None

        futures = loop.run_until_complete(
            Core.ask_analyzer(new_video_path, deploy, kc, reporter.frame_path, reporter.extra_path, self.fmp, self.fpb)
        )

        if futures is None:
            return None
        start, end, cost, classifier = futures.data

        result = {
            "total_path": Path(reporter.total_path).name,
            "title": reporter.title,
            "query": reporter.query,
            "stage": {"start": start, "end": end, "cost": f"{cost:.5f}"},
            "frame": Path(reporter.frame_path).name
        }

        if classifier:
            if isinstance(
                    template_file := loop.run_until_complete(
                        achieve(self.atom_total_temp)
                    ), Exception
            ):
                return Show.console.print(f"[bold red]{template_file}")

            original_inform = reporter.draw(
                classifier_result=classifier,
                proto_path=reporter.proto_path,
                template_file=template_file
            )
            result["extra"] = Path(reporter.extra_path).name
            result["proto"] = Path(original_inform).name

        logger.debug(f"Restore: {result}")
        reporter.load(result)

        self.enforce(reporter, classifier, start, end, cost)

        loop.run_until_complete(
            reporter.ask_create_total_report(
                os.path.dirname(reporter.total_path),
                loop.run_until_complete(achieve(self.main_share_temp)),
                loop.run_until_complete(achieve(self.main_total_temp)),
                deploy.group
            )
        )
        return reporter.total_path

    def video_dir_task(self, folder: str):
        reporter = Report(self.total_place)

        deploy = Deploy(self.initial_deploy)
        for attr in self.attrs:
            if any(line.startswith(f"--{attr}") for line in self.lines):
                logger.debug(f"Set {attr} = {(attribute := getattr(self, attr))}")
                setattr(deploy, attr, attribute)

        loop = asyncio.get_event_loop()

        if self.quick:
            logger.debug(f"{const.DESC} Analyzer: 快速模式 ...")
            for video in self.accelerate(folder):
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
                    result = {
                        "total_path": Path(reporter.total_path).name,
                        "title": reporter.title,
                        "query": reporter.query,
                        "stage": {"start": 0, "end": 0, "cost": 0},
                        "frame": Path(reporter.frame_path).name
                    }
                    logger.debug(f"Quick: {result}")
                    reporter.load(result)

            loop.run_until_complete(
                reporter.ask_invent_total_report(
                    os.path.dirname(reporter.total_path),
                    loop.run_until_complete(achieve(self.view_share_temp)),
                    loop.run_until_complete(achieve(self.view_total_temp)),
                    deploy.group
                )
            )
            return reporter.total_path

        elif self.keras and not self.basic:
            logger.info(f"{const.DESC} Analyzer: 智能模式 ...")
            kc = KerasClassifier(data_size=self.model_shape, aisle=self.model_aisle)
            try:
                kc.load_model(self.model_place)
            except ValueError as err:
                logger.error(f"{err}")
                kc = None
        else:
            logger.info(f"{const.DESC} Analyzer: 基础模式 ...")
            kc = None

        for video in self.accelerate(folder):
            reporter.title = video.title
            for path in video.sheet:
                reporter.query = os.path.basename(path).split(".")[0]
                shutil.copy(path, reporter.video_path)
                new_video_path = os.path.join(reporter.video_path, os.path.basename(path))

                futures = loop.run_until_complete(
                    Core.ask_analyzer(new_video_path, deploy, kc, reporter.frame_path, reporter.extra_path, self.fmp,
                                      self.fpb)
                )
                if futures is None:
                    continue
                start, end, cost, classifier = futures.data

                result = {
                    "total_path": Path(reporter.total_path).name,
                    "title": reporter.title,
                    "query": reporter.query,
                    "stage": {"start": start, "end": end, "cost": f"{cost:.5f}"},
                    "frame": Path(reporter.frame_path).name
                }

                if classifier:
                    if isinstance(
                            template_file := loop.run_until_complete(
                                achieve(self.atom_total_temp)
                            ), Exception
                    ):
                        return Show.console.print(f"[bold red]{template_file}")

                    original_inform = reporter.draw(
                        classifier_result=classifier,
                        proto_path=reporter.proto_path,
                        template_file=template_file
                    )
                    result["extra"] = Path(reporter.extra_path).name
                    result["proto"] = Path(original_inform).name

                logger.debug(f"Restore: {result}")
                reporter.load(result)

                self.enforce(reporter, classifier, start, end, cost)

        loop.run_until_complete(
            reporter.ask_create_total_report(
                os.path.dirname(reporter.total_path),
                loop.run_until_complete(achieve(self.main_share_temp)),
                loop.run_until_complete(achieve(self.main_total_temp)),
                deploy.group
            )
        )
        return reporter.total_path

    def train_model(self, video_file: str):
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

        deploy = Deploy(self.initial_deploy)
        for attr in self.attrs:
            if any(line.startswith(f"--{attr}") for line in self.lines):
                logger.debug(f"Set {attr} = {(attribute := getattr(self, attr))}")
                setattr(deploy, attr, attribute)

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
            silently_load_hued=False, not_transform_gray=True
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

    def build_model(self, src: str):
        if not os.path.isdir(src):
            return logger.error("训练模型需要一个分类文件夹 ...")

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
        w, h = self.shape if self.shape else (image.shape if image else (0, 0))
        name = "Gray" if image_aisle == 1 else "Hued"
        new_model_name = f"Keras_{name}_W{w}_H{h}_{random.randint(10000, 99999)}.h5"
        fc = FramixClassifier(color=image_color, aisle=image_aisle, data_size=self.shape)
        fc.build(final_path, new_model_path, new_model_name)

    async def combines_main(self, merge: list, group: bool):
        major, total = await asyncio.gather(
            achieve(self.main_share_temp), achieve(self.main_total_temp),
            return_exceptions=True
        )
        tasks = [
            Report.ask_create_total_report(m, major, total, group) for m in merge
        ]
        error_list = await asyncio.gather(*tasks)
        for e in error_list:
            if isinstance(e, Exception):
                logger.error(e)

    async def combines_view(self, merge: list, group: bool):
        views, total = await asyncio.gather(
            achieve(self.view_share_temp), achieve(self.view_total_temp),
            return_exceptions=True
        )
        tasks = [
            Report.ask_invent_total_report(m, views, total, group) for m in merge
        ]
        error_list = await asyncio.gather(*tasks)
        for e in error_list:
            if isinstance(e, Exception):
                logger.error(e)

    async def painting(self, deploy):

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

    async def analysis(self, deploy) -> typing.Optional[bool]:

        device_events = {}
        all_stop_event = asyncio.Event()

        async def timing_less(amount, serial, events) -> None:
            stop_event_control = events["stop_event"] if deploy.alone else all_stop_event
            while True:
                if events["head_event"].is_set():
                    for i in range(amount):
                        row = amount - i if amount - i <= 10 else 10
                        logger.warning(f"{serial} 剩余时间 -> {amount - i:02} 秒 {'----' * row} ...")
                        if stop_event_control.is_set() and i != amount:
                            logger.success(f"{serial} 主动停止 ...")
                            break
                        elif events["fail_event"].is_set():
                            logger.error(f"{serial} 意外停止 ...")
                            break
                        await asyncio.sleep(1)
                    return logger.warning(f"{serial} 剩余时间 -> 00 秒")
                elif events["fail_event"].is_set():
                    return logger.error(f"{serial} 意外停止 ...")
                await asyncio.sleep(0.2)

        async def timing_many(serial, events) -> None:
            stop_event_control = events["stop_event"] if deploy.alone else all_stop_event
            while True:
                if events["head_event"].is_set():
                    while True:
                        if stop_event_control.is_set():
                            return logger.success(f"{serial} 主动停止 ...")
                        elif events["fail_event"].is_set():
                            return logger.error(f"{serial} 意外停止 ...")
                        await asyncio.sleep(0.2)
                elif events["fail_event"].is_set():
                    return logger.error(f"{serial} 意外停止 ...")
                await asyncio.sleep(0.2)

        async def start_record(serial, dst, events):

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

            video_flag = f"{time.strftime('%Y%m%d%H%M%S')}_{random.randint(100, 999)}.mkv"
            video_temp = f"{os.path.join(dst, 'screen')}_{video_flag}"
            cmd = [self.scc, "-s", serial, "--no-audio", "--video-bit-rate", "8M", "--max-fps", "60"]
            cmd += ["--record", video_temp]

            transports = await Terminal.cmd_link(*cmd)
            asyncio.create_task(input_stream())
            asyncio.create_task(error_stream())
            await asyncio.sleep(1)

            return video_temp, transports

        async def close_record(video_temp, transports, events):
            if _platform == "win32":
                await Terminal.cmd_line("taskkill", "/im", "scrcpy.exe")
            else:
                transports.terminate()
                await transports.wait()

            well, fail, basis = f"成功", f"失败", os.path.basename(video_temp)
            for _ in range(10):
                if events["done_event"].is_set():
                    logger.info(f"视频录制{well}: {basis}")
                    return True
                elif events["fail_event"].is_set():
                    return logger.info(f"视频录制{fail}: {basis}")
                await asyncio.sleep(0.2)
            return logger.info(f"视频录制{fail}: {basis}")

        async def commence():

            # Wait Device Online
            async def wait_for_device(serial):
                logger.info(f"wait-for-device {serial} ...")
                await Terminal.cmd_line(self.adb, "-s", serial, "wait-for-device")

            await asyncio.gather(
                *(wait_for_device(device.serial) for device in device_list)
            )

            todo_list = []

            fmt_dir = reporter.clock() if self.quick or self.basic or self.keras else None
            for device in device_list:
                await asyncio.sleep(0.2)
                device_events[device.serial] = {
                    "head_event": asyncio.Event(), "done_event": asyncio.Event(),
                    "stop_event": asyncio.Event(), "fail_event": asyncio.Event()
                }
                if fmt_dir:
                    reporter.query = os.path.join(fmt_dir, device.serial)
                video_temp, transports = await start_record(
                    device.serial, reporter.video_path, device_events[device.serial]
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
                logger.debug(f"{const.DESC} Analyzer: 快速模式 ...")
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
                    result = {
                        "total_path": Path(total_path).name,
                        "title": title,
                        "query": query,
                        "stage": {"start": 0, "end": 0, "cost": 0},
                        "frame": Path(frame_path).name
                    }
                    logger.debug(f"Quick: {result}")
                    reporter.load(result)

            elif self.basic or self.keras:
                logger.debug(f"{const.DESC} Analyzer: {'智能模式' if kc else '基础模式'} ...")

                # TODO
                with ProcessPoolExecutor(_power, None, Active.active, (_level,)) as executor:
                    tasks = [
                        _main_loop.run_in_executor(
                            executor, self.amazing, video_temp, deploy, kc, self.fmp, self.fpb
                        ) for video_temp, *_, frame_path, extra_path, _ in task_list
                    ]
                futures = await asyncio.gather(*tasks)

                # TODO
                # futures = await asyncio.gather(
                #     *(Core.ask_analyzer(
                #         video_temp, deploy, kc,
                #         frame_path, extra_path, ffmpeg=self.ffmpeg, ffprobe=self.ffprobe
                #     ) for video_temp, *_, frame_path, extra_path, _ in task_list)
                # )

                for future, todo in zip(futures, task_list):
                    if future is None:
                        continue

                    start, end, cost, classifier = future.data
                    *_, total_path, title, query_path, query, frame_path, extra_path, proto_path = todo

                    result = {
                        "total_path": Path(total_path).name,
                        "title": title,
                        "query": query,
                        "stage": {"start": start, "end": end, "cost": f"{cost:.5f}"},
                        "frame": Path(frame_path).name
                    }

                    if classifier:
                        if isinstance(
                                template_file := await achieve(self.atom_total_temp), Exception
                        ):
                            return Show.console.print(f"[bold red]{template_file}")

                        original_inform = reporter.draw(
                            classifier_result=classifier,
                            proto_path=proto_path,
                            template_file=template_file,
                        )
                        result["extra"] = Path(extra_path).name
                        result["proto"] = Path(original_inform).name

                    logger.debug(f"Restore: {result}")
                    reporter.load(result)

                    self.enforce(reporter, classifier, start, end, cost)

            else:
                return logger.debug(f"{const.DESC} Analyzer: 录制模式 ...")

        async def device_mode_view():
            Show.console.print(f"[bold]<Link> <{'单设备模式' if len(device_list) == 1 else '多设备模式'}>")
            for device in device_list:
                Show.console.print(f"[bold #00FFAF]Connect:[/bold #00FFAF] {device}")

        async def all_time(style):
            if style.strip().lower() == "less":
                await asyncio.gather(
                    *(timing_less(timer_mode, serial, events) for serial, events in device_events.items())
                )
            elif style.strip().lower() == "many":
                await asyncio.gather(
                    *(timing_many(serial, events) for serial, events in device_events.items())
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
                *(close_record(video_temp, transports, events)
                  for (_, events), (video_temp, transports, *_) in zip(device_events.items(), task_list))
            )
            for (idx, effective), (video_temp, *_) in zip(enumerate(effective_list), task_list):
                if not effective:
                    logger.info(f"移除录制失败的视频: {Path(video_temp).name} ...")
                    task_list.pop(idx)

            if deploy.alone:
                logger.warning(f"独立控制模式不会平衡视频录制时间 ...")
                return task_list

            if len(task_list) == 0:
                logger.warning(f"没有有效任务 ...")
                return task_list

            duration_list = await asyncio.gather(
                *(Switch.ask_video_length(self.fpb, video_temp) for video_temp, *_ in task_list)
            )
            duration_list = [duration for duration in duration_list if not isinstance(duration, Exception)]
            if len(duration_list) == 0:
                task_list.clear()
                return task_list

            standard = min(duration_list)
            logger.info(f"标准录制时间: {standard}")
            balance_task = [
                balance(duration, video_src)
                for duration, (video_src, *_) in zip(duration_list, task_list)
            ]
            video_dst_list = await asyncio.gather(*balance_task)
            for idx, dst in enumerate(video_dst_list):
                task_list[idx][0] = dst

        async def event_check():
            for _, event in device_events.items():
                if event["fail_event"].is_set():
                    return False
            return True

        async def clean_check():
            all_stop_event.clear()
            for _, event in device_events.items():
                for _, v in event.items():
                    if isinstance(v, asyncio.Event):
                        v.clear()
            device_events.clear()

        async def combines_report():
            combined = False
            if len(reporter.range_list) > 0:
                combines = getattr(_mission, "combines_view") if self.quick else getattr(_mission, "combines_main")
                await combines([os.path.dirname(reporter.total_path)], deploy.group)
                combined = True
            return combined

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
                        *(is_function(device_cmds, device, medias) for device in device_list)
                )):
                    if device_func is None:
                        logger.error(f"There is no such command {device_cmds} ...")
                        break
                    if device_func.__name__ == "audio_player":
                        device_func_list = [device_func_list[0]]
                        break

                if not (method_args := device_action.get("args", None)):
                    continue

                yield [dynamically(device_func, method_args, device.serial)
                       for device_func, device in zip(device_func_list, device_list) if device_func]

            # TODO
            for device in device_list:
                device_events[device.serial]["stop_event"].set() if deploy.alone else all_stop_event.set()

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
        manage = Manage(self.adb)
        device_list = await manage.operate_device()

        titles = {"quick": "Quick", "basic": "Basic", "keras": "Keras"}
        input_title = next((title for key, title in titles.items() if getattr(self, key)), "Video")
        reporter = Report(self.total_place)

        if self.keras and not self.quick and not self.basic:
            kc = KerasClassifier(data_size=deploy.model_shape, aisle=deploy.model_aisle)
            try:
                kc.load_model(self.model_place)
            except ValueError as err:
                logger.error(f"{err}")
                kc = None
        else:
            kc = None
        # Initialization ===============================================================================================

        # Flick Loop ===================================================================================================
        if self.quick:
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
                            first = ["Notepad"] if _platform == "win32" else ["open", "-W", "-a", "TextEdit"]
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
                    check = await event_check()
                    device_list = device_list if check else await manage.operate_device()
                finally:
                    await clean_check()
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
                    return logger.error(f"缺少有效的脚本文件 {' '.join(self.fully)}...")
                script_storage = script_data

            from engine.medias import Medias
            medias = Medias()

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

                            # TODO
                            # for _device in device_list:
                            #     device_events[
                            #         _device.serial
                            #     ]["stop_event"].set() if deploy.alone else all_stop_event.set()

                            await all_time("many")
                            await all_over()
                            await analysis_tactics()
                            check = await event_check()
                            device_list = device_list if check else await manage.operate_device()
                        finally:
                            await clean_check()

            if await combines_report():
                return True
            return Show.console.print(f"[bold red]没有可以生成的报告 ...\n")
        # Other Loop ===================================================================================================

        return None


class Core(object):

    @staticmethod
    async def ask_analyzer(vision, deploy, kc, *args) -> typing.Optional["Review"]:

        frame_path, extra_path, fmp, fpb = args

        async def validate():
            screen_cap = None
            if os.path.isfile(vision):
                screen = cv2.VideoCapture(vision)
                if screen.isOpened():
                    screen_cap = Path(vision)
                screen.release()
            elif os.path.isdir(vision):
                file_list = [
                    file for file in os.listdir(vision) if os.path.isfile(os.path.join(vision, file))
                ]
                if len(file_list) >= 1:
                    screen = cv2.VideoCapture(open_file := os.path.join(vision, file_list[0]))
                    if screen.isOpened():
                        screen_cap = Path(open_file)
                    screen.release()
            return screen_cap

        async def frame_flip():
            change_record = os.path.join(
                os.path.dirname(vision),
                f"screen_fps{deploy.frate}_{random.randint(100, 999)}.mp4"
            )

            duration = await Switch.ask_video_length(fpb, vision)
            vision_start, vision_close, vision_limit = await Switch.ask_magic_point(
                Parser.parse_mills(deploy.start),
                Parser.parse_mills(deploy.close),
                Parser.parse_mills(deploy.limit),
                duration
            )
            vision_start = Parser.parse_times(vision_start)
            vision_close = Parser.parse_times(vision_close)
            vision_limit = Parser.parse_times(vision_limit)
            logger.info(f"视频时长: [{duration}] [{Parser.parse_times(duration)}]")
            logger.info(f"start=[{vision_start}] - close=[{vision_close}] - limit=[{vision_limit}]")

            await Switch.ask_video_change(
                fmp, deploy.frate, vision, change_record,
                start=vision_start, close=vision_close, limit=vision_limit
            )
            logger.info(f"视频转换完成: {Path(change_record).name}")
            os.remove(vision)
            logger.info(f"移除旧的视频: {Path(vision).name}")

            if deploy.shape:
                original_shape = await Switch.ask_video_larger(fpb, change_record)
                w, h, ratio = await Switch.ask_magic_frame(original_shape, deploy.shape)
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

            compress_hook = CompressHook(1, None, False)
            cutter.add_hook(compress_hook)

            if len(crop_list := deploy.crops) > 0 and sum([j for i in crop_list for j in i.values()]) > 0:
                for crop in crop_list:
                    x, y, x_size, y_size = crop.values()
                    crop_hook = PaintCropHook((y_size, x_size), (y, x))
                    cutter.add_hook(crop_hook)
                    logger.debug(f"{crop_hook.__class__.__name__}: {x, y, x_size, y_size}")

            if len(omit_list := deploy.omits) > 0 and sum([j for i in omit_list for j in i.values()]) > 0:
                for omit in omit_list:
                    x, y, x_size, y_size = omit.values()
                    omit_hook = PaintOmitHook((y_size, x_size), (y, x))
                    cutter.add_hook(omit_hook)
                    logger.debug(f"{omit_hook.__class__.__name__}: {x, y, x_size, y_size}")

            save_hook = FrameSaveHook(extra_path)
            cutter.add_hook(save_hook)

            res = cutter.cut(
                video=video, block=deploy.block
            )

            stable, unstable = res.get_range(
                threshold=deploy.thres, offset=deploy.shift
            )

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

            classify = kc.classify(
                video=video, valid_range=stable, keep_data=True
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
                logger.warning(f"{const.DESC} Analyzer recalculate ...")
            except IndexError as e:
                logger.error(f"{e}")
                for i, unstable_stage in enumerate(classify.get_specific_stage_range("-3")):
                    Show.console.print(f"[bold]第 {i:02} 个非稳定阶段")
                    Show.console.print(f"[bold]{'=' * 30}")
                    for j, frame in enumerate(unstable_stage):
                        Show.console.print(f"[bold]第 {j:05} 帧: {frame}")
                    Show.console.print(f"[bold]{'=' * 30}\n")
                start_frame = classify.get_important_frame_list()[0]
                end_frame = classify.get_important_frame_list()[-1]
                logger.warning(f"{const.DESC} Analyzer recalculate ...")

            if start_frame == end_frame:
                logger.warning(f"{start_frame} == {end_frame}")
                start_frame, end_frame = classify.data[0], classify.data[-1]
                logger.warning(f"{const.DESC} Analyzer recalculate ...")

            time_cost = end_frame.timestamp - start_frame.timestamp
            logger.info(
                f"图像分类结果: [开始帧: {start_frame.timestamp:.5f}] [结束帧: {end_frame.timestamp:.5f}] [总耗时: {time_cost:.5f}]"
            )
            return start_frame.frame_id, end_frame.frame_id, time_cost

        async def frame_forge(frame):
            try:
                (_, codec), pic_path = cv2.imencode(".png", frame.data), os.path.join(
                    frame_path, f"{frame.frame_id}_{format(round(frame.timestamp, 5), '.5f')}.png"
                )
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

            logger.debug(f"运行环境: {_platform}")
            if _platform == "win32":
                forge_result = await asyncio.gather(
                    *(frame_forge(frame) for frame in frames), return_exceptions=True
                )
            else:
                tasks = [
                    [frame_forge(frame) for frame in chunk] for chunk in
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

            start_frame, end_frame = frames[0], frames[-1]

            time_cost = end_frame.timestamp - start_frame.timestamp
            return (start_frame.frame_id, end_frame.frame_id, time_cost), None

        async def analytics_keras():
            classify, frames = await frame_flow()

            logger.debug(f"运行环境: {_platform}")
            if _platform == "win32":
                flick_result, *forge_result = await asyncio.gather(
                    frame_flick(classify), *(frame_forge(frame) for frame in frames),
                    return_exceptions=True
                )
            else:
                tasks = [
                    [frame_forge(frame) for frame in chunk] for chunk in
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

        # Analyzer first ===============================================================================================
        if (screen_record := await validate()) is None:
            return logger.error(f"{vision} 不是一个标准的视频文件或视频文件已损坏 ...")
        logger.info(f"{screen_record.name} 可正常播放，准备加载视频 ...")
        # Analyzer first ===============================================================================================

        # Analyzer last ================================================================================================
        (start, end, cost), classifier = await analytics_keras() if kc else await analytics_basic()
        return Review(start, end, cost, classifier)
        # Analyzer last ================================================================================================


async def achieve(template_path: str) -> str | Exception:
    try:
        async with aiofiles.open(template_path, "r", encoding="utf-8") as f:
            template_file = await f.read()
    except FileNotFoundError as e:
        return e
    return template_file


async def arithmetic(*args, **__) -> None:

    async def initialization(transfer):
        proc = members if (members := len(transfer)) <= power else power
        rank = "ERROR" if members > 1 else level
        return proc, Active.active, (rank,)

    async def multiple_merge(transfer):
        if len(transfer) <= 1:
            return None
        template_total = await achieve(mission.view_total_temp if mission.quick else mission.main_total_temp)
        Report.merge_report(results, template_total, mission.quick)

    mission, cmd_lines, level, power, loop, *_ = args
    # --video ==========================================================================================================
    if video_list := cmd_lines.video:
        with Pool(*(await initialization(video_list))) as pool:
            results = pool.starmap(mission.video_task, [(i,) for i in video_list])
        await multiple_merge(video_list)
        sys.exit(0)
    # --stack ==========================================================================================================
    elif stack_list := cmd_lines.stack:
        with Pool(*(await initialization(stack_list))) as pool:
            results = pool.starmap(mission.video_dir_task, [(i,) for i in stack_list])
        await multiple_merge(stack_list)
        sys.exit(0)
    # --train ==========================================================================================================
    elif train_list := cmd_lines.train:
        with Pool(*(await initialization(train_list))) as pool:
            pool.starmap(mission.train_model, [(i,) for i in train_list])
        sys.exit(0)
    # --build ==========================================================================================================
    elif build_list := cmd_lines.build:
        with Pool(*(await initialization(build_list))) as pool:
            pool.starmap(mission.build_model, [(i,) for i in build_list])
        sys.exit(0)

    return None


async def scheduling(*args, **__) -> None:
    mission, cmd_lines, level, power, loop, *_ = args

    deploy = Deploy(mission.initial_deploy)
    for attr in mission.attrs:
        if any(line.startswith(f"--{attr}") for line in mission.lines):
            logger.debug(f"Set {attr} = {(attribute := getattr(mission, attr))}")
            setattr(deploy, attr, attribute)

    # --flick --carry --fully ==========================================================================================
    if cmd_lines.flick or cmd_lines.carry or cmd_lines.fully:
        await mission.analysis(deploy)
    # --paint ==========================================================================================================
    elif cmd_lines.paint:
        await mission.painting(deploy)
    # --union ==========================================================================================================
    elif cmd_lines.union:
        await mission.combines_view(cmd_lines.union, mission.group)
    # --merge ==========================================================================================================
    elif cmd_lines.merge:
        await mission.combines_main(cmd_lines.merge, mission.group)
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

    _carry = _cmd_lines.carry
    _fully = _cmd_lines.fully

    _quick = _cmd_lines.quick
    _basic = _cmd_lines.basic
    _keras = _cmd_lines.keras

    _alone = _cmd_lines.alone
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

    _mission = Mission(
        _carry, _fully, _quick, _basic, _keras,
        _alone, _group, _boost, _color, _shape, _scale,
        _start, _close, _limit, _begin, _final,
        _frate, _thres, _shift, _block, _crops, _omits,
        attrs=_attrs,
        lines=_lines,
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

    from multiprocessing import Pool
    from concurrent.futures import ProcessPoolExecutor

    _main_loop = asyncio.get_event_loop()

    try:
        _main_loop.run_until_complete(arithmetic(
            _mission, _cmd_lines, _level, _power, _main_loop)
        )
        _main_loop.run_until_complete(scheduling(
            _mission, _cmd_lines, _level, _power, _main_loop)
        )
    except KeyboardInterrupt:
        sys.exit(0)
