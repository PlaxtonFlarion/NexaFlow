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
from loguru import logger
from rich.prompt import Prompt
from frameflow.database import DataBase
from frameflow.show import Show
from frameflow.manage import Manage
from frameflow.parameters import Deploy, Option

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
    Show.console.print("[bold red]Only compatible with Windows and macOS platforms ...")
    time.sleep(5)
    sys.exit(1)

_tools_path = os.path.join(_job_path, "archivix", "tools")
_model_path = os.path.join(_job_path, "archivix", "molds", "model.h5")
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
        "[bold]Only compatible with [bold red]Windows[/bold red] and [bold red]macOS[/bold red] platforms ...[bold]")
    time.sleep(5)
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
    from nexaflow.hook import CropHook, OmitHook, FrameSaveHook, PaintCropHook, PaintOmitHook
    from nexaflow.classifier.keras_classifier import KerasClassifier
    from nexaflow.classifier.framix_classifier import FramixClassifier
except (RuntimeError, ModuleNotFoundError) as err:
    Show.console.print(f"[bold red]Error: {err}")
    time.sleep(5)
    sys.exit(1)


class Parser(object):

    @staticmethod
    def parse_cmd():

        def parse_shape(dim_str):
            if dim_str:
                shape = [int(i) for i in re.split(r'[\s,;]+', dim_str)]
                return tuple(shape) if len(shape) == 2 else (shape[0], shape[0])
            return None

        def parse_scale(dim_str):
            try:
                return int(dim_str)
            except ValueError:
                try:
                    return float(dim_str)
                except ValueError:
                    return None

        parser = ArgumentParser(description="Command Line Arguments Framix")

        parser.add_argument('--flick', action='store_true', help='循环分析视频帧')
        parser.add_argument('--paint', action='store_true', help='绘制图片分割线条')
        parser.add_argument('--video', action='append', help='分析视频')
        parser.add_argument('--stack', action='append', help='分析视频文件集合')
        parser.add_argument('--merge', action='append', help='聚合时间戳报告')
        parser.add_argument('--union', action='append', help='聚合视频帧报告')
        parser.add_argument('--train', action='append', help='归类图片文件')
        parser.add_argument('--build', action='append', help='训练模型文件')

        parser.add_argument('--alone', action='store_true', help='独立模式')
        parser.add_argument('--quick', action='store_true', help='快速模式')
        parser.add_argument('--basic', action='store_true', help='基础模式')
        parser.add_argument('--keras', action='store_true', help='智能模式')

        parser.add_argument('--boost', action='store_true', help='跳帧模式')
        parser.add_argument('--color', action='store_true', help='彩色模式')
        parser.add_argument('--crops', action='append', help='获取区域')
        parser.add_argument('--omits', action='append', help='忽略区域')

        parser.add_argument('--shape', nargs='?', const=None, type=parse_shape, help='图片尺寸')
        parser.add_argument('--scale', nargs='?', const=None, type=parse_scale, help='缩放比例')

        # --debug 不展示
        parser.add_argument('--debug', action='store_true', help='调试模式')

        return parser.parse_args()


class Missions(object):

    def __init__(self, alone: bool, quick: bool, basic: bool, keras: bool, *args, **kwargs):
        self.alone, self.quick, self.basic, self.keras = alone, quick, basic, keras
        self.boost, self.color, self.crops, self.omits, self.shape, self.scale = args

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

        if not self.shape and not self.scale:
            self.shape = (350, 700)

    @staticmethod
    def only_video(folder: str):

        class Entry(object):

            def __init__(self, title: str, place: str, sheet: list):
                self.title = title
                self.place = place
                self.sheet = sheet

        return [
            Entry(
                os.path.basename(root), root,
                [os.path.join(root, f) for f in sorted(file) if "log" not in f.split(".")[-1]]
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
        deploy.boost = self.boost
        deploy.color = self.color
        deploy.crops = self.crops
        deploy.omits = self.omits

        looper = asyncio.get_event_loop()

        if self.quick:
            logger.debug(f"Analyzer: 快速模式 ...")
            video_filter = [f"fps={deploy.fps}"] if deploy.color else [f"fps={deploy.fps}", "format=gray"]
            if self.shape:
                w, h = self.shape
                video_filter.append(f"scale={w}:{h}")
                logger.debug(f"Image Shape: {self.shape}")
            elif self.scale:
                scale = max(0.1, min(1.0, self.scale if self.scale else 1.0))
                video_filter.append(f"scale=iw*{scale}:ih*{scale}")
                logger.debug(f"Image Scale: {self.scale}")
            else:
                video_filter.append(f"scale=iw*0.5:ih*0.5")

            looper.run_until_complete(
                ask_video_detach(
                    self.ffmpeg, video_filter, new_video_path, reporter.frame_path
                )
            )
            result = {
                "total_path": reporter.total_path,
                "title": reporter.title,
                "query_path": reporter.query_path,
                "query": reporter.query,
                "stage": {"start": 0, "end": 0, "cost": 0},
                "frame": reporter.frame_path,
            }
            logger.debug(f"Quick: {result}")
            reporter.load(result)

            looper.run_until_complete(
                reporter.ask_invent_total_report(
                    os.path.dirname(reporter.total_path),
                    get_template(self.view_temp),
                    get_template(self.view_total_temp))
            )
            return reporter.total_path

        elif self.basic:
            logger.debug(f"Analyzer: 基础模式 ...")
            kc = None

        elif self.keras:
            logger.debug(f"Analyzer: 智能模式 ...")
            kc = KerasClassifier(
                target_size=self.shape,
                compress_rate=self.scale,
                data_size=deploy.model_size
            )
            kc.load_model(self.model_path)

        else:
            logger.debug(f"Analyzer: 基础模式 ...")
            kc = None

        futures = looper.run_until_complete(
            analyzer(
                new_video_path, deploy, kc, reporter.frame_path, reporter.extra_path,
                ffmpeg=self.ffmpeg, shape=self.shape, scale=self.scale
            )
        )

        if futures is None:
            return None
        start, end, cost, classifier = futures

        if self.keras and not self.basic:
            original_inform = reporter.draw(
                classifier_result=classifier,
                proto_path=reporter.proto_path,
                template_file=get_template(self.alien),
            )
        else:
            original_inform = ""

        result = {
            "total_path": reporter.total_path,
            "title": reporter.title,
            "query_path": reporter.query_path,
            "query": reporter.query,
            "stage": {"start": start, "end": end, "cost": f"{cost:.5f}"},
            "frame": reporter.frame_path,
            "extra": reporter.extra_path,
            "proto": original_inform,
        }
        logger.debug(f"Restore: {result}")
        reporter.load(result)

        with DataBase(os.path.join(reporter.reset_path, "Framix_Data.db")) as database:
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

        looper.run_until_complete(
            reporter.ask_create_total_report(
                os.path.dirname(reporter.total_path),
                get_template(self.main_temp),
                get_template(self.main_total_temp)
            )
        )
        return reporter.total_path

    def video_dir_task(self, folder: str):
        reporter = Report(total_path=self.initial_report)

        deploy = Deploy(self.initial_deploy)
        deploy.boost = self.boost
        deploy.color = self.color
        deploy.crops = self.crops
        deploy.omits = self.omits

        looper = asyncio.get_event_loop()

        if self.quick:
            logger.debug(f"Analyzer: 快速模式 ...")
            for video in self.only_video(folder):
                reporter.title = video.title
                for path in video.sheet:
                    reporter.query = os.path.basename(path).split(".")[0]
                    shutil.copy(path, reporter.video_path)
                    new_video_path = os.path.join(reporter.video_path, os.path.basename(path))

                    video_filter = [f"fps={deploy.fps}"] if deploy.color else [f"fps={deploy.fps}", "format=gray"]
                    if self.shape:
                        w, h = self.shape
                        video_filter.append(f"scale={w}:{h}")
                        logger.debug(f"Image Shape: {self.shape}")
                    elif self.scale:
                        scale = max(0.1, min(1.0, self.scale if self.scale else 1.0))
                        video_filter.append(f"scale=iw*{scale}:ih*{scale}")
                        logger.debug(f"Image Scale: {self.scale}")
                    else:
                        video_filter.append(f"scale=iw*0.5:ih*0.5")

                    looper.run_until_complete(
                        ask_video_detach(
                            self.ffmpeg, video_filter, new_video_path, reporter.frame_path
                        )
                    )
                    result = {
                        "total_path": reporter.total_path,
                        "title": reporter.title,
                        "query_path": reporter.query_path,
                        "query": reporter.query,
                        "stage": {"start": 0, "end": 0, "cost": 0},
                        "frame": reporter.frame_path,
                    }
                    logger.debug(f"Quick: {result}")
                    reporter.load(result)

            looper.run_until_complete(
                reporter.ask_invent_total_report(
                    os.path.dirname(reporter.total_path),
                    get_template(self.view_temp),
                    get_template(self.view_total_temp)
                )
            )
            return reporter.total_path

        elif self.basic:
            logger.debug(f"Analyzer: 基础模式 ...")
            kc = None

        elif self.keras:
            logger.debug(f"Analyzer: 智能模式 ...")
            kc = KerasClassifier(
                target_size=self.shape,
                compress_rate=self.scale,
                data_size=deploy.model_size
            )
            kc.load_model(self.model_path)

        else:
            logger.debug(f"Analyzer: 基础模式 ...")
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
                        ffmpeg=self.ffmpeg, shape=self.shape, scale=self.scale
                    )
                )
                if futures is None:
                    continue
                start, end, cost, classifier = futures

                if self.basic:
                    original_inform = ""
                elif self.keras:
                    original_inform = reporter.draw(
                        classifier_result=classifier,
                        proto_path=reporter.proto_path,
                        template_file=get_template(self.alien),
                    )
                else:
                    original_inform = ""

                result = {
                    "total_path": reporter.total_path,
                    "title": reporter.title,
                    "query_path": reporter.query_path,
                    "query": reporter.query,
                    "stage": {"start": start, "end": end, "cost": f"{cost:.5f}"},
                    "frame": reporter.frame_path,
                    "extra": reporter.extra_path,
                    "proto": original_inform,
                }
                logger.debug(f"Restore: {result}")
                reporter.load(result)

                with DataBase(os.path.join(reporter.reset_path, "Framix_Data.db")) as database:
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

        looper.run_until_complete(
            reporter.ask_create_total_report(
                os.path.dirname(reporter.total_path),
                get_template(self.main_temp),
                get_template(self.main_total_temp),
            )
        )
        return reporter.total_path

    def train_model(self, video_file: str):
        reporter = Report(total_path=self.initial_report)
        reporter.title = f"Model_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}"
        if not os.path.exists(reporter.query_path):
            os.makedirs(reporter.query_path, exist_ok=True)

        deploy = Deploy(self.initial_deploy)
        deploy.boost = self.boost
        deploy.color = self.color
        deploy.crops = self.crops
        deploy.omits = self.omits

        video_temp_file = os.path.join(
            reporter.query_path, f"tmp_fps{deploy.fps}.mp4"
        )
        asyncio.run(
            ask_video_change(self.ffmpeg, deploy.fps, video_file, video_temp_file)
        )

        video = VideoObject(video_temp_file)
        video.load_frames(deploy.color, self.shape, self.scale)

        cutter = VideoCutter(
            step=deploy.step,
            compress_rate=self.scale,
            target_size=self.shape
        )
        res = cutter.cut(
            video=video,
            block=deploy.block,
            window_size=deploy.window_size,
            window_coefficient=deploy.window_coefficient
        )
        stable, unstable = res.get_range(
            threshold=deploy.threshold,
            offset=deploy.offset
        )
        res.pick_and_save(
            range_list=stable,
            frame_count=20,
            to_dir=reporter.query_path,
            meaningful_name=True,
        )

        os.remove(video_temp_file)

    def build_model(self, src: str):
        if os.path.isdir(src):
            real_path, file_list = "", []
            logger.debug(f"搜索文件夹: {src}")
            for root, dirs, files in os.walk(src, topdown=False):
                for name in files:
                    file_list.append(os.path.join(root, name))
                for name in dirs:
                    if len(name) == 1 and re.search(r"0", name):
                        real_path = os.path.dirname(os.path.join(root, name))
                        logger.debug(f"分类文件夹: {real_path}")
                        break
            if real_path and len(file_list) > 0:
                new_model_path = os.path.join(real_path, f"Create_Model_{time.strftime('%Y%m%d%H%M%S')}")
                new_model_name = f"Keras_Model_{random.randint(10000, 99999)}.h5"

                fc = FramixClassifier(data_size=self.shape)
                fc.build(real_path, new_model_path, new_model_name)
            else:
                logger.error("文件夹未正确分类 ...")
        else:
            logger.error("训练模型需要一个分类文件夹 ...")

    async def combines_main(self, merge: list):
        major, total = await asyncio.gather(
            ask_get_template(self.main_temp), ask_get_template(self.main_total_temp),
            return_exceptions=True
        )
        tasks = [
            Report.ask_create_total_report(m, major, total) for m in merge
        ]
        error = await asyncio.gather(*tasks)
        for e in error:
            if isinstance(e, Exception):
                logger.error(e)

    async def combines_view(self, merge: list):
        views, total = await asyncio.gather(
            ask_get_template(self.view_temp), ask_get_template(self.view_total_temp),
            return_exceptions=True
        )
        tasks = [
            Report.ask_invent_total_report(m, views, total) for m in merge
        ]
        error = await asyncio.gather(*tasks)
        for e in error:
            if isinstance(e, Exception):
                logger.error(e)

    async def painting(self):
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

                if self.color:
                    old_image = toolbox.imread(image_save_path)
                    new_image = VideoFrame(0, 0, old_image)
                else:
                    old_image = toolbox.imread(image_save_path)
                    old_image = toolbox.turn_grey(old_image)
                    new_image = VideoFrame(0, 0, old_image)

                if len(self.crops) > 0:
                    for crop in self.crops:
                        if len(crop) == 4 and sum(crop) > 0:
                            x, y, x_size, y_size = crop
                        paint_crop_hook = PaintCropHook((y_size, x_size), (y, x))
                        paint_crop_hook.do(new_image)

                if len(self.omits) > 0:
                    for omit in self.omits:
                        if len(omit) == 4 and sum(omit) > 0:
                            x, y, x_size, y_size = omit
                            paint_omit_hook = PaintOmitHook((y_size, x_size), (y, x))
                            paint_omit_hook.do(new_image)

                cv2.imencode(".png", new_image.data)[1].tofile(image_save_path)

                image_file = Image.open(image_save_path)
                image_file = image_file.convert("RGB")

                original_w, original_h = image_file.size
                if self.shape:
                    shape_w, shape_h = self.shape
                    twist_w, twist_h = min(original_w, shape_w), min(original_h, shape_h)
                else:
                    twist_w, twist_h = original_w, original_h

                min_scale, max_scale = 0.3, 1.0
                if self.scale:
                    image_scale = max_scale if self.scale > max_scale else (
                        min_scale if self.scale < min_scale else self.scale)
                else:
                    image_scale = min_scale if twist_w == original_w or twist_h == original_h else max_scale

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

    async def analysis(self):

        device_events = {}
        all_stop_event = asyncio.Event()

        # Time
        async def timepiece(amount, serial, events):
            stop_event_control = events["stop_event"] if self.alone else all_stop_event
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

            stop_event_control = events["stop_event"] if self.alone else all_stop_event
            temp_video = f"{os.path.join(dst, 'screen')}_{time.strftime('%Y%m%d%H%M%S')}_{random.randint(100, 999)}.mkv"
            cmd = [
                self.scrcpy, "-s", serial,
                "--no-audio", "--video-bit-rate", "8M", "--max-fps", "60",
                "--record", temp_video
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
            start_time_str = str(datetime.timedelta(seconds=start_time_point))
            end_time_str = str(datetime.timedelta(seconds=duration))
            Show.console.print(f"[bold]{os.path.basename(video_src)} {duration} [{start_time_str} - {end_time_str}]")
            video_dst = os.path.join(
                os.path.dirname(video_src), f"tailor_fps{deploy.fps}_{random.randint(100, 999)}.mp4"
            )
            await ask_video_tailor(
                self.ffmpeg, video_src, video_dst, start_time_str, end_time_str
            )
            return video_dst

        # Record
        async def commence():

            async def device_online(serial):
                Show.console.print(f"[bold]wait-for-device [{serial}] ...[/bold]")
                await Terminal.cmd_line(self.adb, "-s", serial, "wait-for-device")

            await asyncio.gather(
                *(device_online(device.serial) for device in device_list)
            )
            todo_list = []

            if self.quick or self.basic or self.keras:
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

                if not self.alone and len(todo_list) > 1:
                    duration_list = await asyncio.gather(
                        *(ask_video_length(self.ffprobe, temp_video) for temp_video, *_ in todo_list)
                    )
                    duration_list = [duration for duration in duration_list if not isinstance(duration, Exception)]
                    if len(duration_list) == 0:
                        todo_list.clear()
                        return todo_list

                    standard = min(duration_list)
                    Show.console.print(f"[bold]标准录制时间: {standard}")
                    balance_task = [
                        video_balance(standard, duration, video_src)
                        for duration, (video_src, *_) in zip(duration_list, todo_list)
                    ]
                    video_dst_list = await asyncio.gather(*balance_task)
                    for idx, dst in enumerate(video_dst_list):
                        todo_list[idx][0] = dst

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

            return todo_list

        # Analysis Mode
        async def analysis_tactics():
            if len(task_list) == 0:
                return False

            if self.quick:
                logger.debug(f"Analyzer: 快速模式 ...")
                video_filter = [f"fps={deploy.fps}"] if deploy.color else [f"fps={deploy.fps}", "format=gray"]
                if self.shape:
                    w, h = self.shape
                    video_filter.append(f"scale={w}:{h}")
                    logger.debug(f"Image Shape: {self.shape}")
                elif self.scale:
                    scale = max(0.1, min(1.0, self.scale if self.scale else 1.0))
                    video_filter.append(f"scale=iw*{scale}:ih*{scale}")
                    logger.debug(f"Image Scale: {self.scale}")
                else:
                    video_filter.append(f"scale=iw*0.5:ih*0.5")

                await asyncio.gather(
                    *(ask_video_detach(self.ffmpeg, video_filter, temp_video, frame_path) for
                      temp_video, *_, frame_path, _, _ in task_list)
                )
                for *_, total_path, title, query_path, query, frame_path, _, _ in task_list:
                    result = {
                        "total_path": total_path,
                        "title": title,
                        "query_path": query_path,
                        "query": query,
                        "stage": {"start": 0, "end": 0, "cost": 0},
                        "frame": frame_path,
                    }
                    logger.debug(f"Quick: {result}")
                    reporter.load(result)

            elif self.basic or self.keras:
                futures = await asyncio.gather(
                    *(analyzer(
                        temp_video, deploy, kc, frame_path, extra_path,
                        ffmpeg=self.ffmpeg, shape=self.shape, scale=self.scale
                    ) for temp_video, *_, frame_path, extra_path, _ in task_list)
                )

                for future, todo in zip(futures, task_list):
                    if future is None:
                        continue

                    start, end, cost, classifier = future
                    *_, total_path, title, query_path, query, frame_path, extra_path, proto_path = todo

                    if self.basic:
                        logger.debug(f"Analyzer: 基础模式 ...")
                        original_inform = ""
                    elif self.keras:
                        logger.debug(f"Analyzer: 智能模式 ...")
                        template_file = await ask_get_template(self.alien)
                        original_inform = reporter.draw(
                            classifier_result=classifier,
                            proto_path=proto_path,
                            template_file=template_file,
                        )
                    else:
                        original_inform = ""

                    result = {
                        "total_path": total_path,
                        "title": title,
                        "query_path": query_path,
                        "query": query,
                        "stage": {"start": start, "end": end, "cost": f"{cost:.5f}"},
                        "frame": frame_path,
                        "extra": extra_path,
                        "proto": original_inform,
                    }
                    logger.debug(f"Restore: {result}")
                    reporter.load(result)

                    with DataBase(os.path.join(os.path.dirname(total_path), "Nexa_Recovery", "Framix_Data.db")) as database:
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
                return False

        # Device View
        async def device_mode_view():
            if len(device_list) == 1:
                Show.console.print(f"[bold]<Link> <单设备模式>")
            else:
                Show.console.print(f"[bold]<Link> <多设备模式>")
            for device in device_list:
                Show.console.print(f"[bold #00FFAF]Connect:[/bold #00FFAF] {device}")

        # Start
        manage = Manage(self.adb)
        device_list = await manage.operate_device()

        # Initialization ===============================================================================================
        reporter = Report(self.initial_report)
        if self.quick:
            reporter.title = f"Quick_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        elif self.basic:
            reporter.title = f"Basic_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        elif self.keras:
            reporter.title = f"Keras_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        else:
            reporter.title = f"Video_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

        deploy = Deploy(self.initial_deploy)
        deploy.boost = self.boost
        deploy.color = self.color
        deploy.crops = self.crops
        deploy.omits = self.omits

        if self.keras and not self.quick and not self.basic:
            kc = KerasClassifier(
                target_size=self.shape,
                compress_rate=self.scale,
                data_size=deploy.model_size
            )
            kc.load_model(self.model_path)
        else:
            kc = None
        # Initialization ===============================================================================================

        # Loop
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
                                src_hd = f"Record_{time.strftime('%Y%m%d_%H%M%S')}" if self.alone else f"Framix_{time.strftime('%Y%m%d_%H%M%S')}"
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
                                await self.combines_view([os.path.dirname(reporter.total_path)])
                            else:
                                await self.combines_main([os.path.dirname(reporter.total_path)])
                            finally:
                                break
                        Show.console.print(f"[bold red]没有可以生成的报告 ...[/bold red]\n")
                        continue
                    elif select == "deploy":
                        logger.warning("修改 deploy.json 文件后请完全退出编辑器进程再继续操作 ...")
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


def worker_init(log_level: str):
    logger.remove(0)
    log_format = "| <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sys.stderr, format=log_format, level=log_level.upper())


def get_template(template_path: str):
    with open(template_path, mode="r", encoding="utf-8") as f:
        template_file = f.read()
    return template_file


async def ask_get_template(template_path: str):
    try:
        async with aiofiles.open(template_path, mode="r", encoding="utf-8") as f:
            template_file = await f.read()
    except FileNotFoundError as e:
        return e
    else:
        return template_file


async def ask_video_change(ffmpeg, fps: int, src: str, dst: str):
    cmd = [
        ffmpeg, "-i", src, "-vf", f"fps={fps}",
        "-c:v", "libx264", "-crf", "18", "-c:a", "copy", dst
    ]
    await Terminal.cmd_line(*cmd)


async def ask_video_detach(ffmpeg, video_filter: list, src: str, dst: str):
    cmd = [
        ffmpeg, "-i", src, "-vf", ",".join(video_filter),
        f"{os.path.join(dst, 'frame_%05d.png')}"
    ]
    await Terminal.cmd_line(*cmd)


async def ask_video_tailor(ffmpeg, src: str, dst: str, start: str, end: str):
    cmd = [
        ffmpeg, "-ss", start, "-i", src, "-t", end, "-c", "copy", dst
    ]
    await Terminal.cmd_line(*cmd)


async def ask_video_length(ffprobe, src: str):
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


async def analyzer(
        vision_path: str, deploy: "Deploy", kc: "KerasClassifier", *args, **kwargs
):
    frame_path, extra_path = args
    ffmpeg = kwargs["ffmpeg"]
    shape, scale = kwargs["shape"], kwargs["scale"]

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
        await ask_video_change(ffmpeg, deploy.fps, vision_path, change_record)
        logger.info(f"视频转换完成: {os.path.basename(change_record)}")
        os.remove(vision_path)
        logger.info(f"移除旧的视频: {os.path.basename(vision_path)}")

        video = VideoObject(change_record)
        task, hued = video.load_frames(deploy.color, shape, scale)
        return video, task, hued

    async def frame_flow():
        video, task, hued = await frame_flip()
        cutter = VideoCutter(
            step=deploy.step,
            compress_rate=scale,
            target_size=shape
        )

        if len(deploy.crops) > 0:
            for crop in deploy.crops:
                x, y, x_size, y_size = crop
                crop_hook = CropHook((y_size, x_size), (y, x))
                cutter.add_hook(crop_hook)

        if len(deploy.omits) > 0:
            for omit in deploy.omits:
                x, y, x_size, y_size = omit
                omit_hook = OmitHook((y_size, x_size), (y, x))
                cutter.add_hook(omit_hook)

        save_hook = FrameSaveHook(extra_path)
        cutter.add_hook(save_hook)

        res = cutter.cut(
            video=video,
            block=deploy.block,
            window_size=deploy.window_size,
            window_coefficient=deploy.window_coefficient
        )

        stable, unstable = res.get_range(
            threshold=deploy.threshold,
            offset=deploy.offset
        )

        files = os.listdir(extra_path)
        files.sort(key=lambda n: int(n.split("(")[0]))
        total_images = len(files)
        interval = total_images // 11 if total_images > 12 else 1
        for index, file in enumerate(files):
            if index % interval != 0:
                os.remove(
                    os.path.join(extra_path, file)
                )

        draws = os.listdir(extra_path)
        for draw in draws:
            toolbox.draw_line(
                os.path.join(extra_path, draw)
            )

        classify = kc.classify(video=video, valid_range=stable, keep_data=True)

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
        try:
            start_frame = classify.get_not_stable_stage_range()[0][1]
            end_frame = classify.get_not_stable_stage_range()[-1][-1]
        except AssertionError:
            start_frame = classify.get_important_frame_list()[0]
            end_frame = classify.get_important_frame_list()[-1]

        if start_frame == end_frame:
            start_frame = classify.data[0]
            end_frame = classify.data[-1]

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

    if kc is None:
        (start, end, cost), classifier = await analytics_basic()
    else:
        (start, end, cost), classifier = await analytics_keras()
    return start, end, cost, classifier


async def ask_main():
    if cmd_lines.flick:
        await missions.analysis()
    elif cmd_lines.paint:
        await missions.painting()
    elif cmd_lines.merge and len(cmd_lines.merge) > 0:
        await missions.combines_main(cmd_lines.merge)
    elif cmd_lines.union and len(cmd_lines.union) > 0:
        await missions.combines_view(cmd_lines.union)
    else:
        Show.help_document()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        Show.help_document()
        sys.exit(0)

    from multiprocessing import Pool, freeze_support

    freeze_support()

    from argparse import ArgumentParser

    cmd_lines = Parser.parse_cmd()
    _level = "DEBUG" if cmd_lines.debug else "INFO"
    worker_init(_level)

    # Debug Mode =======================================================================================================
    logger.debug(f"Level: {_level}")

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

    _boost = cmd_lines.boost
    _color = cmd_lines.color

    _shape = cmd_lines.shape
    _scale = cmd_lines.scale

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

    # Debug Mode =======================================================================================================
    logger.debug(f"Initial-Report: {_initial_report}")
    logger.debug(f"Initial-Deploy: {_initial_deploy}")
    logger.debug(f"Initial-Option: {_initial_option}")
    # Debug Mode =======================================================================================================

    missions = Missions(
        _alone, _quick, _basic, _keras, _boost, _color, _crops, _omits, _shape, _scale,
        model_path=_model_path,
        main_total_temp=_main_total_temp, main_temp=_main_temp,
        view_total_temp=_view_total_temp, view_temp=_view_temp,
        alien=_alien,
        initial_report=_initial_report, initial_deploy=_initial_deploy, initial_option=_initial_option,
        adb=_adb, ffmpeg=_ffmpeg, ffprobe=_ffprobe, scrcpy=_scrcpy,
    )

    if cmd_lines.stack and len(cmd_lines.stack) > 0:
        members = len(cmd_lines.stack)
        if members == 1:
            missions.video_dir_task(cmd_lines.stack[0])
        else:
            processes = members if members <= cpu else cpu
            with Pool(processes=processes, initializer=worker_init, initargs=("ERROR",)) as pool:
                results = pool.starmap(missions.video_dir_task, [(i,) for i in cmd_lines.stack])
            template_total = get_template(missions.view_total_temp) if missions.quick else get_template(
                missions.main_total_temp)
            Report.merge_report(results, template_total, missions.quick)
        sys.exit(0)
    elif cmd_lines.video and len(cmd_lines.video) > 0:
        members = len(cmd_lines.video)
        if members == 1:
            missions.video_task(cmd_lines.video[0])
        else:
            processes = members if members <= cpu else cpu
            with Pool(processes=processes, initializer=worker_init, initargs=("ERROR",)) as pool:
                results = pool.starmap(missions.video_task, [(i,) for i in cmd_lines.video])
            template_total = get_template(missions.view_total_temp) if missions.quick else get_template(
                missions.main_total_temp)
            Report.merge_report(results, template_total, missions.quick)
        sys.exit(0)
    elif cmd_lines.train and len(cmd_lines.train) > 0:
        members = len(cmd_lines.train)
        if members == 1:
            missions.train_model(cmd_lines.train[0])
        else:
            processes = members if members <= cpu else cpu
            with Pool(processes=processes, initializer=worker_init, initargs=("ERROR",)) as pool:
                pool.starmap(missions.train_model, [(i,) for i in cmd_lines.train])
        sys.exit(0)
    elif cmd_lines.build and len(cmd_lines.build) > 0:
        members = len(cmd_lines.build)
        if members == 1:
            missions.build_model(cmd_lines.build[0])
        else:
            processes = members if members <= cpu else cpu
            with Pool(processes=processes, initializer=worker_init, initargs=("ERROR",)) as pool:
                pool.starmap(missions.build_model, [(i,) for i in cmd_lines.build])
        sys.exit(0)
    else:
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(ask_main())
            sys.exit(0)
        except KeyboardInterrupt:
            sys.exit(0)
