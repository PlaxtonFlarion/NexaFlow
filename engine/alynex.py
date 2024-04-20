import os
import re
import cv2
import time
import numpy
import random
import typing
import asyncio
import aiofiles
from pathlib import Path
from loguru import logger
from engine.switch import Switch
from engine.active import Review
from frameflow.skills.config import Deploy
from frameflow.skills.parser import Parser
from frameflow.skills.show import Show
from nexaflow import const
from nexaflow import toolbox
from nexaflow.report import Report
from nexaflow.video import VideoObject
from nexaflow.cutter.cutter import VideoCutter
from nexaflow.hook import FrameSaveHook
from nexaflow.hook import PaintCropHook, PaintOmitHook
from nexaflow.classifier.keras_classifier import KerasStruct


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


if __name__ == '__main__':
    pass
