import os
import cv2
import random
import asyncio
from pathlib import Path
from loguru import logger
from typing import Union, Optional
from concurrent.futures import ThreadPoolExecutor
from engine.record import Record
from engine.player import Player
from engine.switch import Switch
from nexaflow import toolbox
from nexaflow.report import Report
from nexaflow.cutter.cutter import VideoCutter
from nexaflow.video import VideoObject, Frame
from nexaflow.classifier.keras_classifier import KerasClassifier
from nexaflow.hook import BaseHook, CropHook, OmitHook, FrameSaveHook, CompressHook
from nexaflow.classifier.base import ClassifierResult, SingleClassifierResult


class Review(object):

    data = tuple()

    def __init__(self, start: int, end: int, cost: float, classifier: "ClassifierResult" = None):
        self.data = start, end, cost, classifier

    def __str__(self):
        start, end, cost, classifier = self.data
        kc = "KC" if classifier else "None"
        return f"<Review start={start} end={end} cost={cost} classifier={kc}>"

    __repr__ = __str__


class Filmer(object):

    def train_model(self, video_file: str, **kwargs):
        pass

    def build_model(self, src: str):
        pass


class Alynex(object):

    def __init__(
            self,
            report: Report,
            model_file: str,
            model_shape: tuple = (256, 256),
            model_aisle: int = 1
    ):

        self.__kc = KerasClassifier(data_size=model_shape, aisle=model_aisle)
        self.__kc.load_model(model_file)
        self.__cliper: list["BaseHook"] = []
        self.__report: Optional["Report"] = report
        self.__player: Optional["Player"] = Player()
        self.__record: Optional["Record"] = Record()
        self.__switch: Optional["Switch"] = Switch()

    @property
    def kc(self) -> "KerasClassifier":
        return self.__kc

    @property
    def cliper(self) -> list["BaseHook"]:
        return self.__cliper

    @property
    def report(self) -> "Report":
        return self.__report

    @property
    def player(self) -> "Player":
        return self.__player

    @property
    def record(self) -> "Record":
        return self.__record

    @property
    def switch(self) -> "Switch":
        return self.__switch

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

    def crop_hook(self, x: int | float, y: int | float, x_size: int | float, y_size: int | float) -> None:
        hook = CropHook((y_size, x_size), (y, x))
        self.__cliper.append(hook)

    def omit_hook(self, x: int | float, y: int | float, x_size: int | float, y_size: int | float) -> None:
        hook = OmitHook((y_size, x_size), (y, x))
        self.__cliper.append(hook)

    def analyzer(self, template_file: str, **kwargs) -> Optional["Review"]:
        alone = kwargs.get("alone", False)
        group = kwargs.get("group", False)
        boost = kwargs.get("boost", False)
        color = kwargs.get("color", False)

        shape = kwargs.get("shape", None)
        scale = kwargs.get("scale", None)
        start = kwargs.get("start", None)
        close = kwargs.get("close", None)
        limit = kwargs.get("limit", None)

        begin = kwargs.get("begin", (0, 1))
        final = kwargs.get("final", (-1, -1))

        crops = kwargs.get("crops", [])
        omits = kwargs.get("omits", [])

        frate = kwargs.get("fps", 60)
        thres = kwargs.get("threshold", 0.97)
        shift = kwargs.get("offset", 3)
        block = kwargs.get("block", 6)

        def validate():
            screen_tag, screen_cap = None, None
            if os.path.isfile(self.report.video_path):
                screen = cv2.VideoCapture(self.report.video_path)
                if screen.isOpened():
                    screen_tag = os.path.basename(self.report.video_path)
                    screen_cap = self.report.video_path
                screen.release()
            elif os.path.isdir(self.report.video_path):
                if len(
                        file_list := [
                            file for file in os.listdir(self.report.video_path) if os.path.isfile(
                                os.path.join(self.report.video_path, file)
                            )
                        ]
                ) > 1 or len(file_list) == 1:
                    screen = cv2.VideoCapture(os.path.join(self.report.video_path, file_list[0]))
                    if screen.isOpened():
                        screen_tag = os.path.basename(file_list[0])
                        screen_cap = os.path.join(self.report.video_path, file_list[0])
                    screen.release()
            return screen_tag, screen_cap

        def frame_flip():
            change_record = os.path.join(
                os.path.dirname(screen_record),
                f"screen_fps{frate}_{random.randint(100, 999)}.mp4"
            )
            asyncio.run(
                self.switch.ask_video_change("ffmpeg", frate, screen_record, change_record)
            )
            logger.info(f"视频转换完成: {Path(change_record).name}")
            os.remove(screen_record)
            logger.info(f"移除旧的视频: {Path(screen_record).name}")

            if shape:
                original_shape = asyncio.run(Switch.ask_video_larger("ffprobe", change_record))
                w, h, ratio = asyncio.run(Switch.ask_magic_frame(original_shape, shape))
                target_shape = w, h
                target_scale = scale
                logger.info(f"调整宽高比: {w} x {h}")
            elif scale:
                target_shape = shape
                target_scale = max(0.1, min(1.0, scale))
            else:
                target_shape = shape
                target_scale = 0.4

            video = VideoObject(change_record)
            task, hued = video.load_frames(
                silently_load_hued=color,
                not_transform_gray=False,
                shape=target_shape,
                scale=target_scale
            )
            return video, task, hued

        def frame_flow():
            video, task, hued = frame_flip()
            cutter = VideoCutter()

            compress_hook = CompressHook(1, None, False)
            cutter.add_hook(compress_hook)

            for hook in self.cliper:
                cutter.add_hook(hook)

            save_hook = FrameSaveHook(self.report.extra_path)
            cutter.add_hook(save_hook)

            res = cutter.cut(
                video=video, block=block
            )

            stable, unstable = res.get_range(
                threshold=thres, offset=shift
            )

            file_list = os.listdir(self.report.extra_path)
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
                    os.remove(os.path.join(self.report.extra_path, file))

            for draw in os.listdir(self.report.extra_path):
                toolbox.draw_line(os.path.join(self.report.extra_path, draw))

            classify = self.kc.classify(
                video=video, valid_range=stable, keep_data=True
            )

            important_frames = classify.get_important_frame_list()

            pbar = toolbox.show_progress(classify.get_length(), 50, "Faster")
            frames_list = []
            if boost:
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

            if color:
                video.hued_data = tuple(hued.result())
                logger.info(f"彩色帧已加载: {video.frame_details(video.hued_data)}")
                task.shutdown()
                frames = [video.hued_data[frame.frame_id - 1] for frame in frames_list]
            else:
                frames = [frame for frame in frames_list]

            return classify, frames

        def frame_flick(classify):
            logger.info(f"阶段划分: {classify.get_ordered_stage_set()}")
            begin_stage, begin_frame = begin
            final_stage, final_frame = final
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
                    logger.info(f"[bold]第 {i:02} 个非稳定阶段")
                    logger.info(f"[bold]{'=' * 30}")
                    for j, frame in enumerate(unstable_stage):
                        logger.info(f"[bold]第 {j:05} 帧: {frame}")
                    logger.info(f"[bold]{'=' * 30}\n")
                start_frame = classify.get_important_frame_list()[0]
                end_frame = classify.get_important_frame_list()[-1]
                logger.warning(f"Framix Analyzer recalculate ...")

            if start_frame == end_frame:
                logger.warning(f"{start_frame} == {end_frame}")
                start_frame, end_frame = classify.data[0], classify.data[-1]
                logger.warning(f"Framix Analyzer recalculate ...")

            time_cost = end_frame.timestamp - start_frame.timestamp
            logger.info(
                f"图像分类结果: [开始帧: {start_frame.timestamp:.5f}] [结束帧: {end_frame.timestamp:.5f}] [总耗时: {time_cost:.5f}]"
            )

            original_inform = self.report.draw(
                classifier_result=classify,
                proto_path=self.report.proto_path,
                template_file=self.report.get_template(template_file)
            )
            result = {
                "total_path": self.report.total_path,
                "title": self.report.title,
                "query_path": self.report.query_path,
                "query": self.report.query,
                "stage": {"start": start_frame.frame_id, "end": end_frame.frame_id, "cost": f"{time_cost:.5f}"},
                "frame": self.report.frame_path,
                "extra": self.report.extra_path,
                "proto": original_inform
            }
            logger.debug(f"Restore: {result}")
            self.report.load(result)
            return start_frame.frame_id, end_frame.frame_id, time_cost

        def frame_forge(frame: Union[SingleClassifierResult | Frame]):
            short_timestamp = format(round(frame.timestamp, 5), ".5f")
            pic_name = f"{frame.frame_id}_{short_timestamp}.png"
            pic_path = os.path.join(self.report.frame_path, pic_name)
            # cv2.imwrite(pic_path, frame.data)
            # logger.debug(f"frame saved to {pic_path}")
            cv2.imencode(".png", frame.data)[1].tofile(pic_path)

        def analytics():
            classify, frames = frame_flow()
            with ThreadPoolExecutor() as executor:
                executor.map(frame_forge, [frame for frame in frames])
                future = executor.submit(frame_flick, classify)
            return future.result()

        tag, screen_record = validate()
        if not tag or not screen_record:
            return logger.error(f"{tag} 不是一个标准的mp4视频文件，或视频文件已损坏 ...")
        logger.info(f"{tag} 可正常播放，准备加载视频 ...")

        start, end, cost = analytics()
        self.__cliper.clear()
        return Review(start, end, cost)


if __name__ == '__main__':
    pass
