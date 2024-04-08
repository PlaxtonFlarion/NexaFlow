import os
import re
import cv2
import time
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

    def train_model(self, video_file: str) -> None:
        pass

    def build_model(self, src: str):
        pass


class Alynex(object):

    model_size: tuple = (256, 256)
    fps: int = 60
    block: int = 6
    threshold: Union[int | float] = 0.97
    offset: int = 3
    compress_rate: float = 0.5

    kc: KerasClassifier = KerasClassifier(data_size=model_size, aisle=1)

    def __init__(self, model_file: str, report: Report):
        self.kc.load_model(model_file)
        self.cliper_list: list["BaseHook"] = []
        self.__report: Optional[Report] = report
        self.__player: Optional[Player] = Player()
        self.__record: Optional[Record] = Record()
        self.__switch: Optional[Switch] = Switch()

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
        self.cliper_list.append(hook)

    def omit_hook(self, x: int | float, y: int | float, x_size: int | float, y_size: int | float) -> None:

        hook = OmitHook((y_size, x_size), (y, x))
        self.cliper_list.append(hook)

    def analyzer(
            self, alien: str, boost: bool = True, color: bool = True, **kwargs
    ) -> Optional["Review"]:

        self.block = kwargs.get("block", 6)
        self.threshold = kwargs.get("threshold", 0.97)
        self.offset = kwargs.get("offset", 3)
        self.compress_rate = kwargs.get("compress_rate", 0.5)

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
                os.path.dirname(screen_record), f"screen_fps60_{random.randint(100, 999)}.mp4"
            )
            asyncio.run(
                self.switch.ask_video_change("ffmpeg", 60, screen_record, change_record)
            )
            logger.info(f"视频转换完成: {Path(change_record).name}")
            os.remove(screen_record)
            logger.info(f"移除旧的视频: {Path(screen_record).name}")

            video = VideoObject(change_record)
            task, hued = video.load_frames(
                silently_load_hued=color,
                not_transform_gray=False
            )
            return video, task, hued

        def frame_flow():
            video, task, hued = frame_flip()
            cutter = VideoCutter()

            compress_hook = CompressHook(1, None, False)
            cutter.add_hook(compress_hook)

            for hook in self.cliper_list:
                cutter.add_hook(hook)

            save_hook = FrameSaveHook(self.report.extra_path)
            cutter.add_hook(save_hook)

            res = cutter.cut(
                video=video,
                block=Alynex.block
            )

            stable, unstable = res.get_range(
                threshold=Alynex.threshold,
                offset=Alynex.offset
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

            classify = Alynex.kc.classify(
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
            logger.info(f"图像分类结果: [开始帧: {start_frame.timestamp:.5f}] [结束帧: {end_frame.timestamp:.5f}] [总耗时: {time_cost:.5f}]")

            original_inform = self.report.draw(
                classifier_result=classify,
                proto_path=self.report.proto_path,
                template_file=self.report.get_template(alien)
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

        self.cliper_list.clear()
        start, end, cost = analytics()
        return Review(start, end, cost)


if __name__ == '__main__':
    pass
