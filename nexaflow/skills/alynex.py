import os
import cv2
import time
import random
import asyncio
from loguru import logger
from typing import List, Union, Optional
from concurrent.futures import ThreadPoolExecutor
from nexaflow import toolbox
from nexaflow.skills.report import Report
from nexaflow.skills.record import Record
from nexaflow.skills.player import Player
from nexaflow.skills.switch import Switch
from nexaflow.cutter.cutter import VideoCutter
from nexaflow.video import VideoObject, Frame
from nexaflow.classifier.keras_classifier import KerasClassifier
from nexaflow.hook import BaseHook, CropHook, OmitHook, FrameSaveHook
from nexaflow.classifier.base import ClassifierResult, SingleClassifierResult


class Alynex(object):

    target_size: tuple = (350, 700)
    fps: int = 60
    step: int = 1
    block: int = 6
    threshold: Union[int | float] = 0.97
    offset: int = 3
    compress_rate: float = 0.5
    window_size: int = 1
    window_coefficient: int = 2

    kc: KerasClassifier = KerasClassifier(
        target_size=target_size, data_size=target_size
    )

    def __init__(self):
        self.__report: Optional[Report] = None
        self.__record: Optional[Record] = Record()
        self.__player: Optional[Player] = Player()
        self.__ffmpeg: Optional[Switch] = Switch()
        self.__filmer: Optional[Alynex._Filmer] = Alynex._Filmer()
        self.__framix: Optional[Alynex._Framix] = None

    def __str__(self):
        return (f"""
        <Alynex for NexaFlow
        Target Size: {self.target_size}
        Fps: {self.fps}
        Step: {self.step}
        Block: {self.block}
        Threshold: {self.threshold}
        Offset: {self.offset}
        Compress Rate: {self.compress_rate}
        Window Size: {self.window_size}
        Window Coefficient: {self.window_coefficient}
        >
        """)

    __repr__ = __str__

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def report(self) -> "Report":
        assert self.__report, f"{self.activate.__name__} first ..."
        return self.__report

    @property
    def record(self) -> "Record":
        return self.__record

    @property
    def player(self) -> "Player":
        return self.__player

    @property
    def ffmpeg(self) -> "Switch":
        return self.__ffmpeg

    @property
    def filmer(self) -> "Alynex._Filmer":
        return self.__filmer

    @property
    def framix(self) -> "Alynex._Framix":
        assert self.__framix, f"{self.activate.__name__} first ..."
        return self.__framix

    @staticmethod
    def only_video(folder: str) -> List:

        class Entry(object):

            def __init__(self, title: str, place: str, sheet: list):
                self.title = title
                self.place = place
                self.sheet = sheet

        return [
            Entry(
                os.path.basename(root), root,
                [os.path.join(root, f) for f in sorted(file)]
            )
            for root, _, file in os.walk(folder) if file
        ]

    def activate(self, models: str, total_path: str, write_log: bool = True):
        if not self.__report:
            self.__report = Report(total_path, write_log)
            self.__framix = Alynex._Framix(self.report)
            Alynex.kc.load_model(models)

    class _Filmer(object):

        @staticmethod
        def train_model(video_file: str) -> None:
            model_path = os.path.join(
                os.path.dirname(video_file),
                f"Model_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}"
            )
            if not os.path.exists(model_path):
                os.makedirs(model_path, exist_ok=True)

            # 将视频切分成帧
            video = VideoObject(video_file, fps=Alynex.fps)
            # 新建帧，计算视频总共有多少帧，每帧多少ms
            video.load_frames()
            # 压缩视频
            cutter = VideoCutter(
                target_size=Alynex.target_size
            )
            # 计算每一帧视频的每一个block的ssim和峰值信噪比
            res = cutter.cut(
                video=video,
                block=Alynex.block,
                window_size=Alynex.window_size,
                window_coefficient=Alynex.window_coefficient
            )
            # 计算出判断A帧到B帧之间是稳定还是不稳定
            stable, unstable = res.get_range(
                threshold=Alynex.threshold,
                offset=Alynex.offset
            )
            # 保存分类后的图片
            res.pick_and_save(
                range_list=stable,
                frame_count=20,
                to_dir=model_path,
                meaningful_name=True
            )

        @staticmethod
        def build_model(src: str) -> None:
            new_model_path = os.path.join(src, f"Create_Model_{time.strftime('%Y%m%d%H%M%S')}")
            new_model_name = f"Keras_Model_{random.randint(10000, 99999)}.h5"
            final_model = os.path.join(new_model_path, new_model_name)
            if not os.path.exists(new_model_path):
                os.makedirs(new_model_path, exist_ok=True)

            Alynex.kc.train(src)
            Alynex.kc.save_model(final_model, overwrite=True)

    class _Framix(object):

        def __init__(self, report: "Report"):
            self.__framix_list: List["BaseHook"] = []
            self.__reporter = report

        @property
        def framix_list(self) -> List["BaseHook"]:
            return self.__framix_list

        def crop_hook(
                self,
                x: Union[int | float], y: Union[int | float],
                x_size: Union[int | float], y_size: Union[int | float]
        ) -> None:
            """获取区域"""
            hook = CropHook((y_size, x_size), (y, x))
            self.framix_list.append(hook)

        def omit_hook(
                self,
                x: Union[int | float], y: Union[int | float],
                x_size: Union[int | float], y_size: Union[int | float]
        ) -> None:
            """忽略区域"""
            hook = OmitHook((y_size, x_size), (y, x))
            self.framix_list.append(hook)

        def pixel_wizard(self, video: "VideoObject") -> "ClassifierResult":

            cutter = VideoCutter(
                target_size=Alynex.target_size
            )

            # 应用视频帧处理单元
            for mix in self.framix_list:
                cutter.add_hook(mix)

            save_hook = FrameSaveHook(self.__reporter.extra_path)
            cutter.add_hook(save_hook)

            # 计算每一帧视频的每一个block的ssim和峰值信噪比
            res = cutter.cut(
                video=video,
                block=Alynex.block,
                window_size=Alynex.window_size,
                window_coefficient=Alynex.window_coefficient
            )
            # 计算出判断A帧到B帧之间是稳定还是不稳定
            stable, unstable = res.get_range(
                threshold=Alynex.threshold,
                offset=Alynex.offset
            )

            # 保存十二张hook图
            files = os.listdir(self.__reporter.extra_path)
            files.sort(key=lambda x: int(x.split("(")[0]))
            total_images = len(files)
            interval = total_images // 11 if total_images > 12 else 1
            for index, file in enumerate(files):
                if index % interval != 0:
                    os.remove(
                        os.path.join(self.__reporter.extra_path, file)
                    )

            # 为图片绘制线条
            draws = os.listdir(self.__reporter.extra_path)
            for draw in draws:
                toolbox.draw_line(
                    os.path.join(self.__reporter.extra_path, draw)
                )

            # 开始图像分类
            classify = Alynex.kc.classify(video=video, valid_range=stable, keep_data=True)
            return classify

    class _Review(object):

        def __init__(self, *args: str):
            self.start, self.end, self.cost, *_ = args

        def __str__(self):
            return f"<Review Start: {self.start} End: {self.end} Cost: {self.cost}>"

        __repr__ = __str__

    def analyzer(
            self, boost: bool = True, color: bool = True, focus: bool = False, **kwargs
    ) -> Optional["Alynex._Review"]:
        """
        智能分类帧数据
        :param boost: 跳帧模式
        :param color: 彩色模式
        :param focus: 转换视频
        :param kwargs: 视频分析配置
        :return: 分析结果
        """

        self.step = kwargs.get("step", 1)
        self.block = kwargs.get("block", 6)
        self.threshold = kwargs.get("threshold", 0.97)
        self.offset = kwargs.get("threshold", 3)
        self.compress_rate = kwargs.get("compress_rate", 0.5)
        self.window_size = kwargs.get("window_size", 1)
        self.window_coefficient = kwargs.get("window_coefficient", 2)

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
            if focus:
                change_record = os.path.join(
                    os.path.dirname(screen_record), f"screen_fps60_{random.randint(100, 999)}.mp4"
                )
                asyncio.run(self.ffmpeg.video_change(screen_record, change_record))
                logger.info(f"视频转换完成: {os.path.basename(change_record)}")
                os.remove(screen_record)
                logger.info(f"移除旧的视频: {os.path.basename(screen_record)}")
            else:
                change_record = screen_record

            video = VideoObject(change_record)
            task, hued = video.load_frames(color)
            return video, task, hued

        def frame_flow():
            video, task, hued = frame_flip()
            classify = self.framix.pixel_wizard(video)
            important_frames: List["SingleClassifierResult"] = classify.get_important_frame_list()

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
            before, after, final = f"{start_frame.timestamp:.5f}", f"{end_frame.timestamp:.5f}", f"{time_cost:.5f}"
            logger.info(f"图像分类结果: [开始帧: {before}] [结束帧: {after}] [总耗时: {final}]")

            original_inform = self.report.draw(
                classifier_result=classify,
                proto_path=self.report.proto_path,
                target_size=Alynex.target_size
            )
            result = {
                "total_path": self.report.total_path,
                "title": self.report.title,
                "query_path": self.report.query_path,
                "query": self.report.query,
                "stage": {
                    "start": start_frame.frame_id,
                    "end": end_frame.frame_id,
                    "cost": f"{time_cost:.5f}"
                },
                "frame": self.report.frame_path,
                "extra": self.report.extra_path,
                "proto": original_inform
            }
            logger.debug(f"Restore: {result}")
            self.report.load(result)
            return before, after, final

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
            logger.error(f"{tag} 不是一个标准的mp4视频文件，或视频文件已损坏 ...")
            return None
        logger.info(f"{tag} 可正常播放，准备加载视频 ...")

        start, end, cost = analytics()
        return Alynex._Review(start, end, cost)


if __name__ == '__main__':
    pass
