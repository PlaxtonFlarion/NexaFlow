import os
import cv2
import asyncio
import aiofiles
from loguru import logger
from typing import List, Union
from nexaflow import toolbox
from nexaflow.constants import Constants
from nexaflow.skills.report import Report
from nexaflow.skills.record import Record
from nexaflow.skills.player import Player
from nexaflow.skills.switch import Switch
from nexaflow.cutter.cutter import VideoCutter
from nexaflow.video import VideoObject, Frame
from nexaflow.classifier.keras_classifier import KerasClassifier
from nexaflow.hook import BaseHook, CropHook, OmitHook, FrameSaveHook
from nexaflow.classifier.base import ClassifierResult, SingleClassifierResult

VIDEOS: str = os.path.join(Constants.WORK, "model", "model_video")
STABLE: str = os.path.join(Constants.WORK, "model", "stable")
MODELS: str = os.path.join(Constants.WORK, "model", "model.h5")


class Alynex(object):

    target_size: tuple = (350, 700)
    block: int = 6
    threshold: Union[int | float] = 0.97
    offset: int = 3
    compress_rate: float = 0.5
    window_size: int = 1
    window_coefficient: int = 2

    def __init__(self):
        self.__report: Report = Report()
        self.__record: Record = Record()
        self.__player: Player = Player()
        self.__ffmpeg: Switch = Switch()
        self.__framix: Alynex._Framix = Alynex._Framix()
        self.__filmer: Alynex._Filmer = Alynex._Filmer()

    def __str__(self):
        return (f"""
        <Alynex for NexaFlow
        Target Size: {self.target_size}
        Block: {self.block}
        Threshold: {self.threshold}
        Offset: {self.offset}
        Compress Rate: {self.compress_rate}
        Window Size: {self.window_size}
        Window Coefficient: {self.window_coefficient}
        >
        """)

    __repr__ = __str__

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def report(self) -> "Report":
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
    def framix(self) -> "Alynex._Framix":
        return self.__framix

    @property
    def filmer(self) -> "Alynex._Filmer":
        return self.__filmer

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
            for root, _, file in os.walk(
                os.path.join(Constants.WORK, "data", folder)
            ) if file
        ]

    class _Filmer(object):

        @staticmethod
        def train_model() -> None:
            """
            :1: scrcpy --no-audio --record file.mp4
            :2: ffmpeg -i file.mp4 -t 5 model.mp4
            :3: ffmpeg -i file.mp4 -t 5 -c copy model.mp4
            :4: ffmpeg -i file.mp4 -ss 00:00:00 -t 00:00:05 -c copy model.mp4
            """
            # 将视频切分成帧
            video = VideoObject(VIDEOS, fps=60)
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
                to_dir=STABLE,
                meaningful_name=True
            )

        @staticmethod
        def build_model() -> None:
            # 从分类后的图片构建模型
            cl = KerasClassifier(target_size=Alynex.target_size)
            cl.train(STABLE)
            cl.save_model(MODELS, overwrite=True)

    class _Framix(object):

        def __init__(self):
            self.framix_list: List["BaseHook"] = []

        async def crop_hook(
                self,
                x: Union[int | float], y: Union[int | float],
                x_size: Union[int | float], y_size: Union[int | float]
        ) -> None:
            """获取区域"""
            hook = CropHook((y_size, x_size), (y, x))
            self.framix_list.append(hook)

        async def omit_hook(
                self,
                x: Union[int | float], y: Union[int | float],
                x_size: Union[int | float], y_size: Union[int | float]
        ) -> None:
            """忽略区域"""
            hook = OmitHook((y_size, x_size), (y, x))
            self.framix_list.append(hook)

        async def pixel_wizard(
                self,
                video: "VideoObject",
                model: str,
                extra_path: str
        ) -> "ClassifierResult":

            cutter = VideoCutter(
                target_size=Alynex.target_size
            )

            # 应用视频帧处理单元
            for mix in self.framix_list:
                cutter.add_hook(mix)

            save_hook = FrameSaveHook(extra_path)
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
            files = os.listdir(extra_path)
            files.sort(key=lambda x: int(x.split("(")[0]))
            total_images = len(files)
            interval = total_images // 11 if total_images > 12 else 1
            for index, file in enumerate(files):
                if index % interval != 0:
                    os.remove(
                        os.path.join(extra_path, file)
                    )

            # 为图片绘制线条
            draws = os.listdir(extra_path)
            for draw in draws:
                toolbox.draw_line(
                    os.path.join(extra_path, draw)
                )

            # 开始图像分类
            cl = KerasClassifier(target_size=Alynex.target_size)
            cl.load_model(model)
            classify = cl.classify(video=video, valid_range=stable, keep_data=True)

            return classify

    class _Review(object):

        def __init__(self, *args: str):
            self.start, self.end, self.cost, *_ = args

        def __str__(self):
            return f"<Review Start: {self.start} End: {self.end} Cost: {self.cost}>"

        __repr__ = __str__

    async def analyzer(
            self, boost: bool = True, color: bool = True, shift: bool = False, **kwargs
    ) -> "Alynex._Review":
        """
        智能分类帧数据
        :param boost: 快速分析
        :param color: 彩色帧
        :param shift: 转换视频格式和帧率
        :param kwargs: 分析配置
        :return: 分析结果
        """

        async def ini():
            self.block = kwargs.get("block", 6)
            self.threshold = kwargs.get("threshold", 0.97)
            self.offset = kwargs.get("threshold", 3)
            self.compress_rate = kwargs.get("compress_rate", 0.5)
            self.window_size = kwargs.get("window_size", 1)
            self.window_coefficient = kwargs.get("window_coefficient", 2)

        async def validate():
            return next(
                ((file_path, file) for file in os.listdir(self.report.video_path)
                 if cv2.VideoCapture(file_path := os.path.join(self.report.video_path, file)).release() or True),
                None
            )

        async def exchange():
            screen_record, screen_tag = await validate()
            if not screen_record:
                logger.error(f"{screen_tag} 不是一个标准的mp4视频文件，或视频文件已损坏 ...")
                return None
            logger.info(f"{screen_tag} 可正常播放，准备加载视频 ...")
            if shift:
                change_record = screen_record.split('.')[0] + ".mp4"
                await self.ffmpeg.video_change(screen_record, change_record)
                logger.info(f"视频转换完成: {change_record}")
                os.remove(screen_record)
                logger.info(f"移除旧的视频: {screen_record}")
            else:
                change_record = screen_record
            return change_record

        async def loading():
            change_record = await exchange()
            video = VideoObject(change_record)
            task, hued = video.load_frames(color)
            return video, task, hued

        async def intelligence():
            (video, task, hued), *_ = await asyncio.gather(loading(), ini())
            classify = await self.framix.pixel_wizard(video, MODELS, self.report.extra_path)
            important_frames: List["SingleClassifierResult"] = classify.get_important_frame_list()

            pbar = toolbox.show_progress(classify.get_length(), 50, "Faster   ")
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

            frames = await color_mode(video, task, hued, frames_list)
            return classify, frames_list, frames

        async def color_mode(video, task, hued, frames_list):
            if color:
                video.hued_data = tuple(hued.result())
                logger.info(f"彩色帧已加载: {video.frame_details(video.hued_data)}")
                task.shutdown()
                frames = [video.hued_data[frame.frame_id - 1] for frame in frames_list]
            else:
                frames = [frame for frame in frames_list]
            return frames

        async def sort_frames(classify, frames_list):
            start_frame = classify.get_not_stable_stage_range()[0][1]
            end_frame = classify.get_not_stable_stage_range()[-1][-1]
            if start_frame.frame_id == end_frame.frame_id:
                start_frame = frames_list[0]
                end_frame = frames_list[-1]

            time_cost = end_frame.timestamp - start_frame.timestamp
            before, after, final = f"{start_frame.timestamp:.5f}", f"{end_frame.timestamp:.5f}", f"{time_cost:.5f}"
            logger.info(f"图像分类结果: [开始帧: {before}] [结束帧: {after}] [总耗时: {final}]")

            original_inform = await self.report.draw(
                classifier_result=classify,
                proto_path=self.report.proto_path,
                target_size=Alynex.target_size
            )
            result = {
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
            await self.report.load(result)
            return before, after, final

        async def keep_images(frame: Union[SingleClassifierResult | Frame]):
            short_timestamp = format(round(frame.timestamp, 5), ".5f")
            pic_name = f"{frame.frame_id}_{short_timestamp}.png"
            pic_path = os.path.join(self.report.frame_path, pic_name)
            _, codec = cv2.imencode(".png", frame.data)
            async with aiofiles.open(pic_path, "wb") as file:
                await file.write(codec.tobytes())

        async def well_done():
            classify, frames_list, frames = await intelligence()
            result, *_ = await asyncio.gather(
                sort_frames(classify, frames_list),
                *(keep_images(frame) for frame in frames)
            )
            return result

        # loop = asyncio.get_event_loop()
        # start, end, cost = loop.run_until_complete(well_done())
        try:
            start, end, cost = await well_done()
        except asyncio.exceptions.CancelledError:
            start, end, cost = await well_done()

        return Alynex._Review(start, end, cost)


if __name__ == '__main__':
    pass
