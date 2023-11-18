import os
import cv2
import time
import typing
import tempfile
import numpy as np
import imageio_ffmpeg
import moviepy.editor as mpy
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from nexaflow import toolbox


class Frame(object):

    def __init__(self, frame_id: int, timestamp: float, data: np.ndarray):
        self.frame_id: int = frame_id
        self.timestamp: float = timestamp
        self.data: np.ndarray = data

    @staticmethod
    def initial(cap: cv2.VideoCapture, frame: np.ndarray) -> "Frame":
        raise NotImplementedError

    def copy(self) -> "Frame":
        raise NotImplementedError


class ColorFrame(Frame):

    def __init__(self, frame_id: int, timestamp: float, data: np.ndarray):
        super().__init__(frame_id, timestamp, data)

    def __str__(self):
        return f"<ColorFrame id={self.frame_id} timestamp={self.timestamp}>"

    @staticmethod
    def initial(cap: cv2.VideoCapture, frame: np.ndarray) -> "ColorFrame":
        frame_id = toolbox.get_current_frame_id(cap)
        timestamp = toolbox.get_current_frame_time(cap)
        new_frame = toolbox.compress_frame(frame, 0.5, (350, 700), True)
        return ColorFrame(frame_id, timestamp, new_frame)

    def copy(self) -> "ColorFrame":
        return ColorFrame(self.frame_id, self.timestamp, self.data[:])


class VideoFrame(Frame):

    def __init__(self, frame_id: int, timestamp: float, data: np.ndarray):
        super().__init__(frame_id, timestamp, data)

    def __str__(self):
        return f"<VideoFrame id={self.frame_id} timestamp={self.timestamp}>"

    @staticmethod
    def initial(cap: cv2.VideoCapture, frame: np.ndarray) -> "VideoFrame":
        frame_id = toolbox.get_current_frame_id(cap)
        timestamp = toolbox.get_current_frame_time(cap)
        new_frame = toolbox.compress_frame(frame, 0.5, (350, 700), False)
        return VideoFrame(frame_id, timestamp, new_frame)

    def copy(self) -> "VideoFrame":
        return VideoFrame(self.frame_id, self.timestamp, self.data[:])

    def contain_image(
        self, *, image_path: str = None, image_object: np.ndarray = None, **kwargs
    ) -> typing.Dict[str, typing.Any]:
        """
        检查给定图像（通过路径或numpy对象）是否存在于当前帧中，并返回匹配的字典
        """
        assert image_path or (
            image_object is not None
        ), "should fill image_path or image_object"

        if image_path:
            logger.debug(f"found image path, use it first: {image_path}")
            return toolbox.match_template_with_path(image_path, self.data, **kwargs)
        image_object = toolbox.turn_grey(image_object)
        return toolbox.match_template_with_object(image_object, self.data, **kwargs)


class _BaseFrameOperator(object):

    def __init__(self, video: "VideoObject"):
        """
        初始化，接受一个`VideoObject`作为参数
        """
        self.cur_ptr: int = 0
        self.video: "VideoObject" = video

    def get_frame_by_id(self, frame_id: int) -> typing.Optional["VideoFrame"]:
        """
        抽象方法，需要在子类中实现。用于获取特定ID的帧
        """
        raise NotImplementedError

    def get_length(self) -> int:
        """
        返回视频的帧数
        """
        return self.video.frame_count


class MemFrameOperator(_BaseFrameOperator):

    def get_frame_by_id(self, frame_id: int) -> typing.Optional["VideoFrame"]:
        """
        从内存中获取特定ID的帧
        """
        if frame_id > self.get_length():
            return None

        frame_id = frame_id - 1
        return self.video.grey_data[frame_id].copy()


class FileFrameOperator(_BaseFrameOperator):

    def get_frame_by_id(self, frame_id: int) -> typing.Optional["VideoFrame"]:
        """
        从文件中读取特定ID的帧
        """
        if frame_id > self.get_length():
            return None
        with toolbox.video_capture(self.video.path) as cap:
            toolbox.video_jump(cap, frame_id)
            success, frame = cap.read()
            video_frame = VideoFrame.initial(cap, frame) if success else None
        return video_frame


class VideoObject(object):

    def __init__(
        self,
        path: typing.Union[str, os.PathLike],
        fps: int = None,
    ):
        """
        初始化，检查文件路径是否有效，执行其他一些初始化操作
        """
        assert os.path.isfile(path), f"video {path} not existed"
        self.path: str = str(path)
        self.grey_data: typing.Optional[typing.Tuple["VideoFrame"]] = tuple()  # 灰度帧
        self.hued_data: typing.Optional[typing.Tuple["ColorFrame"]] = tuple()  # 彩色帧

        if fps:
            video_path = os.path.join(tempfile.mkdtemp(), f"tmp_{fps}.mp4")
            logger.debug(f"convert video, and bind path to {video_path}")
            logger.info(f"转换视频: {video_path}")
            toolbox.fps_convert(
                fps, self.path, video_path, imageio_ffmpeg.get_ffmpeg_exe()
            )
            self.path = video_path

        with toolbox.video_capture(self.path) as cap:
            self.frame_count = toolbox.get_frame_count(cap)
            self.frame_size = toolbox.get_frame_size(cap)

        logger.info(f"视频已生成，视频帧长度: {self.frame_count} 分辨率: {self.frame_size}")

    def __str__(self):
        return f"<VideoObject path={self.path}>"

    __repr__ = __str__

    def sync_timestamp(self, frame_data: tuple[VideoFrame]) -> None:
        assert frame_data, "load_frames() first"
        vid = mpy.VideoFileClip(self.path)

        vid_count = vid.reader.nframes
        pbar = toolbox.show_progress(vid_count, 153, "Synzer   ")
        for frame_id, (timestamp, _) in enumerate(vid.iter_frames(with_times=True)):
            if frame_id >= len(frame_data):
                break
            # frame_id_real = frame_id + 1
            if not frame_data[frame_id].timestamp:
                # logger.debug(f"fix frame {frame_id_real}'s timestamp: {timestamp}")
                frame_data[frame_id].timestamp = timestamp
            pbar.update(1)
        pbar.close()

    def sync_backstage(self, frame_data: tuple[ColorFrame]) -> None:
        assert frame_data, "load_frames() first"
        vid = mpy.VideoFileClip(self.path)

        for frame_id, (timestamp, _) in enumerate(vid.iter_frames(with_times=True)):
            if frame_id >= len(frame_data):
                break
            # frame_id_real = frame_id + 1
            if not frame_data[frame_id].timestamp:
                # logger.debug(f"fix frame {frame_id_real}'s timestamp: {timestamp}")
                frame_data[frame_id].timestamp = timestamp

    def clean_frames(self):
        """
        清除所有帧数据
        """
        self.grey_data = tuple()
        self.hued_data = tuple()

    @staticmethod
    def frame_details(frame_type):
        each_cost = frame_type[0].data.nbytes / (1024 ** 2)
        total_cost = each_cost * len(frame_type)
        frame_size = frame_type[0].data.shape[::-1]
        return f"{frame_type[0].__class__.__name__}: [{each_cost:.2f} MB] [{total_cost:.2f} MB] {frame_size}"

    def load_frames(self, color: bool = False):
        """
        从文件中加载所有帧到内存
        """
        logger.info(f"加载视频帧到内存: {self.path}")

        def load_stream(frames: type[VideoFrame]):
            pbar = toolbox.show_progress(self.frame_count, 180, "Loader   ")
            data: list[VideoFrame] = []
            with toolbox.video_capture(self.path) as cap:
                for success, frame in iter(lambda: cap.read(), (False, None)):
                    if success:
                        data.append(frames.initial(cap, frame))
                        pbar.update(1)
            pbar.close()
            return data

        def back_ground(frames: type[ColorFrame]):
            data: list[ColorFrame] = []
            with toolbox.video_capture(self.path) as cap:
                for success, frame in iter(lambda: cap.read(), (False, None)):
                    if success:
                        data.append(frames.initial(cap, frame))
            return data

        def load_stream_sync(brand):
            self.sync_timestamp(tuple(frame_data := load_stream(brand)))
            return frame_data

        def back_ground_sync(brand):
            self.sync_backstage(tuple(frame_data := back_ground(brand)))
            return frame_data

        start_time, task, hued = time.time(), None, None
        if color:
            task = ThreadPoolExecutor()
            hued = task.submit(back_ground_sync, ColorFrame)

        grey = load_stream_sync(VideoFrame)
        self.grey_data = tuple(grey)
        logger.info(f"灰度帧已加载: {self.frame_details(self.grey_data)}")
        logger.info(f"视频加载耗时: {time.time() - start_time:.2f} 秒")
        return task, hued

    def _read_from_file(self) -> typing.Generator["VideoFrame", None, None]:
        """
        从文件中读取帧
        """
        with toolbox.video_capture(self.path) as cap:
            success, frame = cap.read()
            while success:
                yield VideoFrame.initial(cap, frame)
                success, frame = cap.read()

    def _read_from_mem(self) -> typing.Generator["VideoFrame", None, None]:
        """
        从内存中读取帧
        """
        for each_frame in self.grey_data:
            yield each_frame

    def _read(self) -> typing.Generator["VideoFrame", None, None]:
        """
        选择从文件还是从内存中读取帧
        """
        if self.grey_data:
            yield from self._read_from_mem()
        else:
            yield from self._read_from_file()

    def get_iterator(self) -> typing.Generator["VideoFrame", None, None]:
        """
        获取帧的迭代器
        """
        return self._read()

    def get_operator(self) -> _BaseFrameOperator:
        """
        根据是否已经加载帧，返回相应的FrameOperator（`MemFrameOperator`或`FileFrameOperator`）
        """
        if self.grey_data:
            return MemFrameOperator(self)
        return FileFrameOperator(self)

    def __iter__(self):
        """
        返回一个用于迭代帧的迭代器
        """
        return self.get_iterator()


if __name__ == '__main__':
    pass
