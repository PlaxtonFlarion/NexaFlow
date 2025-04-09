#
#  __     ___     _
#  \ \   / (_) __| | ___  ___
#   \ \ / /| |/ _` |/ _ \/ _ \
#    \ V / | | (_| |  __/ (_) |
#     \_/  |_|\__,_|\___|\___/
#

import os
import cv2
import typing
import tempfile
import numpy as np
import moviepy.editor as mpy
from loguru import logger
from nexaflow import toolbox


class VideoFrame(object):

    def __init__(self, frame_id: int, timestamp: float, data: np.ndarray):
        self.frame_id: int = frame_id
        self.timestamp: float = timestamp
        self.data: np.ndarray = data

    def __str__(self):
        return f"<VideoFrame id={self.frame_id} timestamp={self.timestamp}>"

    @staticmethod
    def initial(
            cap: cv2.VideoCapture,
            frame: np.ndarray,
            scale: typing.Optional[typing.Union[int, float]] = None,
            shape: typing.Optional[tuple] = None,
            color: typing.Optional[bool] = None,
    ) -> "VideoFrame":

        frame_id = toolbox.get_current_frame_id(cap)
        timestamp = toolbox.get_current_frame_time(cap)
        new_frame = toolbox.compress_frame(frame, scale, shape, color)
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
        return self.video.frames_data[frame_id].copy()


class DocFrameOperator(_BaseFrameOperator):

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
        path: typing.Union[str, "os.PathLike"],
        fps: typing.Optional[int] = None,
    ):
        """
        初始化，检查文件路径是否有效，执行其他一些初始化操作
        """
        assert os.path.isfile(path), f"video {path} not existed"
        self.name: str = str(os.path.basename(path))
        self.path: str = str(path)
        self.frames_data: typing.Optional[typing.Tuple["VideoFrame"]] = tuple()

        if fps:
            video_path = os.path.join(tempfile.mkdtemp(), f"tmp_{fps}.mp4")
            logger.debug(f"convert video, and bind path to {video_path}")
            logger.info(f"转换视频: {video_path}")
            toolbox.fps_convert(fps, self.path, video_path)
            self.path = video_path

        with toolbox.video_capture(self.path) as cap:
            self.frame_count = toolbox.get_frame_count(cap)
            self.frame_size = toolbox.get_frame_size(cap)

    def __str__(self):
        return f"<VideoObject name={self.name} path={self.path}>"

    __repr__ = __str__

    def sync_timestamp(self, frame_data: tuple["VideoFrame", ...]) -> None:
        assert frame_data, "load_frames() first"
        vid = mpy.VideoFileClip(self.path)

        vid_count = vid.reader.nframes
        progress_bar = toolbox.show_progress(total=vid_count, color=153)
        for frame_id, (timestamp, _) in enumerate(vid.iter_frames(with_times=True)):
            if frame_id >= len(frame_data):
                progress_bar.close()
                break
            # frame_id_real = frame_id + 1
            if not frame_data[frame_id].timestamp:
                # logger.debug(f"fix frame {frame_id_real}'s timestamp: {timestamp}")
                frame_data[frame_id].timestamp = timestamp
            progress_bar.update(1)

    def clean_frames(self):
        """
        清除所有帧数据
        """
        self.frames_data = tuple()

    @staticmethod
    def frame_details(frame_data: tuple["VideoFrame"]):
        frame = frame_data[0]
        every_cost = frame.data.nbytes / (1024 ** 2)
        total_cost = every_cost * len(frame_data)
        frame_size = frame.data.shape[::-1]
        frame_name = frame.__class__.__name__
        frame_info = f"[{every_cost:.2f} MB] [{total_cost:.2f} MB]"
        return f"{frame_name} {frame_info} {frame_size}"

    def load_frames(
            self, scale: int | float = None, shape: tuple = None, color: bool = False
    ):
        """
        从文件中加载所有帧到内存
        """

        progress_bar = toolbox.show_progress(total=self.frame_count, color=180)
        frame_data_list: list["VideoFrame"] = []
        with toolbox.video_capture(self.path) as cap:
            for success, frame in iter(lambda: cap.read(), (False, None)):
                if success:
                    frame_data_list.append(
                        VideoFrame.initial(cap, frame, scale, shape, color)
                    )
                    progress_bar.update(1)
        progress_bar.close()

        self.frames_data = tuple(frame_data_list)
        self.sync_timestamp(self.frames_data)

    def _read_from_doc(self) -> typing.Generator["VideoFrame", None, None]:
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
        for each_frame in self.frames_data:
            yield each_frame

    def _read(self) -> typing.Generator["VideoFrame", None, None]:
        """
        选择从文件还是从内存中读取帧
        """
        if self.frames_data:
            yield from self._read_from_mem()
        else:
            yield from self._read_from_doc()

    def get_iterator(self) -> typing.Generator["VideoFrame", None, None]:
        """
        获取帧的迭代器
        """
        return self._read()

    def get_operator(self) -> _BaseFrameOperator:
        """
        根据是否已经加载帧，返回相应的FrameOperator（`MemFrameOperator`或`DocFrameOperator`）
        """
        if self.frames_data:
            return MemFrameOperator(self)
        return DocFrameOperator(self)

    def __iter__(self):
        """
        返回一个用于迭代帧的迭代器
        """
        return self.get_iterator()


if __name__ == '__main__':
    pass
