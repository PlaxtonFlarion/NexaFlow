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

    def __init__(self, frame_id: int, timestamp: float, data: "np.ndarray"):
        self.frame_id: int = frame_id
        self.timestamp: float = timestamp
        self.data: "np.ndarray" = data

    def __str__(self):
        return f"<VideoFrame id={self.frame_id} timestamp={self.timestamp}>"

    @staticmethod
    def initial(
            cap: "cv2.VideoCapture", frame: "np.ndarray",
            scale: int | float = None, shape: tuple = None, color: bool = None,
    ) -> "VideoFrame":

        frame_id = toolbox.get_current_frame_id(cap)
        timestamp = toolbox.get_current_frame_time(cap)
        new_frame = toolbox.compress_frame(frame, scale, shape, color)

        return VideoFrame(frame_id, timestamp, new_frame)

    def copy(self) -> "VideoFrame":
        return VideoFrame(self.frame_id, self.timestamp, self.data[:])

    def contain_image(
        self, *, image_path: str = None, image_object: np.ndarray = None, **kwargs
    ) -> dict[str, typing.Any]:
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
    """
    VideoObject(path, fps=None)

    视频对象类，用于加载、管理与迭代视频帧数据。

    该类封装了视频路径、帧总数、帧尺寸、帧数据的加载与访问等操作。
    支持在内存中缓存帧，也可从视频文件实时读取，并提供用于帧处理的操作器接口。
    """

    def __init__(
        self,
        path: typing.Union[str, "os.PathLike"],
        fps: typing.Optional[int] = None,
    ):
        """
        初始化视频对象并读取基本信息。

        Parameters
        ----------
        path : str or os.PathLike
            视频文件的路径。

        fps : int, optional
            若指定该值，则会转换视频帧率后读取。

        Notes
        -----
        - 检查路径是否为有效文件；
        - 若指定 `fps`，则使用 `toolbox.fps_convert` 转换为目标帧率；
        - 使用 `toolbox.video_capture` 打开视频，初始化帧总数和帧尺寸；
        - 帧数据初始化为空。
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
        """
        使用 moviepy 的迭代器为帧对象补充时间戳信息。

        Parameters
        ----------
        frame_data : tuple of VideoFrame
            已加载的视频帧对象列表。

        Notes
        -----
        - 通过 `iter_frames(with_times=True)` 获取每帧的时间戳；
        - 若帧对象中时间戳为空则补充；
        - 使用 `toolbox.show_progress` 展示处理进度。
        """
        assert frame_data, "load_frames() first"

        vid = mpy.VideoFileClip(self.path)
        vid_count = vid.reader.nframes

        progress_bar = toolbox.show_progress(total=vid_count, color=153)
        for frame_id, (timestamp, _) in enumerate(vid.iter_frames(with_times=True)):
            if frame_id >= len(frame_data):
                progress_bar.close()
                break

            # Note frame_id_real = frame_id + 1
            if not frame_data[frame_id].timestamp:
                # logger.debug(f"fix frame {frame_id_real}'s timestamp: {timestamp}")
                frame_data[frame_id].timestamp = timestamp

            progress_bar.update(1)

    def clean_frames(self) -> None:
        """
        清空帧数据缓存。

        Notes
        -----
        - 将 `self.frames_data` 设为空元组，释放内存。
        """
        self.frames_data = tuple()

    @staticmethod
    def frame_details(frame_data: tuple["VideoFrame"]) -> str:
        """
        获取帧数据的详细信息字符串。

        Parameters
        ----------
        frame_data : tuple of VideoFrame
            包含若干帧的元组。

        Returns
        -------
        str
            帧类型、单帧内存占用、总内存占用和帧尺寸信息。

        Notes
        -----
        - 返回的内容适用于打印输出或日志记录。
        """
        frame = frame_data[0]

        every_cost = frame.data.nbytes / (1024 ** 2)
        total_cost = every_cost * len(frame_data)
        frame_size = frame.data.shape[::-1]
        frame_name = frame.__class__.__name__
        frame_info = f"[{every_cost:.2f} MB] [{total_cost:.2f} MB]"

        return f"{frame_name} {frame_info} {frame_size}"

    def load_frames(
            self, scale: int | float = None, shape: tuple = None, color: bool = False
    ) -> None:
        """
        加载视频的所有帧并存入内存。

        该方法会逐帧读取视频内容，并根据传入参数调整图像尺寸或颜色模式，最终将所有帧封装为 `VideoFrame` 对象并存入 `self.frames_data` 中。

        Parameters
        ----------
        scale : int or float, optional
            图像缩放比例。例如 0.5 表示将原始尺寸缩小一半。与 `shape` 参数互斥，优先级低于 `shape`。

        shape : tuple, optional
            指定帧的目标尺寸，格式为 (width, height)。若提供此参数，则忽略 `scale`。

        color : bool, optional
            是否保留彩色图像。若为 False，则图像将被转换为灰度。

        Notes
        -----
        - 使用 `cv2.VideoCapture` 读取原始视频帧；
        - 每帧通过 `VideoFrame.initial()` 方法包装成对象；
        - 可指定图像压缩方式（缩放比例或尺寸）与颜色空间；
        - 加载过程显示实时进度条；
        - 最终帧数据被保存为元组，写入 `self.frames_data`；
        - 加载完成后通过 `sync_timestamp()` 方法补全每帧的时间戳信息；
        - 若图像总量较大，可能会消耗较多内存，请酌情使用。

        Raises
        ------
        AssertionError
            如果视频路径无效或帧读取失败，会在 `VideoFrame.initial()` 内部抛出断言异常。

        Workflow
        --------
        1. 初始化进度条以显示帧加载进度；
        2. 打开视频文件，逐帧读取图像；
        3. 每一帧调用 `VideoFrame.initial()` 进行封装和预处理（可缩放、裁剪、转灰）；
        4. 将封装后的帧添加到帧数据列表中；
        5. 所有帧加载完成后，关闭进度条；
        6. 调用 `sync_timestamp()` 方法，根据真实视频时间同步各帧的时间戳；
        7. 最终结果保存在 `self.frames_data` 属性中。
        """
        frame_data_list: list["VideoFrame"] = []

        progress_bar = toolbox.show_progress(total=self.frame_count, color=180)
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
        从磁盘中逐帧读取视频帧（惰性生成器）。

        Yields
        ------
        VideoFrame
            视频帧对象。
        """
        with toolbox.video_capture(self.path) as cap:
            success, frame = cap.read()
            while success:
                yield VideoFrame.initial(cap, frame)
                success, frame = cap.read()

    def _read_from_mem(self) -> typing.Generator["VideoFrame", None, None]:
        """
        从内存中读取帧（惰性生成器）。

        Yields
        ------
        VideoFrame
            视频帧对象。
        """
        for each_frame in self.frames_data:
            yield each_frame

    def _read(self) -> typing.Generator["VideoFrame", None, None]:
        """
        根据当前状态选择读取方式（内存或磁盘）。

        Yields
        ------
        VideoFrame
            视频帧对象。

        Notes
        -----
        - 若已加载帧数据，则从内存中读取；
        - 否则从文件中逐帧读取。
        """
        if self.frames_data:
            yield from self._read_from_mem()
        else:
            yield from self._read_from_doc()

    def get_iterator(self) -> typing.Generator["VideoFrame", None, None]:
        """
        获取帧的迭代器（惰性生成器）。

        Returns
        -------
        generator
            VideoFrame 迭代器。
        """
        return self._read()

    def get_operator(self) -> "_BaseFrameOperator":
        """
        返回视频帧操作器对象。

        Returns
        -------
        _BaseFrameOperator
            若已加载帧，则返回 `MemFrameOperator`；
            否则返回 `DocFrameOperator`。

        Notes
        -----
        - 用于进行帧处理操作的策略选择。
        """
        return MemFrameOperator(self) if self.frames_data else DocFrameOperator(self)

    def __iter__(self):
        """
        实现类的迭代协议。

        Returns
        -------
        generator
            可用于遍历视频帧的迭代器。
        """
        return self.get_iterator()


if __name__ == '__main__':
    pass
