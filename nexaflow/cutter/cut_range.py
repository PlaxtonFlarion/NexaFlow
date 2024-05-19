import typing
import random
import numpy as np
from loguru import logger
from nexaflow import toolbox
from nexaflow import const
from nexaflow.hook import BaseHook
from nexaflow.video import VideoObject, VideoFrame


class VideoCutRange(object):

    def __init__(
        self,
        video: typing.Union[VideoObject, typing.Dict],
        start: int,
        end: int,
        ssim: typing.List[float],
        mse: typing.List[float],
        psnr: typing.List[float],
        start_time: float,
        end_time: float,
    ):
        if isinstance(video, dict):
            self.video = VideoObject(**video)
        else:
            self.video = video

        self.start = start
        self.end = end
        self.ssim = ssim
        self.mse = mse
        self.psnr = psnr
        self.start_time = start_time
        self.end_time = end_time

        if start > end:
            self.start, self.end = self.end, self.start
            self.start_time, self.end_time = self.end_time, self.start_time

        # logger.debug(
        #     f"new a range: {self.start}({self.start_time}) - {self.end}({self.end_time})"
        # )

    def can_merge(self, another: "VideoCutRange", offset: int = None, **_):
        """
        函数 `can_merge` 用于判断两个视频剪辑区域（`VideoCutRange` 对象）是否可以合并。
        这主要依赖于剪辑区间的结束点和开始点是否接近或连续，以及两个剪辑区间是否来自同一视频文件。
        @param another: 另一个 `VideoCutRange` 对象，表示要比较的第二个视频剪辑区间。
        @param offset: 一个可选的整数，用于定义接受两个视频剪辑区间之间间隔的容忍度。
        @param _:
        @return: 函数返回一个布尔值，如果两个剪辑区间既连续（考虑到可能的 `offset`）又来自同一视频文件，则返回 `True`，表示它们可以合并；否则返回 `False`。
        """
        if not offset:
            is_continuous = self.end == another.start
        else:
            is_continuous = self.end + offset >= another.start
        return is_continuous and self.video.path == another.video.path

    def merge(self, another: "VideoCutRange", **kwargs) -> "VideoCutRange":
        assert self.can_merge(another, **kwargs)
        return __class__(
            self.video,
            self.start,
            another.end,
            self.ssim + another.ssim,
            self.mse + another.mse,
            self.psnr + another.psnr,
            self.start_time,
            another.end_time,
        )

    def contain(self, frame_id: int) -> bool:
        return frame_id in range(self.start, self.end + 1)

    contain_frame_id = contain

    def contain_image(
        self, image_path: str = None, image_object: np.ndarray = None, *args, **kwargs
    ) -> typing.Dict[str, typing.Any]:
        target_id = self.pick(*args, **kwargs)[0]
        operator = self.video.get_operator()
        frame = operator.get_frame_by_id(target_id)
        return frame.contain_image(
            image_path=image_path, image_object=image_object, **kwargs
        )

    def pick(
        self, frame_count: int = None, is_random: bool = None, *_, **__
    ) -> typing.List[int]:
        if not frame_count:
            frame_count = 3
        logger.debug(
            f"pick {frame_count} frames "
            f"from {self.start}({self.start_time}) "
            f"to {self.end}({self.end_time}) "
            f"on video {self.video.path}"
        )

        result = list()
        if is_random:
            return random.sample(range(self.start, self.end), frame_count)
        length = self.get_length()

        frame_count += 1
        for _ in range(1, frame_count):
            cur = int(self.start + length / frame_count * _)
            result.append(cur)
        return result

    def get_frames(
        self, frame_id_list: typing.List[int], *_, **__
    ) -> typing.List[VideoFrame]:

        out = list()
        operator = self.video.get_operator()
        for each_id in frame_id_list:
            frame = operator.get_frame_by_id(each_id)
            out.append(frame)
        return out

    def pick_and_get(self, *args, **kwargs) -> typing.List[VideoFrame]:
        picked = self.pick(*args, **kwargs)
        return self.get_frames(picked, *args, **kwargs)

    def get_length(self):
        return self.end - self.start + 1

    def is_stable(
        self, threshold: float = None, psnr_threshold: float = None, **_
    ) -> bool:
        """
        函数 `is_stable` 用于确定一个数据集（可能是图像帧或视频片段）是否稳定。
        这个函数主要基于结构相似性指数（SSIM）和峰值信噪比（PSNR）来评估稳定性。
        这两个指标常用于图像和视频质量评估，其中 SSIM 衡量两张图像的相似性，而 PSNR 是衡量图像重建质量的指标。
        @param threshold: 一个可选的浮点数，设置判断图像稳定性的 SSIM 阈值。如果未指定，函数将使用一个常量值 `const.THRES`。
        @param psnr_threshold: 一个可选的浮点数，设置判断图像稳定性的 PSNR 阈值。
        @param _:
        @return: 函数返回一个布尔值，表示数据集是否稳定。
        """

        threshold = threshold if threshold else const.THRES

        res = np.mean(self.ssim) > threshold
        if res and psnr_threshold:
            res = np.mean(self.psnr) > psnr_threshold

        return res

    def is_loop(self, threshold: float = None, **_) -> bool:
        threshold = threshold if threshold else const.THRES

        operator = self.video.get_operator()
        start_frame = operator.get_frame_by_id(self.start)
        end_frame = operator.get_frame_by_id(self.end)
        return toolbox.compare_ssim(start_frame.data, end_frame.data) > threshold

    def diff(
        self,
        another: "VideoCutRange",
        pre_hooks: typing.List[BaseHook],
        *args,
        **kwargs,
    ) -> typing.List[float]:
        self_picked = self.pick_and_get(*args, **kwargs)
        another_picked = another.pick_and_get(*args, **kwargs)
        return toolbox.multi_compare_ssim(self_picked, another_picked, pre_hooks)

    def __str__(self):
        return f"<VideoCutRange [{self.start}({self.start_time})-{self.end}({self.end_time})] ssim={self.ssim}>"

    __repr__ = __str__


if __name__ == '__main__':
    pass
