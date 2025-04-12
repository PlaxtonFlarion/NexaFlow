#
#    ____      _     ____
#   / ___|   _| |_  |  _ \ __ _ _ __   __ _  ___
#  | |  | | | | __| | |_) / _` | '_ \ / _` |/ _ \
#  | |__| |_| | |_  |  _ < (_| | | | | (_| |  __/
#   \____\__,_|\__| |_| \_\__,_|_| |_|\__, |\___|
#                                     |___/

import typing
import random
import numpy as np
from loguru import logger
from nexaflow import (
    toolbox, const
)
from nexaflow.video import (
    VideoObject, VideoFrame
)
from nexaflow.hook import BaseHook


class VideoCutRange(object):
    """
    表示视频中的一个连续帧区间段，包含相应的相似度信息和时间戳范围。

    该类主要用于处理视频分析中稳定区段、不稳定区段的提取与判断，
    提供帧采样、合并、图像比较、相似度评估等功能，并支持与另一区间进行差异比对。

    Attributes
    ----------
    video : Union[VideoObject, dict]
        视频对象或视频配置字典。

    start : int
        区间起始帧编号。

    end : int
        区间结束帧编号。

    ssim : list of float
        每帧之间的结构相似度指数。

    mse : list of float
        每帧之间的均方误差。

    psnr : list of float
        每帧之间的峰值信噪比。

    start_time : float
        起始帧的时间戳（单位：秒）。

    end_time : float
        结束帧的时间戳（单位：秒）。
    """

    def __init__(
        self,
        video: typing.Union["VideoObject", dict],
        start: int,
        end: int,
        ssim: list[float],
        mse: list[float],
        psnr: list[float],
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

    def can_merge(self, another: "VideoCutRange", offset: int = None, **_) -> bool:
        """
        判断当前剪辑范围是否可与另一个范围合并。

        Parameters
        ----------
        another : VideoCutRange
            另一个待合并的剪辑范围。

        offset : int, optional
            允许的帧数间隔偏移，默认为 0，表示必须完全连续。

        Returns
        -------
        bool
            如果两个范围可合并，则返回 True；否则返回 False。

        Notes
        -----
        - 合并条件包括：
            1. 两个范围位于同一个视频 (`video.path` 相同)；
            2. 起止帧足够接近，满足连续或带偏移条件。
        - `offset` 允许在帧编号之间有最大 `offset` 个帧的间隔。
        """
        if not offset:
            is_continuous = self.end == another.start
        else:
            is_continuous = self.end + offset >= another.start
        return is_continuous and self.video.path == another.video.path

    def merge(self, another: "VideoCutRange", **kwargs) -> "VideoCutRange":
        """
        合并两个视频剪辑范围，生成新的 `VideoCutRange` 对象。

        Parameters
        ----------
        another : VideoCutRange
            另一个待合并的剪辑范围。

        **kwargs :
            传递给 `can_merge()` 的其他合并容差参数，如 `offset`。

        Returns
        -------
        VideoCutRange
            合并后的新剪辑范围，包含两段的统计信息和时间范围。

        Raises
        ------
        AssertionError
            如果两个范围不可合并，将抛出断言异常。

        Notes
        -----
        - 合并将拼接两个区间的 `ssim`、`mse`、`psnr` 列表；
        - 起始点为当前范围的 `start`，终止点为 `another.end`；
        - 时间范围也将更新为两段的总起止时间。
        """
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
        """
        判断指定帧 ID 是否包含在当前剪辑范围中。

        Parameters
        ----------
        frame_id : int
            要判断的帧编号。

        Returns
        -------
        bool
            若该帧编号在 `[start, end]` 区间内（闭区间），则返回 True；否则返回 False。

        Notes
        -----
        - 当前剪辑范围是闭区间 `[start, end]`；
        - 支持通过 `contain_frame_id` 别名访问该方法。
        """
        return frame_id in range(self.start, self.end + 1)

    contain_frame_id = contain

    def contain_image(
        self, image_path: str = None, image_object: np.ndarray = None, *args, **kwargs
    ) -> dict[str, typing.Any]:
        """
        判断当前剪辑范围中是否存在与指定图像匹配的帧。

        Parameters
        ----------
        image_path : str, optional
            输入图像的路径。

        image_object : np.ndarray, optional
            图像对象，若未提供路径，可直接传入图像矩阵。

        *args :
            传递给 `pick()` 的额外参数，用于控制取样帧的逻辑。

        **kwargs :
            传递给 `contain_image()` 的匹配参数，如匹配方法、精度控制等。

        Returns
        -------
        dict[str, Any]
            匹配结果字典，由 `VideoFrame.contain_image()` 返回，包含匹配位置、相似度评分等信息。

        Notes
        -----
        - 默认从当前范围中采样一帧（由 `pick()` 决定），并对其执行图像匹配；
        - 若需更精细匹配，可通过自定义采样策略或增加帧数；
        - 图像匹配的执行依赖于 `VideoFrame.contain_image()` 方法，支持模板匹配等多种算法。
        """
        target_id = self.pick(*args, **kwargs)[0]
        operator = self.video.get_operator()
        frame = operator.get_frame_by_id(target_id)

        return frame.contain_image(
            image_path=image_path, image_object=image_object, **kwargs
        )

    def pick(
        self, frame_count: int = None, is_random: bool = None, *_, **__
    ) -> list[int]:
        """
        从当前视频帧区间中选择指定数量的帧编号。

        该方法用于在当前视频区间范围内选取若干代表性的帧索引，支持等距采样或随机采样方式。
        默认情况下采用等距采样，确保帧分布均匀；若指定 `is_random=True`，则会在区间范围内随机抽样。

        Parameters
        ----------
        frame_count : int, optional
            需要选取的帧数量，默认为 3。如果帧数超过区间长度，会抛出异常。

        is_random : bool, optional
            是否启用随机采样模式，默认为 False，表示使用等距采样。

        *_ : tuple
            预留参数，用于兼容扩展。

        **__ : dict
            预留参数，用于兼容扩展。

        Returns
        -------
        list[int]
            选中的帧编号列表，按照采样策略生成。

        Raises
        ------
        ValueError
            如果指定帧数超过当前区间长度，在启用随机采样时可能引发采样错误。

        Notes
        -----
        - 等距采样将起始点和结束点之间平均划分为 `frame_count+1` 份，从中间断点处采样。
        - 随机采样不会保证顺序和均匀性，但可能更适用于避免过拟合的模型训练。
        - `frame_count` 实际参与计算时会自动加 1，以增强采样间隔的分布精度。
        - 该方法不会返回起始帧或结束帧编号。
        - 采样结果在日志中会记录选取范围与数量。
        """
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
        self, frame_id_list: list[int], *_, **__
    ) -> list["VideoFrame"]:
        """
        根据帧编号列表获取对应帧图像对象。

        该方法根据输入的帧 ID 列表，从视频文件或缓存中提取对应的帧数据，封装为 `VideoFrame` 实例返回。

        Parameters
        ----------
        frame_id_list : list of int
            指定要提取的帧编号列表，编号从 1 开始。

        *_ : tuple
            预留位置参数，用于扩展兼容。

        **__ : dict
            预留关键字参数，用于扩展兼容。

        Returns
        -------
        list of VideoFrame
            包含 `VideoFrame` 实例的列表，每个实例包含帧编号、时间戳和图像数据。

        Notes
        -----
        - 该方法通过调用 `video.get_operator()` 获取帧读取器，并使用 `get_frame_by_id()` 方法逐帧读取。
        - 如果帧编号越界或帧不存在，返回列表中将缺失对应帧。
        - 建议搭配 `pick()` 或 `pick_and_get()` 方法联合使用，以避免重复实现采样逻辑。
        """
        out = list()
        operator = self.video.get_operator()
        for each_id in frame_id_list:
            frame = operator.get_frame_by_id(each_id)
            out.append(frame)
        return out

    def pick_and_get(self, *args, **kwargs) -> list["VideoFrame"]:
        """
        在当前范围内选取若干帧并返回对应的帧图像对象。

        该方法相当于 `pick()` 与 `get_frames()` 的组合操作：
        首先根据采样策略从当前区间内选取若干帧编号，然后加载对应帧数据，返回 `VideoFrame` 对象列表。

        Parameters
        ----------
        *args : tuple
            可变参数，传递给 `pick()` 和 `get_frames()` 方法，用于指定帧数、是否随机等。
        **kwargs : dict
            关键字参数，支持以下常用选项：
            - frame_count (int): 采样帧数，默认 3。
            - is_random (bool): 是否启用随机采样模式，默认为 False。
            - compress_rate / target_size / to_grey 等参数将传递至图像处理流程。

        Returns
        -------
        list of VideoFrame
            包含 `VideoFrame` 实例的列表，每个对象封装帧编号、时间戳及图像数据。

        Notes
        -----
        - 推荐在进行数据准备（如训练样本采样）时使用此方法，可减少重复逻辑。
        - 若帧数过多或图像较大，建议配合压缩参数避免内存开销。

        Workflow
        --------
        1. 调用 `self.pick()` 获取采样帧编号列表。
        2. 调用 `self.get_frames()` 加载指定帧编号对应的图像帧。
        3. 返回完整的 `VideoFrame` 对象列表。
        """
        picked = self.pick(*args, **kwargs)
        return self.get_frames(picked, *args, **kwargs)

    def get_length(self) -> int:
        """
        获取当前视频片段的帧数长度。

        该方法计算当前片段的帧数跨度，定义为 `end - start + 1`，确保包含起始和结束帧。

        Returns
        -------
        int
            视频片段的帧数，即从 start 到 end（包含）的帧数。

        Notes
        -----
        - 起始帧和结束帧都包含在区间内，因此加 1。
        - 在构建样本、提取特征或进行段分析时经常用于过滤长度过短的片段。
        - 本方法不依赖任何帧图像加载，仅基于数值计算。
        """
        return self.end - self.start + 1

    def is_stable(
        self, threshold: float = None, psnr_threshold: float = None, **_
    ) -> bool:
        """
        判断当前视频片段是否为稳定状态。

        稳定性是通过结构相似度（SSIM）均值与可选的峰值信噪比（PSNR）均值判断的：
        - 当 SSIM 均值大于给定阈值 `threshold` 时，初步判断为稳定；
        - 若设置了 `psnr_threshold`，则需同时满足 PSNR 均值大于该值。

        Parameters
        ----------
        threshold : float, optional
            判断稳定状态所用的 SSIM 阈值，默认为 const.THRES。

        psnr_threshold : float, optional
            可选的 PSNR 阈值，若设置则需 SSIM 与 PSNR 均满足阈值条件才判为稳定。

        **_ : dict
            预留参数，用于向后兼容。

        Returns
        -------
        bool
            若当前片段满足稳定性标准，则返回 True，否则返回 False。

        Notes
        -----
        - 通常用于划分稳定阶段与不稳定阶段，是后续分类、摘要或数据筛选的前提。
        - 若仅关注 SSIM，可忽略 `psnr_threshold` 参数。
        - SSIM 越接近 1 表示相似度越高；PSNR 越高表示信号质量越好。
        """
        threshold = threshold if threshold else const.THRES

        res = np.mean(self.ssim) > threshold
        if res and psnr_threshold:
            res = np.mean(self.psnr) > psnr_threshold

        return res

    def is_loop(self, threshold: float = None, **__) -> bool:
        """
        判断当前视频片段是否存在“循环”（Loop）行为。

        通过对比当前区间的起始帧和结束帧的结构相似度（SSIM），若相似度高于指定阈值，
        则认为该区间可能是循环动画或视频内容未发生明显变化。

        Parameters
        ----------
        threshold : float, optional
            判断是否循环的 SSIM 阈值，默认为 const.THRES。

        **__ : dict
            预留参数，用于向后兼容。

        Returns
        -------
        bool
            若起始帧与结束帧的结构相似度高于阈值，则返回 True，表示存在循环；否则返回 False。

        Notes
        -----
        - 该方法可用于检测广告、背景动画、过渡镜头等重复片段。
        - 判断逻辑依赖于 `toolbox.compare_ssim()` 的实现。
        """
        threshold = threshold if threshold else const.THRES

        operator = self.video.get_operator()
        start_frame = operator.get_frame_by_id(self.start)
        end_frame = operator.get_frame_by_id(self.end)
        return toolbox.compare_ssim(start_frame.data, end_frame.data) > threshold

    def diff(
        self,
        another: "VideoCutRange",
        pre_hooks: list["BaseHook"],
        *args,
        **kwargs,
    ) -> list[float]:
        """
        比较当前视频片段与另一个视频片段的帧内容差异（SSIM 相似度）。

        该方法用于对两个 `VideoCutRange` 区间进行内容相似性评估。内部会从每个区间中提取若干关键帧，
        并应用可选的预处理钩子（如灰度转换、裁剪等），之后利用 SSIM 指标计算每对关键帧之间的相似度。

        Parameters
        ----------
        another : VideoCutRange
            要进行比较的另一个视频区间对象。

        pre_hooks : list of BaseHook
            针对每帧图像的预处理钩子列表，用于标准化、去噪、二值化、对齐等操作。

        *args : tuple
            传递给帧选取函数的其他参数（如帧数、是否随机等）。

        **kwargs : dict
            同样传递给帧选取和预处理流程的其他关键字参数。

        Returns
        -------
        list of float
            每对帧之间计算得到的结构相似度（SSIM）得分序列，取值范围为 [0.0, 1.0]。

        Notes
        -----
        - 建议抽取的帧数一致，否则可能导致对齐误差。
        - 分数越接近 1.0 表示越相似；越接近 0.0 表示差异显著。
        - 可用于视频版本对比、动作检测验证、稳定性分析等场景。

        Workflow
        --------
        1. 从当前区间与目标区间分别抽取等量关键帧。
        2. 应用预处理钩子（如灰度转换、模糊、裁剪等）。
        3. 使用 SSIM 算法逐帧比较两组关键帧。
        4. 返回相似度得分序列。
        """
        self_picked = self.pick_and_get(*args, **kwargs)
        another_picked = another.pick_and_get(*args, **kwargs)

        return toolbox.multi_compare_ssim(self_picked, another_picked, pre_hooks)

    def __str__(self):
        return f"<VideoCutRange [{self.start}({self.start_time})-{self.end}({self.end_time})] ssim={self.ssim}>"

    __repr__ = __str__


if __name__ == '__main__':
    pass
