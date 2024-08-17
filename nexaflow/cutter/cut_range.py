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
from nexaflow import toolbox
from nexaflow import const
from nexaflow.hook import BaseHook
from nexaflow.video import VideoObject, VideoFrame


class VideoCutRange(object):

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
    ) -> dict[str, typing.Any]:
        """
        方法: `contain_image`

        功能:
            在视频帧中检查是否包含指定的图像。可以通过提供图像路径或直接提供图像对象（矩阵）进行检测。

        参数:
            - image_path (str, 可选): 待检测图像的文件路径。如果提供了 `image_path`，将从该路径加载图像。
            - image_object (np.ndarray, 可选): 直接提供的待检测图像数据（NumPy 数组）。如果提供了 `image_object`，将直接使用该图像进行检测。
            - *args: 位置参数，用于传递给 `pick` 方法来选择目标帧的 ID。
            - **kwargs: 关键字参数，用于进一步配置图像检测或传递给 `contain_image` 方法。

        操作流程:
            1. 使用 `self.pick(*args, **kwargs)` 选择目标帧的 ID。该方法根据提供的参数从视频片段中挑选帧，默认选择第一个帧 ID。
            2. 获取视频操作对象 `operator`，该对象提供了从视频中提取帧的功能。
            3. 使用 `operator.get_frame_by_id(target_id)` 获取选定的帧对象。
            4. 调用 `frame.contain_image` 方法，在获取的帧中检查是否包含指定的图像。这个方法可以根据提供的 `image_path` 或 `image_object` 进行检测。
            5. 返回 `frame.contain_image` 的结果，该结果通常是一个字典，包含检测过程中的各种信息。

        返回:
            dict[str, typing.Any]: 返回一个字典，包含检测结果的详细信息。该字典可能包含匹配的置信度、位置等信息，具体取决于 `contain_image` 方法的实现。

        异常处理:
            - 需确保 `self.pick` 方法能够返回有效的帧 ID。
            - 确保 `operator.get_frame_by_id` 返回有效的帧对象。
            - 如果图像加载或检测过程失败，需要处理相应的异常并记录日志。

        使用示例:
            假设提供了 `image_path`，该方法将选择一个目标帧，并检测该帧中是否包含指定的图像。如果包含，则返回相关信息。
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
        方法: `pick`

        功能:
            从视频的帧范围内选择指定数量的帧索引。可以选择随机挑选帧或按均匀间隔挑选帧。

        参数:
            - frame_count (int, 可选): 要选择的帧的数量。如果未指定，则默认选择 3 帧。
            - is_random (bool, 可选): 指定是否随机选择帧。如果为 `True`，则随机选择指定数量的帧；否则按均匀间隔选择帧。
            - *_: 位置参数，未使用，在此方法中被忽略。
            - **__: 关键字参数，未使用，在此方法中被忽略。

        操作流程:
            1. 如果 `frame_count` 未指定，默认为 3。
            2. 记录调试信息，输出帧选择的范围和视频路径。
            3. 初始化一个空的结果列表 `result`。
            4. 如果 `is_random` 为 `True`，则从 `self.start` 到 `self.end` 范围内随机选择 `frame_count` 个帧索引并返回。
            5. 如果 `is_random` 为 `False` 或未指定，按均匀间隔从 `self.start` 到 `self.end` 选择帧：
                - 计算视频片段的长度 `length`。
                - 根据 `frame_count` 均匀地选择指定数量的帧索引并将其添加到 `result` 列表中。
            6. 返回包含所选帧索引的 `result` 列表。

        返回:
            list[int]: 返回一个包含所选帧索引的列表。

        异常处理:
            - 确保 `self.start` 和 `self.end` 之间有足够的帧可供选择，特别是在随机选择时，可能会抛出 `ValueError` 异常。
            - 如果 `frame_count` 大于 `self.start` 和 `self.end` 之间的帧数，需处理并返回适当的错误或调整 `frame_count`。

        使用示例:
            假设 `self.start` 为 0，`self.end` 为 100，`self.video.path` 为 `"video.mp4"`：
            - 若 `is_random` 为 `True` 且 `frame_count` 为 5，可能返回 `[12, 37, 65, 78, 90]`。
            - 若 `is_random` 为 `False` 且 `frame_count` 为 3，则按均匀间隔返回 `[25, 50, 75]`。
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
        方法: `get_frames`

        功能:
            根据提供的帧 ID 列表，从视频中提取对应的帧并返回这些帧的列表。

        参数:
            - frame_id_list (list[int]): 要提取的帧 ID 的列表。每个帧 ID 对应视频中的一个具体帧。
            - *_: 位置参数，未使用，在此方法中被忽略。
            - **__: 关键字参数，未使用，在此方法中被忽略。

        操作流程:
            1. 初始化一个空的输出列表 `out`，用于存储提取的帧。
            2. 获取视频操作对象 `operator`，该对象提供了从视频中提取帧的功能。
            3. 对于 `frame_id_list` 中的每个帧 ID：
                - 调用 `operator.get_frame_by_id(each_id)` 获取对应的帧对象。
                - 将提取的帧对象添加到输出列表 `out` 中。
            4. 返回包含提取的帧对象的列表 `out`。

        返回:
            list["VideoFrame"]: 返回一个包含提取的 `VideoFrame` 对象的列表。

        异常处理:
            - 需确保 `frame_id_list` 中的每个帧 ID 在视频范围内有效。若帧 ID 超出范围，`operator.get_frame_by_id` 可能会抛出异常。
            - 若视频对象未正确加载或操作对象不可用，需处理并返回适当的错误或日志记录。

        使用示例:
            假设 `frame_id_list` 为 `[1, 10, 20]`，返回的 `out` 列表可能包含这些 ID 对应的 `VideoFrame` 对象，例如 `[VideoFrame(1), VideoFrame(10), VideoFrame(20)]`。
        """

        out = list()
        operator = self.video.get_operator()
        for each_id in frame_id_list:
            frame = operator.get_frame_by_id(each_id)
            out.append(frame)
        return out

    def pick_and_get(self, *args, **kwargs) -> list["VideoFrame"]:
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
        """
        方法: `is_loop`

        功能:
            判断视频片段的起始帧和结束帧之间的相似度是否超过指定阈值，以确定该片段是否可能是循环的。

        参数:
            - threshold (float, 可选): 相似度阈值，用于确定起始帧和结束帧是否足够相似。默认使用全局常量 `const.THRES`。
            - **_ : 关键字参数，在此方法中被忽略。

        操作流程:
            1. 如果 `threshold` 未提供，则使用默认阈值 `const.THRES`。
            2. 获取视频操作对象 `operator`，该对象提供了从视频中提取帧的功能。
            3. 使用 `operator.get_frame_by_id(self.start)` 获取视频片段的起始帧。
            4. 使用 `operator.get_frame_by_id(self.end)` 获取视频片段的结束帧。
            5. 调用 `toolbox.compare_ssim` 方法，计算起始帧和结束帧的结构相似度指数（SSIM）。
            6. 如果计算得到的相似度值大于阈值 `threshold`，则返回 `True`，表示该片段可能是循环的；否则返回 `False`。

        返回:
            bool: 返回一个布尔值，`True` 表示起始帧和结束帧的相似度超过阈值，视频片段可能是循环的；`False` 表示不是。

        异常处理:
            - 需确保 `start_frame` 和 `end_frame` 能正确获取。如果帧 ID 无效，`operator.get_frame_by_id` 可能会抛出异常。
            - 如果 `toolbox.compare_ssim` 方法在计算相似度时出错，需处理相关异常并记录日志。

        使用示例:
            假设 `self.start` 为帧 ID 1，`self.end` 为帧 ID 100，如果它们的相似度超过阈值 `threshold`，则返回 `True`，表示这个片段可能是循环的。
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
        方法: `diff`

        功能:
            比较两个视频片段 (`VideoCutRange`) 在指定帧范围内的差异。通过选择的帧对比两个视频片段的相似性。

        参数:
            - another ("VideoCutRange"): 另一个用于比较的视频片段对象。
            - pre_hooks (list["BaseHook"]): 在比较帧数据之前应用的一系列钩子函数列表。钩子函数可以对帧数据进行预处理。
            - *args: 位置参数，用于传递给 `pick_and_get` 方法，以确定从每个视频片段中选择哪些帧。
            - **kwargs: 关键字参数，用于传递给 `pick_and_get` 方法，以进一步配置帧选择过程。

        操作流程:
            1. 使用 `self.pick_and_get(*args, **kwargs)` 从当前视频片段中选择并获取一组帧。
            2. 使用 `another.pick_and_get(*args, **kwargs)` 从另一个视频片段中选择并获取一组帧。
            3. 调用 `toolbox.multi_compare_ssim` 方法，对两个视频片段选定的帧组进行逐一比较。`pre_hooks` 列表中的钩子函数会在比较之前应用到每一帧上。
            4. 返回比较结果，这通常是一个包含每对帧相似度评分的浮点数列表。

        返回:
            list[float]: 返回一个浮点数列表，每个浮点数表示两个视频片段在对应帧上的相似度评分。评分通常是结构相似性（SSIM）值，范围在 0 到 1 之间，越接近 1 表示两帧越相似。

        异常处理:
            - 需要确保 `pick_and_get` 方法返回有效的帧数据。
            - 确保 `toolbox.multi_compare_ssim` 能够正确处理帧数据和钩子函数。
            - 如果在帧选择或比较过程中发生错误，应记录日志并处理异常。

        使用示例:
            该方法可以用于检测两个视频片段在相似场景下的差异，例如在视频编辑、质量控制或内容匹配等应用场景中。

        """

        self_picked = self.pick_and_get(*args, **kwargs)
        another_picked = another.pick_and_get(*args, **kwargs)
        return toolbox.multi_compare_ssim(self_picked, another_picked, pre_hooks)

    def __str__(self):
        return f"<VideoCutRange [{self.start}({self.start_time})-{self.end}({self.end_time})] ssim={self.ssim}>"

    __repr__ = __str__


if __name__ == '__main__':
    pass
