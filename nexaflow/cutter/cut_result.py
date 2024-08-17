#
#    ____      _     ____                 _ _
#   / ___|   _| |_  |  _ \ ___  ___ _   _| | |_
#  | |  | | | | __| | |_) / _ \/ __| | | | | __|
#  | |__| |_| | |_  |  _ <  __/\__ \ |_| | | |_
#   \____\__,_|\__| |_| \_\___||___/\__,_|_|\__|
#

import os
import cv2
import uuid
import json
import difflib
import typing
import numpy as np
from loguru import logger
from nexaflow import toolbox
from nexaflow.hook import BaseHook
from nexaflow.video import VideoObject, VideoFrame
from nexaflow.cutter.cutter import VideoCutRange


class VideoCutResult(object):

    def __init__(
            self,
            video: "VideoObject",
            range_list: list["VideoCutRange"],
            cut_kwargs: dict = None,
    ):
        self.video = video
        self.range_list = range_list
        self.cut_kwargs = cut_kwargs or {}

    def get_target_range_by_id(self, frame_id: int) -> "VideoCutRange":
        for each in self.range_list:
            if each.contain(frame_id):
                return each
        raise RuntimeError(f"frame {frame_id} not found in video")

    @staticmethod
    def _length_filter(range_list: list["VideoCutRange"], limit: int) -> list["VideoCutRange"]:
        after = list()
        for each in range_list:
            if each.get_length() >= limit:
                after.append(each)
        return after

    def get_unstable_range(
            self, limit: int = None, range_threshold: float = None, **kwargs
    ) -> list["VideoCutRange"]:
        """
        方法: `get_unstable_range`

        功能:
            获取视频中不稳定的时间段。这些不稳定的时间段可能是视频中变化较大、不可预测的部分。

        参数:
            - limit (int, 可选): 限制返回的不稳定时间段的最大数量。如果未指定或为 `None`，则不限制数量。
            - range_threshold (float, 可选): 用于过滤时间段的相似度阈值。低于此阈值的时间段将被视为循环模式并被排除在结果之外。
            - **kwargs: 关键字参数，传递给 `is_stable` 和 `can_merge` 方法，用于调整这些方法的行为。

        操作流程:
            1. 从 `self.range_list` 中筛选出所有不稳定的时间段，并按开始时间升序排列。
            2. 如果不稳定的时间段少于或等于1，直接返回该列表。
            3. 否则，遍历不稳定的时间段，尝试将相邻且可合并的时间段合并为一个更大的时间段。
            4. 如果指定了 `limit` 参数，调用 `_length_filter` 方法对合并后的时间段列表进行数量限制。
            5. 如果指定了 `range_threshold` 参数，筛选掉在阈值内形成循环模式的时间段。
            6. 返回最终的不稳定时间段列表。

        返回:
            list["VideoCutRange"]: 返回一个 `VideoCutRange` 对象列表，每个对象表示一个不稳定的时间段。

        异常处理:
            - 在处理时间段合并时需要确保索引范围正确，以避免超出列表边界。
            - 如果在合并或筛选过程中发生异常，应记录日志并处理。

        使用示例:
            该方法可用于检测视频中的不稳定区域，如用于分析视频质量、检测视频剪辑中的跳跃或异常变化。
        """

        change_range_list = sorted(
            [i for i in self.range_list if not i.is_stable(**kwargs)],
            key=lambda x: x.start,
        )

        if len(change_range_list) <= 1:
            return change_range_list

        i = 0
        merged_change_range_list = list()
        while i < len(change_range_list) - 1:
            cur = change_range_list[i]
            while cur.can_merge(change_range_list[i + 1], **kwargs):
                i += 1
                cur = cur.merge(change_range_list[i], **kwargs)
                if i + 1 >= len(change_range_list):
                    break
            merged_change_range_list.append(cur)
            i += 1
        if change_range_list[-1].start > merged_change_range_list[-1].end:
            merged_change_range_list.append(change_range_list[-1])

        if limit:
            merged_change_range_list = self._length_filter(
                merged_change_range_list, limit
            )

        if range_threshold:
            merged_change_range_list = [
                i for i in merged_change_range_list if not i.is_loop(range_threshold)
            ]
        # logger.debug(
        #     f"unstable range of [{self.video.path}]: {merged_change_range_list}"
        # )
        return merged_change_range_list

    def get_range(
            self, limit: int = None, unstable_limit: int = None, **kwargs
    ) -> tuple[list["VideoCutRange"], list["VideoCutRange"]]:
        """
        获取视频中稳定和不稳定的时间段范围。

        参数:
            - limit (int, 可选): 限制返回的稳定范围的数量。
            - unstable_limit (int, 可选): 限制返回的不稳定范围的数量。
            - **kwargs: 传递给 get_unstable_range 方法的其他参数。

        返回:
            - tuple[list[VideoCutRange], list[VideoCutRange]]: 返回一个元组，其中第一个元素是稳定范围列表，第二个元素是不稳定范围列表。

        具体流程:
            1. 调用 get_unstable_range 方法获取不稳定范围列表。
            2. 初始化视频的起始和结束帧 ID 和时间戳。
            3. 定义默认的质量参数字典（SSIM、MSE、PSNR）。
            4. 如果未检测到不稳定阶段，则返回整个视频作为稳定范围。
            5. 确定第一个稳定范围的结束帧 ID 和最后一个稳定范围的起始帧 ID。
            6. 初始化稳定范围列表。
            7. 如果存在稳定的起始阶段，将其添加到稳定范围列表。
            8. 遍历不稳定范围列表，将每个不稳定范围之间的阶段作为稳定范围添加到列表中。
            9. 如果存在稳定地结束阶段，将其添加到稳定范围列表。
            10. 根据 limit 参数过滤稳定范围列表。
            11. 将稳定范围列表按开始帧 ID 排序并返回稳定和不稳定范围列表。
        """

        unstable_range_list = self.get_unstable_range(unstable_limit, **kwargs)

        video_start_frame_id = 1
        video_start_timestamp = 0.0

        video_end_frame_id = self.range_list[-1].end
        video_end_timestamp = self.range_list[-1].end_time

        _default = {
            "ssim": [1.0],
            "mse": [0.0],
            "psnr": [0.0],
        }

        if len(unstable_range_list) == 0:
            # logger.warning(
            #     "no unstable stage detected, seems nothing happened in your video"
            # )
            logger.warning("你的视频看上去是静止的 ...")
            return (
                # stable
                [
                    VideoCutRange(
                        video=self.video,
                        start=video_start_frame_id,
                        end=video_end_frame_id,
                        start_time=video_start_timestamp,
                        end_time=video_end_timestamp,
                        **_default,
                    )
                ],
                # unstable
                [],
            )

        first_stable_range_end_id = unstable_range_list[0].start - 1
        end_stable_range_start_id = unstable_range_list[-1].end + 1

        # IMPORTANT: len(ssim_list) + 1 == video_end_frame_id
        range_list: typing.List[VideoCutRange] = list()
        # stable start
        if first_stable_range_end_id >= 1:
            # logger.debug(f"stable start")
            logger.debug("稳定阶段开始 ...")
            range_list.append(
                VideoCutRange(
                    video=self.video,
                    start=video_start_frame_id,
                    end=first_stable_range_end_id,
                    start_time=video_start_timestamp,
                    end_time=self.get_target_range_by_id(
                        first_stable_range_end_id
                    ).end_time,
                    **_default,
                )
            )
        # unstable start
        else:
            # logger.debug("unstable start")
            logger.debug("不稳定阶段开始 ...")

        # stable end
        if end_stable_range_start_id <= video_end_frame_id:
            # logger.debug("stable end")
            logger.debug("稳定阶段结束 ...")
            range_list.append(
                VideoCutRange(
                    video=self.video,
                    start=end_stable_range_start_id,
                    end=video_end_frame_id,
                    start_time=self.get_target_range_by_id(
                        end_stable_range_start_id
                    ).end_time,
                    end_time=video_end_timestamp,
                    **_default,
                )
            )
        # unstable end
        else:
            # logger.debug("unstable end")
            logger.debug("不稳定阶段结束 ...")

        for i in range(len(unstable_range_list) - 1):
            range_start_id = unstable_range_list[i].end + 1
            range_end_id = unstable_range_list[i + 1].start - 1

            if range_start_id > range_end_id:
                range_start_id, range_end_id = range_end_id, range_start_id

            range_list.append(
                VideoCutRange(
                    video=self.video,
                    start=range_start_id,
                    end=range_end_id,
                    start_time=self.get_target_range_by_id(range_start_id).start_time,
                    end_time=self.get_target_range_by_id(range_end_id).start_time,
                    **_default,
                )
            )

        if limit:
            range_list = self._length_filter(range_list, limit)
        # logger.debug(f"stable range of [{self.video.path}]: {range_list}")
        stable_range_list = sorted(range_list, key=lambda x: x.start)
        return stable_range_list, unstable_range_list

    def get_stable_range(self, limit: int = None, **kwargs) -> list["VideoCutRange"]:
        return self.get_range(limit, **kwargs)[0]

    def get_range_dynamic(
            self,
            stable_num_limit: list[int],
            threshold: float,
            step: float = 0.005,
            max_retry: int = 10,
            **kwargs,
    ) -> tuple[list["VideoCutRange"], list["VideoCutRange"]]:
        """
        方法: `get_range_dynamic`

        功能:
            动态调整阈值，获取视频的稳定和不稳定时间段，使稳定时间段的数量符合预期范围。

        参数:
            - stable_num_limit (list[int]): 稳定时间段数量的期望范围，格式为 `[min, max]`，例如 `[1, 3]` 表示希望稳定时间段的数量在1到3之间。
            - threshold (float): 初始相似度阈值，用于确定视频的稳定性。取值范围为 0.0 到 1.0 之间。
            - step (float, 可选): 每次调整阈值的步长。默认为 0.005。
            - max_retry (int, 可选): 最大重试次数。如果达到此次数仍未能使稳定时间段数量符合期望范围，则停止调整。默认为 10。
            - **kwargs: 关键字参数，传递给 `get_range` 方法，用于调整 `get_range` 的行为。

        操作流程:
            1. 确保 `max_retry` 不为零，`stable_num_limit` 的长度为 2，并且 `threshold` 在有效范围内。
            2. 调用 `get_range` 方法，根据当前阈值获取稳定和不稳定的时间段列表，并计算当前稳定时间段的数量。
            3. 如果当前稳定时间段的数量在 `stable_num_limit` 范围内，则返回稳定和不稳定时间段列表。
            4. 如果稳定时间段的数量少于期望范围，则增加阈值；如果多于期望范围，则减少阈值。
            5. 递归调用 `get_range_dynamic`，调整后的阈值继续获取稳定和不稳定时间段，直至满足期望范围或达到最大重试次数。

        返回:
            tuple[list["VideoCutRange"], list["VideoCutRange"]]: 返回一个元组，包含两个列表：
                - 稳定的时间段列表。
                - 不稳定的时间段列表。

        异常处理:
            - 如果 `max_retry` 为零，将抛出断言错误，提示动态获取范围失败。
            - 如果 `stable_num_limit` 长度不为2，将抛出断言错误。
            - 如果 `threshold` 不在 0.0 和 1.0 之间，将抛出断言错误。

        使用示例:
            该方法用于动态调整视频相似度阈值，以便在分析视频稳定性时获得所需数量的稳定时间段。
        """

        assert max_retry != 0, f"fail to get range dynamically: {stable_num_limit}"
        assert len(stable_num_limit) == 2, "num_limit should be something like [1, 3]"
        assert 0.0 < threshold < 1.0, "threshold out of range"

        stable, unstable = self.get_range(threshold=threshold, **kwargs)
        cur_num = len(stable)
        logger.debug(f"current stable range is {cur_num}")
        if stable_num_limit[0] <= cur_num <= stable_num_limit[1]:
            logger.debug(f"range num is fine")
            return stable, unstable

        if cur_num < stable_num_limit[0]:
            logger.debug("too fewer stages")
            threshold += step

        elif cur_num > stable_num_limit[1]:
            logger.debug("too many stages")
            threshold -= step

        return self.get_range_dynamic(
            stable_num_limit, threshold=threshold, max_retry=max_retry - 1, **kwargs
        )

    def thumbnail(
            self,
            target_range: "VideoCutRange",
            to_dir: str = None,
            compress_rate: float = None,
            is_vertical: bool = None,
            *_,
            **__,
    ) -> np.ndarray:
        """
        方法: `thumbnail`

        功能:
            根据指定的视频剪辑范围生成视频缩略图。缩略图可以按指定的压缩率进行压缩，并可以选择垂直或水平堆叠帧。
            生成的缩略图可以保存为 PNG 文件或直接返回处理后的图像数组。

        参数:
            - target_range (`VideoCutRange`): 指定的视频剪辑范围。该范围内的帧将用于生成缩略图。
            - to_dir (str, 可选): 保存缩略图的目标目录路径。如果未指定，缩略图将不会保存到文件，而是仅返回图像数组。
            - compress_rate (float, 可选): 图像压缩率。默认值为 0.1，即图像尺寸将缩小至原始尺寸的 10%。
            - is_vertical (bool, 可选): 如果为 `True`，帧将垂直堆叠，否则水平堆叠。默认值为 `None`，实际会选择水平堆叠。
            - *_: 位置参数，未使用。
            - **__: 关键字参数，未使用。

        操作流程:
            1. 设置默认压缩率为 0.1。
            2. 根据 `is_vertical` 参数选择帧堆叠方式（垂直或水平）以及分隔线生成方式。
            3. 打开视频文件，并跳转到指定的起始帧位置。
            4. 逐帧读取视频帧，按指定的压缩率压缩帧，并将帧与分隔线添加到帧列表中。
            5. 当读取到的帧数量达到指定范围的长度时，结束帧读取。
            6. 根据帧堆叠方式将所有帧和分隔线合并成一个图像数组。
            7. 如果指定了保存目录，将合并后的图像保存为 PNG 文件，并返回图像数组。

        返回:
            `np.ndarray`: 返回合并后的缩略图图像数组。如果指定了 `to_dir`，缩略图也会保存为 PNG 文件。

        异常处理:
            - 如果在处理视频帧时发生错误，未进行特殊的异常处理。

        使用示例:
            该方法用于生成视频片段的缩略图，可以选择垂直或水平堆叠帧，并可选择是否保存生成的缩略图文件。
        """

        if not compress_rate:
            compress_rate = 0.1

        if is_vertical:
            stack_func = np.vstack

            def get_split_line(f):
                return np.zeros((5, f.shape[1]))

        else:
            stack_func = np.hstack

            def get_split_line(f):
                return np.zeros((f.shape[0], 5))

        frame_list = list()
        with toolbox.video_capture(self.video.path) as cap:
            toolbox.video_jump(cap, target_range.start)
            ret, frame = cap.read()
            count = 1
            length = target_range.get_length()
            while ret and count <= length:
                frame = toolbox.compress_frame(frame, compress_rate)
                frame_list.append(frame)
                frame_list.append(get_split_line(frame))
                ret, frame = cap.read()
                count += 1
        merged = stack_func(frame_list)

        if to_dir:
            target_path = os.path.join(
                to_dir, f"thumbnail_{target_range.start}-{target_range.end}.png"
            )
            cv2.imwrite(target_path, merged)
            logger.debug(f"save thumbnail to {target_path}")
        return merged

    def pick_and_save(
            self,
            range_list: list["VideoCutRange"],
            frame_count: int,
            to_dir: str = None,
            prune: float = None,
            meaningful_name: bool = None,
            # in kwargs
            # compress_rate: float = None,
            # target_size: typing.Tuple[int, int] = None,
            # to_grey: bool = None,
            *args,
            **kwargs,
    ) -> str:
        """
        从视频切割范围列表中选择帧并保存到指定目录中。

        参数:
            range_list (list[VideoCutRange]): 视频切割范围列表。
            frame_count (int): 每个切割范围中要选择的帧数。
            to_dir (str, 可选): 保存选定帧的目录。如果未指定，则创建一个带有时间戳的新目录。
            prune (float, 可选): 修剪参数，如果提供，将应用于选定的帧。
            meaningful_name (bool, 可选): 如果为 True，则使用有意义的文件名，否则使用随机 UUID 文件名。
            *args: 传递给 pick 方法的其他位置参数。
            **kwargs: 传递给 pick 方法和 compress_frame 函数的其他关键字参数。

        返回:
            str: 包含选定帧的目录路径。

        具体流程:
            1. 初始化 `stage_list` 列表，用于存储每个切割范围中的选定帧。
            2. 遍历 `range_list` 列表，对每个切割范围对象调用 `pick` 方法选择帧。
            3. 获取每个选定帧的数据，并将其存储在 `stage_list` 列表中。
            4. 如果提供了 `prune` 参数，则对 `stage_list` 列表进行修剪。
            5. 如果未指定 `to_dir` 参数，则创建一个带有时间戳的新目录。
            6. 创建指定目录及其子目录。
            7. 遍历 `stage_list` 列表，将每个选定帧保存到对应的目录中。
            8. 返回保存选定帧的目录路径。
        """

        stage_list = list()
        for index, each_range in enumerate(range_list):
            picked = each_range.pick(frame_count, *args, **kwargs)
            picked_frames = each_range.get_frames(picked)
            logger.debug(f"pick {picked} in range {each_range}")
            stage_list.append((str(index), picked_frames))

        if prune:
            stage_list = self._prune(prune, stage_list)

        if not to_dir:
            to_dir = toolbox.get_timestamp_str()
        # logger.debug(f"try to make dirs: {to_dir}")
        os.makedirs(to_dir, exist_ok=True)

        for each_stage_id, each_frame_list in stage_list:
            each_stage_dir = os.path.join(to_dir, str(each_stage_id))

            if os.path.isdir(each_stage_dir):
                logger.warning(f"sub dir [{each_stage_dir}] already existed")
                logger.warning(
                    "NOTICE: make sure your data will not be polluted by accident"
                )
            os.makedirs(each_stage_dir, exist_ok=True)

            for each_frame_object in each_frame_list:
                if meaningful_name:
                    image_name = (
                        f"{os.path.basename(os.path.splitext(self.video.path)[0])}"
                        f"_"
                        f"{each_frame_object.frame_id}"
                        f"_"
                        f"{each_frame_object.timestamp}"
                        f".png"
                    )
                else:
                    image_name = f"{uuid.uuid4()}.png"

                each_frame_path = os.path.join(each_stage_dir, image_name)
                compressed = toolbox.compress_frame(each_frame_object.data, **kwargs)
                cv2.imwrite(each_frame_path, compressed)
                logger.debug(
                    f"frame [{each_frame_object.frame_id}] saved to {each_frame_path}"
                )

        return to_dir

    @staticmethod
    def _prune(
            threshold: float,
            stages: list[tuple[str, list["VideoFrame"]]],
    ) -> list[tuple[str, list["VideoFrame"]]]:
        """
        方法: `_prune`

        功能:
            对一组视频帧阶段进行修剪，以减少冗余的帧序列。该方法根据结构相似性（SSIM）指数判断两个阶段是否相似，如果相似度高于指定的阈值（`threshold`），则该阶段将被修剪（即从列表中移除）。

        参数:
            - threshold (`float`): 相似性阈值。如果两个阶段之间的最小 SSIM 值大于该阈值，则认为这两个阶段相似，并修剪后续的阶段。
            - stages (`list[tuple[str, list["VideoFrame"]]]`): 包含多个阶段的列表，每个阶段由一个字符串索引和该阶段的视频帧列表组成。

        操作流程:
            1. 记录修剪前的阶段数量和相似性阈值。
            2. 初始化一个空列表 `after` 用于存储修剪后的阶段。
            3. 遍历每个阶段，逐个比较它与其后续阶段之间的相似性。
            4. 使用 `toolbox.multi_compare_ssim` 方法计算两个阶段之间的 SSIM 列表，并取其最小值。
            5. 如果两个阶段的最小 SSIM 值超过阈值，则认为这两个阶段相似，停止进一步比较，并修剪当前阶段。
            6. 如果该阶段没有被修剪，则将其添加到 `after` 列表中。
            7. 返回修剪后的阶段列表。

        返回:
            `list[tuple[str, list["VideoFrame"]]]`: 返回修剪后的阶段列表。列表中的每个阶段由一个字符串索引和该阶段的视频帧列表组成。

        日志:
            - 记录修剪操作的开始、每次阶段比较的 SSIM 结果以及哪些阶段被修剪。

        异常处理:
            - 没有显式的异常处理。如果在 SSIM 计算或列表操作时发生错误，将抛出相应的异常。

        使用示例:
            该方法通常用于视频分析中，以减少冗余的帧序列，从而优化视频处理效率。
        """

        logger.debug(
            f"start pruning ranges, origin length is {len(stages)}, threshold is {threshold}"
        )
        after = list()
        for i in range(len(stages)):
            index, frames = stages[i]
            for j in range(i + 1, len(stages)):
                next_index, next_frames = stages[j]
                ssim_list = toolbox.multi_compare_ssim(frames, next_frames)
                min_ssim = min(ssim_list)
                logger.debug(f"compare {index} with {next_index}: {ssim_list}")
                if min_ssim > threshold:
                    logger.debug(f"stage {index} has been pruned")
                    break
            else:
                after.append(stages[i])
        return after

    def dumps(self) -> str:

        def _handler(obj: object):
            if isinstance(obj, np.ndarray):
                return "<np.ndarray object>"
            return obj.__dict__

        return json.dumps(self, sort_keys=True, default=_handler)

    def dump(self, json_path: str, **kwargs):
        logger.debug(f"dump result to {json_path}")
        assert not os.path.exists(json_path), f"{json_path} already existed"
        with open(json_path, "w+", **kwargs) as f:
            f.write(self.dumps())

    @classmethod
    def loads(cls, content: str) -> "VideoCutResult":
        json_dict: dict = json.loads(content)
        return cls(
            VideoObject(**json_dict["video"]),
            [VideoCutRange(**each) for each in json_dict["range_list"]],
        )

    @classmethod
    def load(cls, json_path: str, **kwargs) -> "VideoCutResult":
        logger.debug(f"load result from {json_path}")
        with open(json_path, **kwargs) as f:
            return cls.loads(f.read())

    def diff(
            self,
            another: "VideoCutResult",
            auto_merge: bool = None,
            pre_hooks: list["BaseHook"] = None,
            output_path: str = None,
            *args,
            **kwargs,
    ) -> "VideoCutResultDiff":
        """
        方法: `diff`

        功能:
            比较当前 `VideoCutResult` 对象与另一个 `VideoCutResult` 对象之间的稳定阶段差异，并生成一个 `VideoCutResultDiff` 对象。该方法可以自动合并结果，并将帧保存到指定的输出目录中。

        参数:
            - another (`VideoCutResult`): 另一个要进行比较的 `VideoCutResult` 对象。
            - auto_merge (`bool`, 可选): 是否自动合并比较结果。默认为 `None`，表示不进行自动合并。如果设置为 `True`，则会自动选择差异最大的阶段进行合并。
            - pre_hooks (`list[BaseHook]`, 可选): 预处理钩子列表，用于在比较之前对数据进行处理。默认为 `None`。
            - output_path (`str`, 可选): 保存比较结果的目录路径。如果提供，比较过程中选定的帧将会保存到该目录中。
            - *args: 传递给 `get_range` 方法的其他参数，用于获取稳定阶段。
            - **kwargs: 传递给 `get_range` 方法的关键字参数，用于获取稳定阶段。

        操作流程:
            1. 调用 `get_range` 方法分别获取当前对象和另一个对象的稳定阶段。
            2. 调用 `pick_and_save` 方法，从获取的稳定阶段中选取帧，并将其保存到指定的 `output_path` 目录中（如果提供）。
            3. 创建 `VideoCutResultDiff` 对象，传入两个稳定阶段列表，初始化差异结果。
            4. 调用 `apply_diff` 方法，应用预处理钩子 `pre_hooks` 对稳定阶段进行差异比较。
            5. 如果 `auto_merge` 为 `True`，则对结果进行自动合并：
                - 遍历所有比较结果，选择每个阶段差异最大的项，更新差异结果数据。
            6. 返回 `VideoCutResultDiff` 对象，包含比较后的差异结果。

        返回:
            `VideoCutResultDiff`: 返回包含稳定阶段差异的 `VideoCutResultDiff` 对象。

        日志:
            - 日志未明确记录在此方法中，但可能由其他调用的方法记录（如 `get_range`, `pick_and_save`, `apply_diff`）。

        异常处理:
            - 没有显式的异常处理。如果在比较或文件操作时发生错误，将抛出相应的异常。

        使用示例:
            该方法用于比较两个视频剪辑的稳定阶段，尤其是在视频处理或分析工作流中，以检测视频之间的差异。
        """

        self_stable, _ = self.get_range(*args, **kwargs)
        another_stable, _ = another.get_range(*args, **kwargs)
        self.pick_and_save(self_stable, 3, to_dir=output_path)
        another.pick_and_save(another_stable, 3, to_dir=output_path)

        result = VideoCutResultDiff(self_stable, another_stable)
        result.apply_diff(pre_hooks)

        if auto_merge:
            after = dict()
            for self_stage_name, each_result in result.data.items():
                max_one = sorted(each_result.items(), key=lambda x: max(x[1]))[-1]
                max_one = (max_one[0], max(max_one[1]))
                after[self_stage_name] = max_one
            result.data = after
        return result

    @staticmethod
    def range_diff(
            range_list_1: list["VideoCutRange"],
            range_list_2: list["VideoCutRange"],
            *args,
            **kwargs,
    ) -> dict[int, dict[int, list[float]]]:
        """
        方法: `range_diff`

        功能:
            比较两个 `VideoCutRange` 列表中的每一对范围，并生成一个嵌套字典，表示两个范围列表之间的差异。该方法通过计算每个范围对之间的差异来提供详细的比较数据。

        参数:
            - range_list_1 (`list[VideoCutRange]`): 第一个视频剪辑范围列表，用于与第二个范围列表进行比较。
            - range_list_2 (`list[VideoCutRange]`): 第二个视频剪辑范围列表，用于与第一个范围列表进行比较。
            - *args: 传递给 `VideoCutRange.diff` 方法的其他参数，用于计算范围之间的差异。
            - **kwargs: 传递给 `VideoCutRange.diff` 方法的关键字参数，用于计算范围之间的差异。

        操作流程:
            1. 获取 `range_list_1` 和 `range_list_2` 的长度，分别表示两个视频剪辑范围列表中稳定阶段的数量。
            2. 检查两个范围列表的长度是否相等。如果不相等，记录一个警告日志，提醒用户两个列表的阶段数量不一致。
            3. 初始化一个空的嵌套字典 `data`，用于存储每对范围之间的差异。
            4. 遍历 `range_list_1` 中的每个范围 `each_self_range`，记录其索引为 `self_id`。
            5. 对于 `range_list_1` 中的每个范围，进一步遍历 `range_list_2` 中的每个范围 `another_self_range`，记录其索引为 `another_id`。
            6. 调用 `each_self_range.diff(another_self_range, *args, **kwargs)` 方法，计算 `each_self_range` 和 `another_self_range` 之间的差异，并将结果存储在 `temp` 字典中，其中 `another_id` 作为键。
            7. 将 `temp` 字典作为值，`self_id` 作为键，存储在 `data` 字典中。
            8. 返回包含所有范围对差异的嵌套字典 `data`。

        返回:
            `dict[int, dict[int, list[float]]]`: 返回一个嵌套字典，其中外层字典的键是 `range_list_1` 中范围的索引，值是另一个字典。内层字典的键是 `range_list_2` 中范围的索引，值是一个浮点数列表，表示这对范围之间的差异。

        日志:
            - 如果两个范围列表的长度不相等，记录一个警告日志，指示阶段数量不一致。
            - 差异计算的详细日志可能由 `VideoCutRange.diff` 方法记录。

        异常处理:
            - 没有显式的异常处理。如果在比较过程中发生错误，将抛出相应的异常。

        使用示例:
            该方法用于比较两个视频剪辑范围列表，特别是在视频分析过程中，以检测不同范围之间的差异。
        """

        self_stable_range_count = len(range_list_1)
        another_stable_range_count = len(range_list_2)
        if self_stable_range_count != another_stable_range_count:
            logger.warning(
                f"stage counts not equal: {self_stable_range_count} & {another_stable_range_count}"
            )

        data = dict()
        for self_id, each_self_range in enumerate(range_list_1):
            temp = dict()
            for another_id, another_self_range in enumerate(range_list_2):
                temp[another_id] = each_self_range.diff(
                    another_self_range, *args, **kwargs
                )
            data[self_id] = temp
        return data


class VideoCutResultDiff(object):
    """
    类: `VideoCutResultDiff`

    功能:
        该类用于比较两个 `VideoCutRange` 列表的差异，并提供分析方法来判断阶段是否丢失、阶段的转移情况以及阶段的详细差异。

    属性:
        - threshold (`float`): 定义判断阶段是否丢失的阈值，默认为 0.7。
        - default_stage_id (`int`): 默认的阶段 ID，用于在找不到匹配阶段时返回，默认为 -1。
        - default_score (`float`): 默认的评分，用于在找不到匹配阶段时返回，默认为 -1.0。
        - origin (`list[VideoCutRange]`): 原始的 `VideoCutRange` 列表。
        - another (`list[VideoCutRange]`): 用于与原始列表进行比较的 `VideoCutRange` 列表。
        - data (`typing.Optional[dict[int, dict[int, list[float]]]]`): 存储 `range_diff` 方法返回的差异数据字典，默认为 None。
    """

    threshold: float = 0.7
    default_stage_id: int = -1
    default_score: float = -1.0

    def __init__(self, origin: list["VideoCutRange"], another: list["VideoCutRange"]):
        """
        构造方法，初始化 `VideoCutResultDiff` 实例。

        参数:
            - origin (`list[VideoCutRange]`): 原始的 `VideoCutRange` 列表。
            - another (`list[VideoCutRange]`): 用于比较的 `VideoCutRange` 列表。
        """
        self.origin = origin
        self.another = another
        self.data: typing.Optional[dict[int, dict[int, list[float]]]] = None

    def apply_diff(self, pre_hooks: list["BaseHook"] = None):
        """
        应用阶段差异分析方法，并存储差异数据。

        参数:
            - pre_hooks (`list[BaseHook]`, 可选): 预处理钩子列表，用于在比较前应用，默认为 None。

        说明:
            该方法调用 `VideoCutResult.range_diff` 方法来计算两个范围列表之间的差异，并将结果存储在 `self.data` 中。
        """
        self.data = VideoCutResult.range_diff(self.origin, self.another, pre_hooks)

    def most_common(self, stage_id: int) -> (int, float):
        """
        获取与指定阶段最匹配的阶段 ID 及其评分。

        参数:
            - stage_id (`int`): 指定的阶段 ID。

        返回:
            - `int`: 与指定阶段最匹配的阶段 ID。
            - `float`: 匹配的评分。

        异常:
            - `AssertionError`: 如果指定的 `stage_id` 不在 `self.data` 中，会引发异常。

        说明:
            该方法遍历指定阶段的差异数据，找到评分最高的匹配阶段。
        """
        assert stage_id in self.data
        ret_k, ret_v = self.default_stage_id, self.default_score
        for k, v in self.data[stage_id].items():
            cur = max(v)
            if cur > ret_v:
                ret_k = k
                ret_v = cur
        return ret_k, ret_v

    def is_stage_lost(self, stage_id: int) -> bool:
        """
        判断指定阶段是否丢失。

        参数:
            - stage_id (`int`): 指定的阶段 ID。

        返回:
            - `bool`: 如果阶段的最高评分低于阈值，则返回 `True`，否则返回 `False`。

        说明:
            该方法调用 `most_common` 方法，判断指定阶段的评分是否低于阈值 `self.threshold`。
        """
        _, v = self.most_common(stage_id)
        return v < self.threshold

    def any_stage_lost(self) -> bool:
        """
        判断是否所有阶段都丢失。

        返回:
            - `bool`: 如果所有阶段的最高评分都低于阈值，则返回 `True`，否则返回 `False`。

        说明:
            该方法遍历 `self.data` 中的所有阶段，检查它们是否都丢失。
        """
        return all((self.is_stage_lost(each) for each in self.data.keys()))

    def stage_shift(self) -> list[int]:
        """
        获取阶段转移情况的列表。

        返回:
            - `list[int]`: 返回所有得分高于阈值的阶段 ID 列表。

        说明:
            该方法遍历 `self.data` 中的阶段，找到评分高于阈值的匹配阶段并返回其 ID。
        """
        ret = list()
        for k in self.data.keys():
            new_k, score = self.most_common(k)
            if score > self.threshold:
                ret.append(new_k)
        return ret

    def stage_diff(self) -> typing.Iterator:
        """
        生成阶段差异的比较结果。

        返回:
            - `typing.Iterator`: 返回阶段差异的迭代器对象。

        说明:
            该方法使用 `difflib.Differ` 对象来比较阶段转移情况和目标阶段列表，并生成差异报告。
        """
        return difflib.Differ().compare(
            [str(each) for each in self.stage_shift()],
            [str(each) for each in range(len(self.another))],
        )


if __name__ == '__main__':
    pass
