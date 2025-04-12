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
import typing
import difflib
import numpy as np
from loguru import logger
from nexaflow import toolbox
from nexaflow.hook import BaseHook
from nexaflow.video import (
    VideoObject, VideoFrame
)
from nexaflow.cutter.cutter import VideoCutRange


class VideoCutResult(object):
    """
    VideoCutResult 类用于组织和管理视频中分割出来的帧区间（VideoCutRange），支持区间提取、稳定/不稳定区域判定、
    帧图像提取与保存、区间动态调整、结果序列化/反序列化、差异比较等操作。

    该类作为分析视频结构和生成训练数据的核心组件之一，围绕一段视频的多个切分段提供灵活、高度可配置的处理方法。

    Attributes
    ----------
    video : VideoObject
        视频对象，包含原始视频路径、帧信息等。

    range_list : list of VideoCutRange
        视频中已分割的区间列表，通常由某种图像相似度算法生成。

    cut_kwargs : dict, optional
        区间处理参数字典，用于支持各种自定义行为或阈值策略。

    Notes
    -----
    本类提供以下核心功能：
    - 基于稳定性、相似性自动筛选并合并区间；
    - 提供动态阈值调整机制，确保最终稳定区间数量落入指定范围；
    - 支持对比两个视频切分结果，生成结构级别的差异报告；
    - 内建帧图像采样与保存方法，可直接导出训练用图像；
    - 支持 JSON 格式序列化/反序列化与持久化管理；
    - 可生成缩略图用于快速浏览视频结构。
    """

    def __init__(
            self,
            video: "VideoObject",
            range_list: list["VideoCutRange"],
            cut_kwargs: dict = None,
    ):
        """
        初始化 VideoCutResult 实例。

        构建用于管理视频切割结果的数据结构，保存原始视频对象、切割区间和可选参数配置。

        Parameters
        ----------
        video : VideoObject
            视频对象，封装了视频路径、帧总数、尺寸等基本信息。

        range_list : list of VideoCutRange
            视频被切割后的帧区间列表，每个区间代表一个稳定或不稳定阶段。

        cut_kwargs : dict, optional
            可选的参数字典，用于指定合并区间、判定稳定性的控制策略（如相似度阈值等）。

        Notes
        -----
        初始化完成后，可通过本类方法执行：
        - 获取稳定区间与不稳定区间；
        - 对区间进行合并与过滤；
        - 保存样本图像；
        - 对比差异；
        - 序列化与持久化等。
        """
        self.video = video
        self.range_list = range_list
        self.cut_kwargs = cut_kwargs or {}

    def get_target_range_by_id(self, frame_id: int) -> "VideoCutRange":
        """
         根据帧 ID 获取所属的切割区间。

         在当前的 range_list 中查找包含指定帧 ID 的切割区间对象。

         Parameters
         ----------
         frame_id : int
             需要定位的帧编号。

         Returns
         -------
         VideoCutRange
             包含该帧编号的切割区间对象。

         Raises
         ------
         RuntimeError
             如果没有任何区间包含该帧，则抛出异常。

         Notes
         -----
         此方法通常用于标定帧所在的稳定或不稳定阶段。
         """
        for each in self.range_list:
            if each.contain(frame_id):
                return each
        raise RuntimeError(f"frame {frame_id} not found in video")

    @staticmethod
    def _length_filter(range_list: list["VideoCutRange"], limit: int) -> list["VideoCutRange"]:
        """
        过滤出长度大于等于指定阈值的区间段。

        该方法遍历给定的区间列表，仅保留帧数大于或等于限制值的区间，用于剔除过短、无实际意义的片段。

        Parameters
        ----------
        range_list : list of VideoCutRange
            待过滤的区间段列表，通常为稳定或不稳定区间。

        limit : int
            最小帧长度阈值，仅保留长度不小于该值的区间。

        Returns
        -------
        list of VideoCutRange
            过滤后的区间列表，所有区间的长度均不小于给定阈值。

        Notes
        -----
        - 可用于稳定段过滤、训练样本筛选等场景；
        - 实际长度由 `VideoCutRange.get_length()` 方法计算。
        """
        after = list()
        for each in range_list:
            if each.get_length() >= limit:
                after.append(each)
        return after

    def get_unstable_range(
            self, limit: int = None, range_threshold: float = None, **kwargs
    ) -> list["VideoCutRange"]:
        """
        获取所有的不稳定阶段区间。

        遍历当前切割区间，筛选出属于不稳定阶段的区间，并可选择性进行合并、过滤与长度限制。

        Parameters
        ----------
        limit : int, optional
            不稳定区间的最小长度，低于该长度的将被过滤。

        range_threshold : float, optional
            用于判断区间是否构成循环结构的阈值，用于进一步剔除异常区间。

        **kwargs : dict
            传递给 `is_stable()` 和 `can_merge()` 的附加参数，如相似度判断依据。

        Returns
        -------
        list of VideoCutRange
            经过筛选和合并后的不稳定区间列表。

        Notes
        -----
        - 此方法先调用每个区间的 `is_stable()` 方法进行稳定性判断；
        - 相邻不稳定区间在满足 `can_merge()` 条件时将被合并；
        - 可通过 `limit` 过滤掉无效的短区间；
        - 可通过 `range_threshold` 剔除自循环区间（如画面闪动但无实际变化）。
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
        获取稳定与不稳定阶段的完整区间列表。

        基于当前 range_list 划分出稳定与不稳定阶段，并返回两者的切割区间列表。

        Parameters
        ----------
        limit : int, optional
            稳定阶段区间的最小帧长度，小于该值的区间将被过滤。

        unstable_limit : int, optional
            不稳定阶段区间的最小帧长度，用于进一步清洗不稳定区间。

        **kwargs : dict
            可传入给 `is_stable()` 与合并逻辑的参数（如相似度阈值）。

        Returns
        -------
        tuple of list of VideoCutRange
            - 第一个元素为稳定阶段区间列表；
            - 第二个元素为不稳定阶段区间列表。

        Notes
        -----
        - 如果视频整体稳定（无显著变化），则返回整段视频为稳定区间；
        - 否则，按前后顺序判断并构建稳定区间，包括：
            - 视频起始至首个不稳定区间前；
            - 任意两个不稳定区间之间；
            - 最后一个不稳定区间至视频结束；
        - 所有区间均会附加默认图像相似性指标（SSIM、MSE、PSNR）。
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
        """
         获取所有稳定阶段的切割区间。

         该方法是 `get_range()` 的简化版本，只返回稳定区间部分。

         Parameters
         ----------
         limit : int, optional
             稳定区间的最小帧长度限制。

         **kwargs : dict
             传入底层稳定性判断或合并逻辑的参数。

         Returns
         -------
         list of VideoCutRange
             稳定阶段的区间列表。

         Notes
         -----
         本方法依赖 `get_range()` 执行稳定/不稳定划分，并仅保留稳定区间部分。
         """
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
        动态调整阈值以获取期望数量的稳定阶段区间。

        此方法通过递归方式调整稳定性阈值，使稳定区间的数量落入指定范围内，用于自动化视频阶段划分。

        Parameters
        ----------
        stable_num_limit : list of int
            期望稳定阶段数量的上下界，例如 [2, 4]。

        threshold : float
            初始的相似度阈值（例如 SSIM 阈值），用于判断是否为不稳定阶段。

        step : float, optional
            每次调整阈值的步长，默认值为 0.005。

        max_retry : int, optional
            最大递归尝试次数，防止无限循环，默认值为 10。

        **kwargs : dict
            传入给 `get_range()` 的额外参数，例如 `range_threshold`。

        Returns
        -------
        tuple of list of VideoCutRange
            返回两个列表：
            - 稳定阶段区间列表；
            - 不稳定阶段区间列表。

        Raises
        ------
        AssertionError
            若参数非法或递归次数用尽仍无法满足要求时抛出。

        Notes
        -----
        - 若稳定区间数量过少，则自动提高阈值（降低不稳定判定强度）；
        - 若稳定区间数量过多，则降低阈值（收紧稳定性标准）；
        - 使用递归方式反复调整，直至满足期望数量区间或达到最大尝试次数。
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
    ) -> "np.ndarray":
        """
        生成指定区间的帧缩略图拼图。

        从目标区间中采样所有帧，进行压缩并拼接成一张横向或纵向的缩略图。

        Parameters
        ----------
        target_range : VideoCutRange
            目标帧区间对象，包含起止帧 ID。

        to_dir : str, optional
            如果提供，将缩略图保存到该目录。

        compress_rate : float, optional
            帧压缩比例，默认值为 0.1。

        is_vertical : bool, optional
            若为 True，按纵向拼接；否则按横向拼接。

        Returns
        -------
        np.ndarray
            拼接后的缩略图图像数据。

        Notes
        -----
        - 缩略图可用于人工快速浏览指定区间的帧演变情况；
        - 若提供保存路径，将图像写入磁盘；
        - 在每帧间插入 5 像素空白分隔线。
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
        从指定区间中采样并保存图像帧。

        本方法会从每个指定的 `VideoCutRange` 区间中抽取若干帧，进行压缩和预处理后保存到指定目录，
        并可选择是否使用语义化文件名以及是否剔除高度相似的样本。

        Parameters
        ----------
        range_list : list of VideoCutRange
            要处理的帧区间列表，每个区间将从中抽取帧图像。
        frame_count : int
            每个区间采样的帧数。
        to_dir : str, optional
            保存目录路径，若未提供则使用时间戳生成目录名。
        prune : float, optional
            剪枝阈值（SSIM），若设置则会剔除与后续区间高度相似的样本区间。
        meaningful_name : bool, optional
            若为 True，使用包含视频名、帧 ID 和时间戳的语义化文件名，否则使用随机 UUID 命名。
        *args, **kwargs :
            传递给 `compress_frame` 的额外参数（如压缩率、目标尺寸、是否转灰度等）。

        Returns
        -------
        str
            实际用于保存图像的目录路径。

        Notes
        -----
        - 每个子目录对应一个稳定/不稳定阶段；
        - 剪枝机制可避免训练集中出现大量视觉冗余区域；
        - 语义命名便于数据追踪和调试。

        Workflow
        --------
        1. 遍历每个区间，调用 `pick()` 获取采样帧；
        2. 若启用剪枝（`prune`），通过 `multi_compare_ssim` 过滤冗余区间；
        3. 确定输出路径，若无提供则使用时间戳创建新目录；
        4. 遍历每帧图像，压缩并保存到子目录下，以语义或 UUID 命名；
        5. 返回保存根目录。
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
        基于 SSIM 相似度进行区间剪枝。

        该方法用于剔除与其他区间图像高度相似的区间，以减少冗余样本，提高训练效率。

        Parameters
        ----------
        threshold : float
            SSIM 相似度阈值。若某区间与其他区间的最小相似度超过该值，则视为冗余区间并被剔除。

        stages : list of tuple
            每个元素为二元组，包含阶段标识符及对应的帧图像列表。

        Returns
        -------
        list of tuple
            被保留的阶段列表，格式与输入相同。

        Notes
        -----
        - 使用 `toolbox.multi_compare_ssim()` 计算每两个阶段之间的图像相似度；
        - 若存在某阶段与后续某阶段所有图像均高度相似，则该阶段被移除；
        - 该方法是采样后数据筛选的重要步骤，适用于压缩样本数据量。
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
        """
        将当前对象序列化为 JSON 字符串。

        该方法会将 `VideoCutResult` 对象序列化为 JSON 格式文本，并处理 `np.ndarray` 类型字段。

        Returns
        -------
        str
            表示当前对象的 JSON 字符串。

        Notes
        -----
        - `np.ndarray` 会被替换为占位字符串 "<np.ndarray object>"；
        - 使用 `json.dumps()` 结合 `default` 回调方式自定义对象转字典；
        - 常用于调试、报告保存或与 `dump()` 结合使用。
        """

        def _handler(obj: object):
            if isinstance(obj, np.ndarray):
                return "<np.ndarray object>"
            return obj.__dict__

        return json.dumps(self, sort_keys=True, default=_handler)

    def dump(self, json_path: str, **kwargs):
        """
         将当前对象保存为 JSON 文件。

         该方法调用 `dumps()` 获取序列化字符串并写入指定路径的文件中。

         Parameters
         ----------
         json_path : str
             保存 JSON 文件的目标路径。

         **kwargs :
             传递给内建 `open()` 函数的附加参数（如 encoding）。

         Raises
         ------
         AssertionError
             如果目标路径已存在，则抛出异常避免覆盖文件。

         Notes
         -----
         - 若路径已存在将触发断言失败；
         - 推荐在保存前通过 `get_timestamp_str()` 或其他方式生成唯一文件名；
         - 可与 `load()` 搭配使用完成模型结果的持久化与还原。
         """
        logger.debug(f"dump result to {json_path}")
        assert not os.path.exists(json_path), f"{json_path} already existed"
        with open(json_path, "w+", **kwargs) as f:
            f.write(self.dumps())

    @classmethod
    def loads(cls, content: str) -> "VideoCutResult":
        """
        从 JSON 字符串中反序列化出 `VideoCutResult` 实例。

        该方法用于将通过 `dumps()` 序列化的 JSON 字符串内容还原为 `VideoCutResult` 对象。

        Parameters
        ----------
        content : str
            JSON 字符串内容，通常由 `dumps()` 或 `dump()` 方法生成。

        Returns
        -------
        VideoCutResult
            反序列化后的 `VideoCutResult` 对象。

        Notes
        -----
        - 会还原 `VideoObject` 实例以及多个 `VideoCutRange` 实例；
        - 不包含帧图像数据，只包含范围信息和路径标识；
        - 适合用于结果重载、测试对比或断点恢复。
        """
        json_dict: dict = json.loads(content)
        return cls(
            VideoObject(**json_dict["video"]),
            [VideoCutRange(**each) for each in json_dict["range_list"]],
        )

    @classmethod
    def load(cls, json_path: str, **kwargs) -> "VideoCutResult":
        """
        从 JSON 文件中加载 `VideoCutResult` 实例。

        该方法封装了 `loads()`，从文件中读取 JSON 内容并反序列化为结果对象。

        Parameters
        ----------
        json_path : str
            JSON 文件的路径。

        **kwargs :
            传递给内建 `open()` 函数的附加参数（如 encoding）。

        Returns
        -------
        VideoCutResult
            加载完成的 `VideoCutResult` 实例。

        Notes
        -----
        - 要求 JSON 文件为 `dump()` 生成的格式；
        - 若路径不存在或文件格式错误，将触发异常；
        - 常用于还原分析任务结果、脚本重用或增量标注场景。
        """
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
        比较当前结果与另一个结果之间的差异，并生成差异分析对象。

        该方法用于分析两个 `VideoCutResult` 的稳定区间差异，支持可选的自动匹配和预处理操作。

        Parameters
        ----------
        another : VideoCutResult
            用于比较的另一个 `VideoCutResult` 对象。
        auto_merge : bool, optional
            是否启用自动匹配合并结果，仅返回相似度最高的配对（默认 False）。
        pre_hooks : list of BaseHook, optional
            应用于帧图像的预处理钩子列表（如灰度、尺寸调整等）。
        output_path : str, optional
            比对图像的输出目录，用于保存中间图像和可视化结果。
        *args, **kwargs :
            传递给 `get_range()` 与差异计算函数的附加参数。

        Returns
        -------
        VideoCutResultDiff
            差异分析对象，包含每个阶段的 SSIM 对比数据和最优匹配信息。

        Notes
        -----
        - 本方法可用于模型训练集构建前的数据过滤与对齐；
        - 支持阶段名不一致或数量不等的情况；
        - 可扩展为图像级别的评估与推荐系统。

        Workflow
        --------
        1. 分别获取两个结果的稳定区间；
        2. 采样稳定区间并保存图像（如指定输出目录）；
        3. 调用 `VideoCutResultDiff` 分析对应区间的差异；
        4. 若开启 `auto_merge`，则自动匹配最相似的阶段作为最终输出；
        5. 返回分析结果。
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
        计算两个稳定区间列表之间的逐区间差异指标。

        本方法用于对比两个视频分析结果中的稳定区段，逐对计算差异性指标（如 SSIM、MSE、PSNR），
        并返回每个区间组合之间的差异值列表，结果以字典形式组织。

        Parameters
        ----------
        range_list_1 : list of VideoCutRange
            第一个结果中的稳定区间列表，作为参照集合。

        range_list_2 : list of VideoCutRange
            第二个结果中的稳定区间列表，作为比较集合。

        *args, **kwargs :
            传递给 `VideoCutRange.diff()` 的附加参数（如对齐策略、相似度指标等）。

        Returns
        -------
        dict of int -> dict of int -> list of float
            差异值字典。结构为 `{i: {j: [...], ...}, ...}`，其中 i 和 j 为区间索引，
            每个值为一个包含多个指标（例如 SSIM、PSNR 等）的浮点值列表。

        Notes
        -----
        - 两个区间数量可以不同，算法仍可运行；
        - 如果区间数量不一致，会给出警告日志提示；
        - 差异指标的计算依赖于 `VideoCutRange.diff()` 的具体实现；
        - 可用于稳定段落的匹配、排序推荐或样本筛选。
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
    用于比较两个视频切割结果的差异。

    该类接受两个 `VideoCutRange` 列表（通常为稳定阶段的分割结果），并评估每个阶段间的相似性匹配程度。
    提供是否缺失阶段、阶段转移分析、以及结构化的差异对比等功能，常用于训练数据验证或多版本模型评估。
    """

    threshold: float = 0.7
    default_stage_id: int = -1
    default_score: float = -1.0

    def __init__(self, origin: list["VideoCutRange"], another: list["VideoCutRange"]):
        self.origin = origin
        self.another = another
        self.data: typing.Optional[dict[int, dict[int, list[float]]]] = None

    def apply_diff(self, pre_hooks: list["BaseHook"] = None):
        """
        应用阶段差异分析方法，并存储差异数据。

        Parameters
        ----------
        pre_hooks : list of BaseHook, optional
            在进行相似度比较前应用的图像处理钩子，例如裁剪、灰度化等。

        Returns
        -------
        None

        Notes
        -----
        - 通过调用 `VideoCutResult.range_diff()` 方法计算差异；
        - 差异结果存储在 `self.data` 字典中。
        """
        self.data = VideoCutResult.range_diff(self.origin, self.another, pre_hooks)

    def most_common(self, stage_id: int) -> (int, float):
        """
        获取与指定阶段最匹配的阶段 ID 及其最大相似度得分。

        Parameters
        ----------
        stage_id : int
            源阶段编号。

        Returns
        -------
        tuple of (int, float)
            匹配得分最高的目标阶段 ID 及对应相似度分值。

        Raises
        ------
        AssertionError
            如果该阶段不在 `self.data` 中。
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
        判断指定阶段是否在目标视频中缺失。

        Parameters
        ----------
        stage_id : int
            源阶段编号。

        Returns
        -------
        bool
            如果最高匹配得分低于设定阈值，则视为缺失。
        """
        _, v = self.most_common(stage_id)
        return v < self.threshold

    def any_stage_lost(self) -> bool:
        """
        判断是否所有源阶段均未在目标中找到匹配。

        Returns
        -------
        bool
            如果所有阶段匹配得分均低于阈值，返回 True。
        """
        return all((self.is_stage_lost(each) for each in self.data.keys()))

    def stage_shift(self) -> list[int]:
        """
        获取源阶段在目标中的最佳匹配阶段编号列表。

        Returns
        -------
        list of int
            与源阶段顺序对应的匹配目标阶段编号（相似度超过阈值）。
        """
        ret = list()
        for k in self.data.keys():
            new_k, score = self.most_common(k)
            if score > self.threshold:
                ret.append(new_k)
        return ret

    def stage_diff(self) -> typing.Iterator:
        """
        使用 difflib 生成源阶段映射与目标阶段顺序之间的差异对比。

        Returns
        -------
        Iterator
            逐行表示差异的迭代器，可用于展示阶段变化情况。
        """
        return difflib.Differ().compare(
            [str(each) for each in self.stage_shift()],
            [str(each) for each in range(len(self.another))],
        )


if __name__ == '__main__':
    pass
