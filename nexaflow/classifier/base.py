#   ____                    ____ _               _  __ _
#  | __ )  __ _ ___  ___   / ___| | __ _ ___ ___(_)/ _(_) ___ _ __
#  |  _ \ / _` / __|/ _ \ | |   | |/ _` / __/ __| | |_| |/ _ \ '__|
#  | |_) | (_| \__ \  __/ | |___| | (_| \__ \__ \ |  _| |  __/ |
#  |____/ \__,_|___/\___|  \____|_|\__,_|___/___/_|_| |_|\___|_|
#

# ==== Notes: 版权申明 ====
# 版权所有 (c) 2024  Framix(画帧秀)
# 此文件受 Framix(画帧秀) 许可证的保护。您可以在 LICENSE.md 文件中查看详细的许可条款。

# ==== Notes: License ====
# Copyright (c) 2024  Framix(画帧秀)
# This file is licensed under the Framix(画帧秀) License. See the LICENSE.md file for more details.

# ==== Notes: ライセンス ====
# Copyright (c) 2024  Framix(画帧秀)
# このファイルは Framix(画帧秀) ライセンスの下でライセンスされています。詳細は LICENSE.md ファイルを参照してください。

import os
import cv2
import json
import typing
import pathlib
import difflib
import numpy as np
from loguru import logger
from collections import OrderedDict
from nexaflow import (
    toolbox, const
)
from nexaflow.video import (
    VideoFrame, VideoObject
)
from nexaflow.hook import BaseHook
from nexaflow.cutter.cut_range import VideoCutRange


class SingleClassifierResult(object):
    """
    表示视频帧的单个分类结果。

    每个实例对应一个帧在某一时刻的分类信息，包括所属阶段、时间戳、帧编号及可选图像数据。
    提供对图像数据的访问、转换、判断和图像匹配等功能，适用于视频分析中的结果标注与回溯。
    """

    def __init__(
        self,
        video_path: str,
        frame_id: int,
        timestamp: float,
        stage: str,
        data: "np.ndarray" = None,
    ):
        self.video_path: str = video_path
        self.frame_id: int = frame_id
        self.timestamp: float = timestamp
        self.stage: str = stage
        self.data: "np.ndarray" = data

    def to_video_frame(self, *args, **kwargs) -> "VideoFrame":
        """
        将当前的分类结果转换为 `VideoFrame` 对象。
        如果数据已经存在，则直接返回该数据；否则，从视频中加载并压缩该帧的数据。
        """
        if self.data is not None:
            return VideoFrame(self.frame_id, self.timestamp, self.data)

        with toolbox.video_capture(self.video_path) as cap:
            frame = toolbox.get_frame(cap, self.frame_id)
            compressed = toolbox.compress_frame(frame, *args, **kwargs)

        return VideoFrame(self.frame_id, self.timestamp, compressed)

    def get_data(self) -> "np.ndarray":
        """
        获取当前帧的图像数据，调用 `to_video_frame()` 来加载和返回帧数据。
        """
        return self.to_video_frame().data

    def is_stable(self) -> bool:
        """
        判断当前帧是否处于稳定阶段。
        如果分类阶段不属于不稳定、忽略或未知阶段，则视为稳定阶段。
        """
        return self.stage not in (
            const.UNSTABLE_FLAG,
            const.IGNORE_FLAG,
            const.UNKNOWN_STAGE_FLAG,
        )

    def contain_image(
            self, *, image_path: str = None, image_object: "np.ndarray" = None, **kwargs
    ) -> dict[str, typing.Any]:
        """
        检查当前帧是否包含特定图像。
        接受图像路径或图像对象作为输入，并调用 `VideoFrame` 的方法来执行图像匹配。
        """
        return self.to_video_frame().contain_image(
            image_path=image_path, image_object=image_object, **kwargs
        )

    def to_dict(self) -> dict:
        return self.__dict__

    def __str__(self):
        return f"<ClassifierResult stage={self.stage} frame_id={self.frame_id} timestamp={self.timestamp}>"

    __repr__ = __str__


class DiffResult(object):
    """
    比较两个分类结果的阶段序列差异。

    用于比较两个 `ClassifierResult` 实例中提取出的阶段顺序是否一致，并提供差异比对信息。
    适用于模型校验、版本对比、流程一致性检测等场景。
    """

    def __init__(self, origin_data: "ClassifierResult", another_data: "ClassifierResult"):
        """
        初始化 DiffResult 对象，接收两个分类结果用于比对。

        Parameters
        ----------
        origin_data : ClassifierResult
            原始分类结果。

        another_data : ClassifierResult
            用于比较的另一个分类结果。
        """
        self.origin_data = origin_data
        self.another_data = another_data

    @property
    def origin_stage_list(self):
        """
        获取原始分类结果中的阶段顺序列表。

        Returns
        -------
        list of str
            原始分类结果中按顺序排列的阶段名称。
        """
        return self.origin_data.get_ordered_stage_set()

    @property
    def another_stage_list(self):
        """
        获取对比分类结果中的阶段顺序列表。

        Returns
        -------
        list of str
            对比分类结果中按顺序排列的阶段名称。
        """
        return self.another_data.get_ordered_stage_set()

    def ok(self) -> bool:
        """
        判断两个分类结果的阶段顺序是否完全一致。

        Returns
        -------
        bool
            如果两个阶段序列完全一致，则返回 True，否则返回 False。
        """
        return self.origin_stage_list == self.another_stage_list

    def get_diff_str(self) -> typing.Iterator[str]:
        """
        逐行比对两个阶段序列的差异。

        使用 difflib 的文本差异比较器，输出类似 diff 工具的对比结果。

        Returns
        -------
        typing.Iterator[str]
            每一行为一个差异结果字符串，标识是否为新增('+')、删除('-')或一致(' ')。
        """
        return difflib.Differ().compare(self.origin_stage_list, self.another_stage_list)


class ClassifierResult(object):

    LABEL_DATA: str = "data"
    LABEL_VIDEO_PATH: str = "video_path"

    def __init__(self, data: list["SingleClassifierResult"]):
        self.video_path: str = data[0].video_path
        self.data: list["SingleClassifierResult"] = data

    def get_timestamp_list(self) -> list[float]:
        return [each.timestamp for each in self.data]

    def get_stage_list(self) -> list[str]:
        return [each.stage for each in self.data]

    def get_length(self) -> int:
        return len(self.data)

    def get_offset(self) -> float:
        return self.data[1].timestamp - self.data[0].timestamp

    def get_ordered_stage_set(self) -> list[str]:
        """
        获取视频帧阶段的有序去重列表。

        该方法遍历阶段序列，移除连续重复项，保留顺序的唯一阶段名称列表，用于阶段顺序展示或汇总报告。

        Returns
        -------
        list of str
            去除连续重复阶段后的阶段名称列表，保持原始阶段顺序。

        Notes
        -----
        - 该方法依赖 `get_stage_list()` 提供的完整阶段序列。
        - 相同阶段连续出现时仅保留一次，非连续重复不会去重。
        - 常用于构建阶段流、报告摘要或图形标注中的阶段轴。

        Examples
        --------
        self.get_stage_list()
        ['A', 'A', 'B', 'B', 'A', 'C', 'C']
        self.get_ordered_stage_set()
        ['A', 'B', 'A', 'C']
        """
        ret = list()
        for each in self.get_stage_list():
            if not ret:
                ret.append(each)
                continue
            if each == ret[-1]:
                continue
            ret.append(each)
        return ret

    def get_stage_set(self) -> typing.Set[str]:
        return set(self.get_stage_list())

    def to_dict(self) -> dict[str, list[list["SingleClassifierResult"]]]:
        """
        将分类结果转换为以阶段名为键的字典结构。

        该方法根据阶段名对分类结果进行归组，返回一个按阶段组织的有序字典。
        若阶段名为数字，将按数值排序；否则按字典序排序。

        Returns
        -------
        dict of str to list of list of SingleClassifierResult
            一个有序字典，每个键是阶段名称，每个值是对应阶段内的所有帧段列表。
            每个帧段为 `SingleClassifierResult` 实例的列表。

        Notes
        -----
        - 若阶段名称为纯数字，将根据数值进行排序（例如 '2' 在 '10' 前）。
        - 若阶段名称为非数字（如 'A', 'B'），则按默认字符串排序。
        - 用于报告绘制、阶段分布展示、进一步分析等结构化处理场景。

        Examples
        --------
        self.to_dict()
        {
            '1': [[SingleClassifierResult, ...], [...]],
            '2': [[SingleClassifierResult, ...], [...]],
            ...
        }
        """
        stage_list = list(self.get_stage_set())

        try:
            int(stage_list[0])
        except ValueError:
            stage_list.sort()
        else:
            stage_list.sort(key=lambda o: int(o))

        d = OrderedDict()
        for each_stage in stage_list:
            d[each_stage] = self.get_specific_stage_range(each_stage)
        return d

    def contain(self, stage_name: str) -> bool:
        return stage_name in self.get_stage_set()

    def first(self, stage_name: str) -> "SingleClassifierResult":
        """
        获取指定阶段的第一个分类结果帧。

        Parameters
        ----------
        stage_name : str
            阶段名称，用于匹配对应的帧标签。

        Returns
        -------
        SingleClassifierResult
            第一个匹配该阶段名称的帧对象。

        Notes
        -----
        - 该方法从前向后遍历分类结果数据，返回第一个阶段匹配的帧。
        - 如果未找到对应阶段，将记录警告日志并返回 None（未显式返回值）。

        Workflow
        --------
        1. 遍历 self.data 中的每一帧；
        2. 若帧的阶段属性与指定阶段名匹配，则立即返回该帧；
        3. 若未匹配任何帧，记录日志提醒阶段不存在。
        """
        for each in self.data:
            if each.stage == stage_name:
                # logger.debug(f"first frame of {stage_name}: {each}")
                return each
        logger.warning(f"no stage named {stage_name} found")

    def last(self, stage_name: str) -> "SingleClassifierResult":
        """
        获取指定阶段的最后一个分类结果帧。

        Parameters
        ----------
        stage_name : str
            阶段名称，用于匹配对应的帧标签。

        Returns
        -------
        SingleClassifierResult
            最后一个匹配该阶段名称的帧对象。

        Notes
        -----
        - 该方法从后向前遍历分类结果数据，返回最后一个阶段匹配的帧。
        - 如果未找到对应阶段，将记录警告日志并返回 None（未显式返回值）。

        Workflow
        --------
        1. 倒序遍历 self.data 中的每一帧；
        2. 若帧的阶段属性与指定阶段名匹配，则立即返回该帧；
        3. 若未匹配任何帧，记录日志提醒阶段不存在。
        """
        for each in self.data[::-1]:
            if each.stage == stage_name:
                # logger.debug(f"last frame of {stage_name}: {each}")
                return each
        logger.warning(f"no stage named {stage_name} found")

    def get_stage_range(self) -> list[list["SingleClassifierResult"]]:
        """
        获取阶段切分后的帧区间列表。

        按照阶段标签，将帧数据分组，返回每一段连续同阶段的帧列表。

        Parameters
        ----------
        self : ClassifierResult
            分类器输出的结构化结果对象。

        Returns
        -------
        list of list of SingleClassifierResult
            每个子列表代表一个阶段区间，内部是连续同阶段的帧对象集合。

        Notes
        -----
        - 从头遍历所有帧数据；
        - 每当检测到阶段变更时，将当前阶段段落收集；
        - 如果末尾帧未被加入，追加最后一段；
        - 若视频仅包含一个阶段，也能正确处理。

        Workflow
        --------
        1. 初始化当前阶段指针和遍历指针；
        2. 逐帧比对当前阶段与下一个阶段是否一致；
        3. 若不一致，则记录当前阶段段落，并更新指针；
        4. 最后检查末尾帧是否被遗漏；
        5. 返回所有阶段区间的帧段落组成的列表。
        """
        result: list[list["SingleClassifierResult"]] = []

        cur = self.data[0]
        cur_index = cur.frame_id - 1
        ptr = cur_index
        length = self.get_length()
        while ptr < length:
            next_one = self.data[ptr]
            if cur.stage == next_one.stage:
                ptr += 1
                continue

            result.append(self.data[cur_index: ptr + 1 - 1] or [self.data[cur_index]])
            cur = next_one
            cur_index = next_one.frame_id - 1

        assert len(result) > 0, "video seems to only contain one stage"

        last_data = self.data[-1]
        last_result = result[-1][-1]
        if last_result != last_data:
            result.append(
                self.data[last_result.frame_id - 1 + 1: last_data.frame_id - 1 + 1]
                or [self.data[last_result.frame_id - 1]]
            )
        # logger.debug(f"get stage range: {result}")
        return result

    def get_specific_stage_range(self, stage_name: str) -> list[list["SingleClassifierResult"]]:
        """
        获取指定阶段的所有连续帧区间。

        遍历所有阶段区间，提取阶段名与指定参数相符的帧段落。

        Parameters
        ----------
        stage_name : str
            目标阶段名称，如 "stable"、"unstable"、"ignore" 等。

        Returns
        -------
        list of list of SingleClassifierResult
            返回所有阶段标签为 `stage_name` 的连续帧段落，每个子列表表示一段连续区域。

        Notes
        -----
        - 内部调用 `get_stage_range()` 分割整段帧；
        - 判断每段的首帧阶段是否与目标阶段一致；
        - 若一致则记录该段。

        Workflow
        --------
        1. 获取按阶段切分的所有帧段落；
        2. 遍历所有段落，提取阶段标签为目标名称的段；
        3. 返回所有符合条件的帧区间段列表。
        """
        ret = list()
        for each_range in self.get_stage_range():
            cur = each_range[0]
            if cur.stage == stage_name:
                ret.append(each_range)
        return ret

    def get_not_stable_stage_range(self) -> list[list["SingleClassifierResult"]]:
        """
        获取所有“非稳定阶段”帧区间。

        包括阶段标签为“unstable”与“ignore”的所有帧段落。

        Returns
        -------
        list of list of SingleClassifierResult
            返回所有非稳定阶段组成的帧区间列表。

        Notes
        -----
        - 非稳定阶段包括 `const.UNSTABLE_FLAG` 和 `const.IGNORE_FLAG`；
        - 会对提取结果按阶段名称排序；
        - 用于筛选不稳定帧区域，常用于关键帧分析。

        Workflow
        --------
        1. 获取阶段为“unstable”的帧区间；
        2. 获取阶段为“ignore”的帧区间；
        3. 合并两个列表；
        4. 根据每段首帧的阶段名进行排序；
        5. 返回合并并排序后的结果。
        """
        unstable = self.get_specific_stage_range(const.UNSTABLE_FLAG)
        ignore = self.get_specific_stage_range(const.IGNORE_FLAG)

        return sorted(unstable + ignore, key=lambda x: x[0].stage)

    def mark_range(self, start: int, end: int, target_stage: str) -> None:
        """
        将指定范围内的帧标记为特定阶段。

        遍历从 `start` 到 `end` 的帧列表，将每帧的阶段标签更新为 `target_stage`。

        Parameters
        ----------
        start : int
            起始帧的索引（包含）。

        end : int
            结束帧的索引（不包含）。

        target_stage : str
            要设置的目标阶段标签，如 "stable"、"unstable" 或 "ignore"。

        Returns
        -------
        None

        Notes
        -----
        - 索引区间为左闭右开 `[start, end)`；
        - 本方法不会做范围检查，调用者需保证索引合法；
        - 本函数是通用的阶段标记函数，建议配合 `mark_range_unstable` 等封装函数使用。
        """
        for each in self.data[start:end]:
            each.stage = target_stage
        # logger.debug(f"range {start} to {end} has been marked as {target_stage}")

    def mark_range_unstable(self, start: int, end: int) -> None:
        """
        将指定范围的帧标记为“不稳定阶段”。

        这是对 `mark_range()` 的包装，使用预设标签 `const.UNSTABLE_FLAG` 进行标记。

        Parameters
        ----------
        start : int
            起始帧索引（包含）。

        end : int
            结束帧索引（不包含）。

        Returns
        -------
        None

        Notes
        -----
        - 用于快速标记帧为“unstable”状态；
        - 默认使用的标签为 `const.UNSTABLE_FLAG`；
        - 实际调用 `mark_range()` 实现。
        """
        self.mark_range(start, end, const.UNSTABLE_FLAG)

    def mark_range_ignore(self, start: int, end: int) -> None:
        """
        将指定范围的帧标记为“忽略阶段”。

        这是对 `mark_range()` 的包装，使用预设标签 `const.IGNORE_FLAG` 进行标记。

        Parameters
        ----------
        start : int
            起始帧索引（包含）。

        end : int
            结束帧索引（不包含）。

        Returns
        -------
        None

        Notes
        -----
        - 用于将帧标记为无效或不参与分析；
        - 实际调用 `mark_range()` 实现。
        """
        self.mark_range(start, end, const.IGNORE_FLAG)

    def time_cost_between(self, start_stage: str, end_stage: str) -> float:
        """
        计算两个阶段之间的时间差（单位：秒）。

        查找 `start_stage` 最后一个帧与 `end_stage` 第一个帧的时间戳差值。

        Parameters
        ----------
        start_stage : str
            起始阶段名称，作为时间差的起点。

        end_stage : str
            终止阶段名称，作为时间差的终点。

        Returns
        -------
        float
            两阶段间的时间成本（单位：秒）。

        Notes
        -----
        - 若阶段不存在，`first()` 和 `last()` 方法会记录日志警告；
        - 时间差为 `end_stage.first.timestamp - start_stage.last.timestamp`。
        """
        return self.first(end_stage).timestamp - self.last(start_stage).timestamp

    def get_important_frame_list(self) -> list["SingleClassifierResult"]:
        """
        提取阶段切换处的关键帧列表。

        遍历帧序列，记录所有阶段发生变化的位置（即前后帧阶段不同的位置），
        并将变换前后的帧都加入到结果中，以捕捉阶段切换的临界信息。

        Returns
        -------
        list of SingleClassifierResult
            关键帧列表，包含：
            - 视频起始帧；
            - 所有阶段切换点处的前一帧与后一帧；
            - 若最后一帧未包含，则补充末尾帧。

        Notes
        -----
        - 该函数适用于绘制视频阶段概览，或用于阶段边界的分析；
        - 若视频所有帧属于同一阶段，则结果将包含首尾两帧；
        - 输出帧顺序与原视频帧顺序一致，适合直接索引或展示。
        """
        result = [self.data[0]]

        prev = self.data[0]
        for cur in self.data[1:]:
            if cur.stage != prev.stage:
                result.append(prev)
                result.append(cur)
            prev = cur

        if result[-1] != self.data[-1]:
            result.append(self.data[-1])
        return result

    def calc_changing_cost(self) -> dict[str, tuple["SingleClassifierResult", "SingleClassifierResult"]]:
        """
        计算每次阶段切换的起始与结束帧。

        遍历帧序列，识别从稳定阶段进入不稳定阶段再返回稳定阶段的过程，
        并记录每个切换的起始帧与结束帧，用于后续统计或报告展示。

        Returns
        -------
        dict of str to tuple of SingleClassifierResult
            每个阶段切换的标签与其对应的起始帧和结束帧，格式如下：
            {
                "from stage_A to stage_B": (起始帧, 结束帧),
                ...
            }

        Notes
        -----
        - 起始帧是切换前的最后一个稳定帧；
        - 结束帧是切换后的第一个稳定帧；
        - 若视频末尾未再次进入稳定阶段，则最后一段不记录；
        - 可用于统计阶段转换耗时或生成报告摘要。
        """
        cost_dict: dict[str, tuple["SingleClassifierResult", "SingleClassifierResult"]] = {}

        i = 0
        while i < len(self.data) - 1:
            cur = self.data[i]
            next_one = self.data[i + 1]

            if not next_one.is_stable():
                for j in range(i + 1, len(self.data)):
                    i = j
                    next_one = self.data[j]
                    if next_one.is_stable():
                        break

                changing_name = f"from {cur.stage} to {next_one.stage}"
                cost_dict[changing_name] = (cur, next_one)
            else:
                i += 1
        return cost_dict

    def dumps(self) -> str:

        def _handler(obj: object):
            if isinstance(obj, np.ndarray):
                return "<np.ndarray object>"
            return obj.__dict__

        return json.dumps(self, sort_keys=True, default=_handler)

    def dump(self, json_path: str, **kwargs):
        logger.debug(f"dump result to {json_path}")
        assert not os.path.isfile(json_path), f"{json_path} already existed"
        with open(json_path, "w+", **kwargs) as f:
            f.write(self.dumps())

    @classmethod
    def load(cls, from_file: str) -> "ClassifierResult":
        assert os.path.isfile(from_file), f"file {from_file} not existed"

        with open(from_file, encoding=const.CHARSET) as f:
            content = json.load(f)

        data = content[cls.LABEL_DATA]
        return ClassifierResult([SingleClassifierResult(**each) for each in data])

    def diff(self, another: "ClassifierResult") -> DiffResult:
        return DiffResult(self, another)

    def is_order_correct(self, should_be: list[str]) -> bool:
        """
        判断阶段顺序是否符合预期。

        根据当前视频数据中的阶段顺序，判断是否与传入的预期阶段顺序一致。
        支持“包含顺序”判断，即只要 `should_be` 顺序完整且按顺序出现在当前阶段中，即可认为正确。

        Parameters
        ----------
        should_be : list of str
            期望的阶段顺序，例如 ["init", "load", "run", "exit"]

        Returns
        -------
        bool
            如果阶段顺序符合预期（可部分包含），返回 True；否则返回 False。

        Notes
        -----
        - 若阶段数量与期望相同，进行全等比较；
        - 若当前阶段数大于期望阶段数，允许中间插入阶段，只需保留顺序不变；
        - 若当前阶段数少于期望阶段数，直接返回 False；
        - 用于验证阶段执行流程是否合理，常用于回归测试或模型校验。
        """
        cur = self.get_ordered_stage_set()
        len_cur, len_should_be = len(cur), len(should_be)
        if len_cur == len_should_be:
            return cur == should_be
        if len_cur < len_should_be:
            return False

        ptr_should, ptr_cur = 0, 0
        while ptr_cur < len_cur:
            if cur[ptr_cur] == should_be[ptr_should]:
                ptr_should += 1
            ptr_cur += 1
            if ptr_should == len_should_be:
                return True
        return False

    get_frame_length = get_offset


class BaseClassifier(object):
    """
    BaseClassifier 视频帧分类器的基类，提供数据加载、预处理、帧遍历与分类逻辑的通用接口。

    该类支持从目录或指定帧列表加载图像数据，提供基础的 hook 机制支持帧级数据增强、压缩等操作。
    具体分类逻辑由子类通过 `_classify_frame` 实现。

    Attributes
    ----------
    compress_rate : float
        帧压缩比例（与 target_size 二选一），用于视频帧缩放处理。

    target_size : tuple[int, int]
        帧目标尺寸（与 compress_rate 二选一），用于图像缩放。

    _hook_list : list[BaseHook]
        图像预处理 Hook 列表，用于应用灰度化、尺寸调整等操作。

    _data : dict[str, list[pathlib.Path]]
        存储从路径或结构体加载的图像数据，以阶段名称为键。

    Notes
    -----
    - 子类必须实现 `_classify_frame` 方法，以定义实际的分类逻辑；
    - 使用 `load()` 方法可自动判断输入类型，选择对应的数据加载方式；
    - 提供 classify() 方法完成帧遍历与分类，并返回结构化结果 ClassifierResult；
    - 仅支持从路径或 `VideoCutRange` 加载帧，不支持内联帧数据（如帧编号列表）；
    - Hook 扩展机制支持链式帧预处理流程；
    - 默认启用 boost 模式，在连续帧中复用上一个分类结果以提高推理效率。

    Raises
    ------
    TypeError
        当 `load` 输入不为字符串或列表类型时抛出；

    DeprecationWarning
        `read_from_list` 方法已弃用；

    NotImplementedError
        `_classify_frame` 方法未在子类中实现时抛出；

    AssertionError
        在使用 boost 模式但未提供有效帧区间时抛出。
    """

    def __init__(
        self,
        compress_rate: float = None,
        target_size: tuple[int, int] = None,
        *args,
        **kwargs,
    ):
        """
        初始化 BaseClassifier 类。

        该方法用于初始化分类器的压缩配置及钩子列表。若未指定压缩率与目标尺寸，将默认设置压缩率为 0.2。

        Parameters
        ----------
        compress_rate : float, optional
            压缩比例（范围为 0~1 之间的小数），用于统一缩放图像尺寸以减少计算负载。

        target_size : tuple[int, int], optional
            指定图像目标尺寸，优先级高于 compress_rate，若指定将直接按此尺寸调整图像。

        *args :
            预留扩展参数，当前未使用。

        **kwargs :
            预留关键字参数，当前未使用。

        Attributes
        ----------
        compress_rate : float
            当前图像压缩比例。

        target_size : tuple[int, int]
            图像目标尺寸。

        _data : dict[str, list[pathlib.Path]]
            存储训练或测试数据的路径或对象。

        _hook_list : list[BaseHook]
            注册的钩子列表，每个钩子用于图像处理的某一步。

        Notes
        -----
        - 若 compress_rate 与 target_size 均未指定，默认设置 compress_rate 为 0.2。
        - 本类为抽象基类，具体分类逻辑需子类实现 `_classify_frame` 方法。
        """
        if compress_rate is None and target_size is None:
            # logger.debug(
            #     f"no compress rate or target size received. set compress rate to 0.2"
            # )
            compress_rate = 0.2

        self.compress_rate = compress_rate
        self.target_size = target_size
        # logger.debug(f"compress rate: {self.compress_rate}")
        # logger.debug(f"target size: {self.target_size}")

        self._data: dict[str, typing.Union[list[pathlib.Path]]] = dict()

        self._hook_list: list["BaseHook"] = list()
        # compress_hook = FrameSizeHook(
        #     overwrite=True, compress_rate=compress_rate, target_size=target_size
        # )
        # grey_hook = GreyHook(overwrite=True)
        # self.add_hook(compress_hook)
        # self.add_hook(grey_hook)

    def add_hook(self, new_hook: "BaseHook") -> None:
        """
        添加一个图像处理钩子到分类器流程中。

        本方法将用户传入的钩子实例追加至 `_hook_list` 中，后续所有帧处理流程中将依序应用这些钩子。

        Parameters
        ----------
        new_hook : BaseHook
            一个继承自 BaseHook 的钩子对象，用于执行图像的预处理操作，如尺寸调整、灰度转换等。

        Returns
        -------
        None

        Notes
        -----
        - 所有添加的钩子将按照添加顺序依次应用于每一帧图像。
        - 钩子对象必须实现 `do()` 方法，接受 `VideoFrame` 实例并返回处理后的帧。
        """
        self._hook_list.append(new_hook)
        # logger.debug(f"add hook: {new_hook.__class__.__name__}")

    def load(self, data: typing.Union[str, list["VideoCutRange"], None], *args, **kwargs) -> None:
        """
        加载训练或推理所需的图像数据。

        根据传入数据类型选择不同的加载方式：
        - 字符串类型视为目录路径，调用 `load_from_dir` 加载目录中的图像数据；
        - 列表类型视为 `VideoCutRange` 对象集合，调用 `load_from_list`；
        - 其他类型则视为非法，抛出类型错误。

        Parameters
        ----------
        data : Union[str, list[VideoCutRange], None]
            数据来源，支持两种类型：
            - str: 数据目录路径，每个子目录代表一个分类；
            - list[VideoCutRange]: 表示某个阶段的帧范围，用于训练。

        *args :
            可选的其他位置参数，传递给对应的加载方法。

        **kwargs :
            可选的其他关键字参数，传递给对应的加载方法。

        Returns
        -------
        None

        Raises
        ------
        TypeError
            如果参数 `data` 既不是字符串也不是列表类型，则抛出异常。

        Workflow
        --------
        1. 判断 data 类型；
        2. 如果为 str，则调用 `load_from_dir`；
        3. 如果为 list，则调用 `load_from_list`；
        4. 否则抛出 TypeError。
        """
        if isinstance(data, str):
            return self.load_from_dir(data, *args, **kwargs)
        if isinstance(data, list):
            return self.load_from_list(data, *args, **kwargs)
        raise TypeError(f"data type error, should be str or typing.List[VideoCutRange]")

    def load_from_list(self, data: list["VideoCutRange"], frame_count: int = None, *_, **__) -> None:
        """
        从 VideoCutRange 列表中加载阶段图像数据。

        本方法用于处理以阶段划分的视频帧区间，通过每个阶段的 `pick()` 方法抽取代表性帧，
        并将这些帧保存到内部 `_data` 字典中以供后续训练或推理使用。

        Parameters
        ----------
        data : list[VideoCutRange]
            一个包含若干阶段（如稳定/不稳定等）帧区间的列表，每项为 VideoCutRange 实例。
        frame_count : int, optional
            每个阶段需采样的帧数量。如果为 None，则使用默认行为（如返回全部或固定比例）。
        *_, **__ :
            保留参数，不参与逻辑，可用于兼容扩展。

        Returns
        -------
        None

        Workflow
        --------
        1. 遍历 `data` 中的每个阶段区间；
        2. 使用 `.pick()` 方法提取指定数量的帧；
        3. 将结果存入 `_data` 属性中，以阶段索引（字符串形式）为 key。
        """
        for stage_name, stage_data in enumerate(data):
            target_frame_list = stage_data.pick(frame_count)
            self._data[str(stage_name)] = target_frame_list

    def load_from_dir(self, dir_path: str, *_, **__) -> None:
        """
        从目录中加载分阶段图像数据。

        本方法用于从指定目录中读取不同阶段的图像文件夹，
        并将每个子目录视为一个阶段（例如稳定、异常等），提取其中的图像路径保存到 `_data` 中。

        Parameters
        ----------
        dir_path : str
            图像数据所在的根目录路径，结构需为多阶段文件夹，如：
            - root/
                ├── stage_A/
                ├── stage_B/
                └── ...

        *_, **__ :
            保留参数，不参与当前逻辑，仅用于兼容接口。

        Returns
        -------
        None

        Workflow
        --------
        1. 遍历 `dir_path` 下的所有子目录；
        2. 忽略非目录项（如根目录下的图像文件）；
        3. 收集每个子目录下的所有图像路径，保存到 `_data` 中；
           以子目录名作为阶段名称（key），图像路径列表作为值（value）。
        """
        p = pathlib.Path(dir_path)
        stage_dir_list = p.iterdir()

        for each in stage_dir_list:
            if each.is_file():
                continue
            stage_name = each.name
            stage_pic_list = [i.absolute() for i in each.iterdir()]
            self._data[stage_name] = stage_pic_list
            # logger.debug(
            #     f"stage [{stage_name}] found, and got {len(stage_pic_list)} pics"
            # )

    def read(self, *args, **kwargs) -> typing.Generator:
        """
        按阶段读取图像数据，返回生成器。

        本方法会根据 `_data` 中记录的阶段数据，生成每一阶段对应的图像读取生成器。
        每次迭代返回一个元组 `(阶段名, 图像生成器)`，用于后续模型训练或分析流程。

        Parameters
        ----------
        *args, **kwargs :
            传递给读取函数 `read_from_path()` 的附加参数，可用于控制图像读取行为。

        Returns
        -------
        typing.Generator
            生成器对象，每次返回一个 `(stage_name, image_generator)` 元组。
            `image_generator` 是从磁盘加载图像的生成器。

        Raises
        ------
        TypeError
            如果数据类型不符合预期（例如帧数据不为 `Path` 类型），则抛出类型错误。

        Notes
        -----
        当前仅支持从路径读取数据（`Path` 类型），不支持帧对象列表等形式。
        """
        for stage_name, stage_data in self._data.items():
            if isinstance(stage_data[0], pathlib.Path):
                yield stage_name, self.read_from_path(stage_data, *args, **kwargs)
            else:
                raise TypeError(
                    f"data type error, should be str or typing.List[VideoCutRange]"
                )

    @staticmethod
    def read_from_path(data: list["pathlib.Path"], *_, **__) -> typing.Generator:
        """
        从文件路径列表中读取图像数据，返回生成器。

        本方法遍历路径列表 `data`，依次使用 `toolbox.imread()` 读取图像文件，并通过生成器逐一返回图像数据。

        Parameters
        ----------
        data : list[pathlib.Path]
            图像文件路径组成的列表。
        *_, **__ :
            保留参数，用于兼容其他上下文调用。

        Returns
        -------
        typing.Generator
            返回一个生成器，每次迭代返回一张图像（`np.ndarray` 类型）。

        Notes
        -----
        - 若图像读取失败，将由 `toolbox.imread()` 抛出异常。
        - 生成器形式适用于大批量图像懒加载场景。
        """
        return (toolbox.imread(each.as_posix()) for each in data)

    def read_from_list(self, data: list[int], video_cap: cv2.VideoCapture = None, *_, **__):
        raise DeprecationWarning("this function already deprecated")

    def _classify_frame(self, frame: "VideoFrame", *args, **kwargs) -> str:
        raise NotImplementedError

    def _apply_hook(self, frame: "VideoFrame", *args, **kwargs) -> "VideoFrame":
        """
        应用所有已注册的 Hook 对当前帧进行处理。

        本方法依次将当前帧 `frame` 传入已注册的所有 Hook 的 `do()` 方法进行处理，并返回最终处理结果。
        Hook 的作用包括但不限于压缩、灰度化、裁剪、忽略区域等操作。

        Parameters
        ----------
        frame : VideoFrame
            当前需要处理的视频帧对象。

        *args :
            传递给 Hook 的位置参数。

        **kwargs :
            传递给 Hook 的关键字参数。

        Returns
        -------
        VideoFrame
            经所有 Hook 处理后的帧对象。

        Notes
        -----
        - 所有 Hook 必须继承自 `BaseHook` 并实现 `do()` 方法。
        - Hook 的应用顺序与添加顺序一致，先添加先执行。
        - 每个 Hook 应返回修改后的 `VideoFrame`，否则可能影响后续操作。
        """
        for each_hook in self._hook_list:
            frame = each_hook.do(frame, *args, **kwargs)
        return frame

    def classify(
        self,
        video: typing.Union[str, "VideoObject"],
        valid_range: list["VideoCutRange"] = None,
        step: int = None,
        keep_data: bool = None,
        boost_mode: bool = None,
        *args,
        **kwargs,
    ) -> "ClassifierResult":
        """
        对视频中的所有帧进行分类，返回分类结果对象。

        该方法基于预设的模型或分类规则，对视频中的帧逐帧进行推理和标注，并生成结构化分类结果。
        支持按步长抽帧、按有效区域分类、缓存结果加速分析，以及是否保留帧图像数据等控制参数。

        Parameters
        ----------
        video : Union[str, VideoObject]
            视频对象或视频路径。可直接传入 `VideoObject`，也可传入视频文件路径。

        valid_range : list of VideoCutRange, optional
            指定的有效帧区间列表，仅在区间内的帧才会被分类，其余帧将标记为 IGNORE。
            仅在 `boost_mode=True` 时启用。

        step : int, optional
            分类的步长，默认为逐帧分类（1），可设为 >1 以加速推理。

        keep_data : bool, optional
            是否保留帧图像数据至分类结果中。默认为 False，可用于后续绘图。

        boost_mode : bool, optional
            是否启用推理加速模式：
            - 若为 True：连续帧将复用上一个结果，仅在变化处进行模型调用。
            - 若为 False：每帧独立推理。
            - 启用时必须提供 `valid_range` 参数。

        Returns
        -------
        ClassifierResult
            分类结果对象，包含所有帧的推理标签、帧号、时间戳等信息。

        Raises
        ------
        AssertionError
            当启用了 `boost_mode` 但未传入 `valid_range` 时抛出。

        Notes
        -----
        - 调用前建议先通过 `add_hook()` 添加预处理 Hook，如灰度、压缩、裁剪等。
        - 推理结果会封装成 `SingleClassifierResult` 列表，并构建为 `ClassifierResult` 返回。
        - 结果可用于后续阶段识别、稳定性判断和报告生成等任务。

        Workflow
        --------
        1. 构造视频读取器（VideoObject 或路径）；
        2. 获取帧操作器（Mem 或 Doc 模式）；
        3. 初始化进度条，遍历帧；
        4. 对每一帧应用 Hook，判断是否在有效范围；
        5. 若在范围内，根据 boost 模式执行分类；
        6. 构建结果对象并返回。
        """

        # logger.debug(f"classify with {self.__class__.__name__}")
        step = step or 1
        boost_mode = boost_mode or True

        assert (boost_mode and valid_range) or (
            not (boost_mode or valid_range)
        ), "boost_mode required valid_range"

        final_result: list["SingleClassifierResult"] = list()
        if isinstance(video, str):
            video = VideoObject(video)

        operator = video.get_operator()
        frame = operator.get_frame_by_id(1)

        prev_result: typing.Optional[str] = None
        progress_bar = toolbox.show_progress(total=video.frame_count, color=38)
        while frame is not None:
            frame = self._apply_hook(frame, *args, **kwargs)
            if valid_range and not any(
                [each.contain(frame.frame_id) for each in valid_range]
            ):
                # logger.debug(
                #     f"frame {frame.frame_id} ({frame.timestamp}) not in target range, skip"
                # )
                result = const.IGNORE_FLAG
                prev_result = None
            else:
                if boost_mode and (prev_result is not None):
                    result = prev_result
                else:
                    prev_result = result = self._classify_frame(frame, *args, **kwargs)
                # logger.debug(
                #     f"frame {frame.frame_id} ({frame.timestamp}) belongs to {result}"
                # )

            final_result.append(
                SingleClassifierResult(
                    video.path,
                    frame.frame_id,
                    frame.timestamp,
                    result,
                    frame.data if keep_data else None,
                )
            )
            frame = operator.get_frame_by_id(frame.frame_id + step)
            progress_bar.update(1)
        progress_bar.close()

        return ClassifierResult(final_result)


class BaseModelClassifier(BaseClassifier):
    """
    BaseModelClassifier 抽象基类，用于定义基于模型的视频帧分类器的统一接口。

    该类继承自 BaseClassifier，主要用于深度学习/机器学习模型相关的分类任务，要求子类实现模型的训练、加载、预测等核心方法。

    Notes
    -----
    - 所有核心方法（如 `save_model`、`load_model`、`train`、`predict` 等）均为抽象接口或异常抛出，需在子类中重写；
    - `read_from_list` 明确限制了数据输入方式，强制要求仅支持从文件中加载；
    - 设计目的是为结构化、模型驱动的分类器提供统一的调用规范。

    Raises
    ------
    - NotImplementedError: 所有抽象方法在未实现时均抛出此异常；
    - ValueError: `read_from_list` 被调用时，统一抛出异常提示不支持该调用方式。
    """

    def save_model(self, model_path: str, overwrite: bool = None):
        raise NotImplementedError

    def load_model(self, model_path: str, overwrite: bool = None):
        raise NotImplementedError

    def clean_model(self):
        raise NotImplementedError

    def train(self, data_path: str = None, *args, **kwargs):
        raise NotImplementedError

    def predict(self, pic_path: str, *_, **__) -> str:
        raise NotImplementedError

    def predict_with_object(self, frame: np.ndarray) -> str:
        raise NotImplementedError

    def read_from_list(self, data: list[int], video_cap: cv2.VideoCapture = None, *_, **__):
        raise ValueError("model-like classifier only support loading data from files")


if __name__ == '__main__':
    pass
