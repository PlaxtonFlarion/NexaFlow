#
#   ____                    ____ _               _  __ _
#  | __ )  __ _ ___  ___   / ___| | __ _ ___ ___(_)/ _(_) ___ _ __
#  |  _ \ / _` / __|/ _ \ | |   | |/ _` / __/ __| | |_| |/ _ \ '__|
#  | |_) | (_| \__ \  __/ | |___| | (_| \__ \__ \ |  _| |  __/ |
#  |____/ \__,_|___/\___|  \____|_|\__,_|___/___/_|_| |_|\___|_|
#

import os
import cv2
import json
import typing
import pathlib
import difflib
import numpy as np
from loguru import logger
from collections import OrderedDict
from nexaflow import toolbox, const
from nexaflow.video import VideoFrame, VideoObject
from nexaflow.cutter.cut_range import VideoCutRange
from nexaflow.hook import BaseHook


class SingleClassifierResult(object):
    """
    单一分类器结果类，用于存储和操作视频中的单个帧的分类信息及其相关数据。

    属性:
        video_path : str
            视频文件的路径。
        frame_id : int
            视频帧的唯一标识符。
        timestamp : float
            帧在视频中的时间戳。
        stage : str
            分类阶段标签，用于标识该帧属于哪个阶段。
        data : np.ndarray, optional
            帧的图像数据（默认为None），如果为空，后续可以通过加载视频帧来获取图像数据。
    用途:
        该类主要用于存储和操作视频分类器生成的单帧分类结果，支持数据的持久化、转换及相关判断操作。
        它可以帮助用户处理视频分类过程中的帧数据，进行图像分析和状态判断等。
    """

    def __init__(
        self,
        video_path: str,
        frame_id: int,
        timestamp: float,
        stage: str,
        data: np.ndarray = None,
    ):
        self.video_path: str = video_path
        self.frame_id: int = frame_id
        self.timestamp: float = timestamp
        self.stage: str = stage
        self.data: np.ndarray = data

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

    def get_data(self) -> np.ndarray:
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
            self, *, image_path: str = None, image_object: np.ndarray = None, **kwargs
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
    `DiffResult` 类简要说明：

    `DiffResult` 类用于比较两个分类器结果 (`ClassifierResult`) 的阶段列表，提供差异分析和比较功能。

    ### 主要功能：
        - **__init__**: 初始化方法，接受两个 `ClassifierResult` 对象，分别表示原始数据和比较数据。
        - **origin_stage_list** (属性): 返回 `origin_data` 中的有序阶段列表，通过调用 `origin_data.get_ordered_stage_set()` 方法获取。
        - **another_stage_list** (属性): 返回 `another_data` 中的有序阶段列表，通过调用 `another_data.get_ordered_stage_set()` 方法获取。
        - **ok**: 判断 `origin_stage_list` 和 `another_stage_list` 是否完全相同。如果相同，返回 `True`，否则返回 `False`。
        - **get_diff_str**: 使用 `difflib.Differ` 对比 `origin_stage_list` 和 `another_stage_list`，生成并返回差异字符串。

    ### 设计意图：
        `DiffResult` 类主要用于比较两个阶段列表的差异，通过简单的接口可以判断两个分类器结果的阶段是否一致，并且可以生成详细的差异报告，帮助用户识别数据中的变化或差异。
    """

    def __init__(self, origin_data: "ClassifierResult", another_data: "ClassifierResult"):
        self.origin_data = origin_data
        self.another_data = another_data

    @property
    def origin_stage_list(self):
        return self.origin_data.get_ordered_stage_set()

    @property
    def another_stage_list(self):
        return self.another_data.get_ordered_stage_set()

    def ok(self) -> bool:
        return self.origin_stage_list == self.another_stage_list

    def get_diff_str(self):
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
        获取按顺序排列的唯一阶段列表。

        返回值:
            - list[str]: 返回一个按顺序排列的阶段标识字符串列表，其中每个阶段只出现一次。

        方法说明:
            1. 初始化一个空列表 `ret` 用于存储唯一的阶段标识。
            2. 遍历通过 `get_stage_list` 方法获取的阶段列表 `each`。
            3. 如果 `ret` 为空，则将当前阶段 `each` 添加到 `ret` 中。
            4. 如果当前阶段 `each` 与 `ret` 中的最后一个阶段不同，则将其添加到 `ret` 中，以确保每个阶段在结果列表中只出现一次。
            5. 最终返回构建好的有序唯一阶段列表 `ret`。

        使用场景:
            该方法适用于需要获取按顺序排列的阶段标识列表的场景，确保返回的列表中每个阶段标识只出现一次，且保持原始顺序。

        注意事项:
            - `get_stage_list` 方法假定返回一个按顺序排列的阶段列表。
            - 该方法不会对阶段列表进行排序或重新排列，只是确保阶段标识在结果中是唯一的并按顺序出现。
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
        将分类结果转换为字典格式，其中每个阶段的结果作为字典的键，对应的分类结果列表作为值。

        返回值:
            - dict[str, list[list["SingleClassifierResult"]]]: 返回一个有序字典，其中键是阶段的标识（字符串），值是该阶段对应的分类结果列表。

        方法说明:
            1. 首先，调用 `get_stage_set` 方法获取所有阶段的集合，并将其转换为列表 `stage_list`。
            2. 尝试将阶段的第一个元素转换为整数，如果成功，则按整数排序，否则按字典顺序排序。
            3. 创建一个 `OrderedDict` 对象 `d`，确保结果字典保持阶段的顺序。
            4. 遍历排序后的阶段列表 `stage_list`，对于每个阶段，调用 `get_specific_stage_range` 方法获取该阶段的分类结果，并将其存储在字典中。
            5. 最终返回构建好的有序字典。

        异常处理:
            - 在尝试将阶段转换为整数时，如果出现 `ValueError` 异常（即阶段不是整数），则按字典顺序排序。
            - 该方法假定 `get_stage_set` 和 `get_specific_stage_range` 方法已经正确实现并返回预期的结果。

        使用示例:
            调用此方法可以将阶段分类结果转换为字典格式，以便更方便地进行数据访问和分析。
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
        for each in self.data:
            if each.stage == stage_name:
                # logger.debug(f"first frame of {stage_name}: {each}")
                return each
        logger.warning(f"no stage named {stage_name} found")

    def last(self, stage_name: str) -> "SingleClassifierResult":
        for each in self.data[::-1]:
            if each.stage == stage_name:
                # logger.debug(f"last frame of {stage_name}: {each}")
                return each
        logger.warning(f"no stage named {stage_name} found")

    def get_stage_range(self) -> list[list["SingleClassifierResult"]]:
        """
        从视频数据中提取并返回各个阶段的帧范围列表。

        此方法遍历内部视频数据（self.data），根据帧的阶段属性（stage）对视频帧进行分组。
        每个分组包含一系列同阶段的连续帧。当检测到阶段变化时，当前阶段的帧列表会被添加到结果列表中，并开始收集下一个阶段的帧。

        返回的列表包含多个子列表，每个子列表代表一个视频阶段中的所有帧。

        示例：
            如果视频有 30 帧，分为 3 个阶段（0-11，12-20，21-30 帧），则此方法将返回:
            [
                [frame_0, frame_1, ..., frame_11],
                [frame_12, frame_13, ..., frame_20],
                [frame_21, frame_22, ..., frame_30]
            ]

        Returns:
            list[list[SingleClassifierResult]]: 包含每个阶段所有帧的列表的列表。

        Raises:
            AssertionError: 如果视频数据中所有帧均为同一阶段，抛出断言错误。
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
        根据指定的阶段名称返回包含该阶段所有帧的列表。

        此方法调用 `get_stage_range` 来获取视频数据中所有阶段的帧范围，然后筛选出与给定阶段名称匹配的范围。
        每个阶段由一个列表表示，其中包含属于该阶段的所有`SingleClassifierResult`对象。

        Parameters:
            stage_name (str): 要检索帧的视频阶段的名称。

        Returns:
            list[list[SingleClassifierResult]]: 包含每个匹配阶段所有帧的列表的列表。每个内部列表包含
            一个阶段的所有帧，只返回匹配指定`stage_name`的阶段。
        """
        ret = list()
        for each_range in self.get_stage_range():
            cur = each_range[0]
            if cur.stage == stage_name:
                ret.append(each_range)
        return ret

    def get_not_stable_stage_range(self) -> list[list["SingleClassifierResult"]]:
        """
        获取视频中不稳定和忽略的阶段范围，并按阶段进行排序。

        返回值:
            - list[list["SingleClassifierResult"]]: 返回一个嵌套列表，包含所有不稳定阶段和忽略阶段的范围。每个范围是一个 `SingleClassifierResult` 对象的列表。

        方法说明:
            该方法首先调用 `get_specific_stage_range` 获取标记为不稳定阶段 (`UNSTABLE_FLAG`) 的范围，然后获取标记为忽略阶段 (`IGNORE_FLAG`) 的范围。
            随后，将这两个列表合并，并根据每个范围的第一个阶段 (`stage`) 进行排序，以确保输出结果按顺序排列。

        异常处理:
            该方法假定 `get_specific_stage_range` 方法已经正确实现并返回预期的结果列表。如果该假设无效，可能会导致返回结果不准确或排序错误。

        使用示例:
            调用此方法可获取视频中所有不稳定和忽略的阶段范围，并按阶段顺序进行后续处理或分析。
        """
        unstable = self.get_specific_stage_range(const.UNSTABLE_FLAG)
        ignore = self.get_specific_stage_range(const.IGNORE_FLAG)
        return sorted(unstable + ignore, key=lambda x: x[0].stage)

    def mark_range(self, start: int, end: int, target_stage: str):
        for each in self.data[start:end]:
            each.stage = target_stage
        # logger.debug(f"range {start} to {end} has been marked as {target_stage}")

    def mark_range_unstable(self, start: int, end: int):
        self.mark_range(start, end, const.UNSTABLE_FLAG)

    def mark_range_ignore(self, start: int, end: int):
        self.mark_range(start, end, const.IGNORE_FLAG)

    def time_cost_between(self, start_stage: str, end_stage: str) -> float:
        return self.first(end_stage).timestamp - self.last(start_stage).timestamp

    def get_important_frame_list(self) -> list["SingleClassifierResult"]:
        """
        从存储的帧数据集中提取关键帧列表。关键帧包括：
            - 数据集中的第一帧；
            - 每个阶段变化前后的帧；
            - 数据集中的最后一帧（如果尚未被包括）。

        关键帧用于标示视频或时间序列数据中的重要转折点，如阶段变化，它们对于后续的数据分析和处理尤为重要，比如视频编辑中的剪辑点检测，或者机器学习模型的训练数据制备。

        Returns:
            list: 包含`SingleClassifierResult`对象的列表，这些对象代表了识别出的关键帧。
                  每个对象包含帧的数据以及可能的其他分类或分析结果。
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
        计算并返回数据集中每次稳定到不稳定状态之间的变化成本。

        此方法遍历`self.data`中的帧序列，识别连续的不稳定帧，并记录下这些阶段变化的开始和结束帧。
        例如，如果一个帧序列从稳定状态变为不稳定，然后再次变为稳定，这个函数将记录下变化开始和结束时的帧。

        Returns:
            dict[str, tuple[SingleClassifierResult, SingleClassifierResult]]:
                一个字典，其键为变化的描述（例如 "from stage1 to stage2"），值为一个元组，其中包含表示变化开始的帧和变化结束的帧的`SingleClassifierResult`对象。
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
        检查当前阶段顺序是否符合预期的顺序。

        参数:
            - should_be (list[str]): 预期的阶段顺序列表。

        返回值:
            - bool: 如果当前阶段顺序与预期顺序相符，返回 `True`；否则返回 `False`。

        方法说明:
            1. 调用 `get_ordered_stage_set` 获取当前阶段的有序列表 `cur`。
            2. 比较当前阶段列表 `cur` 和预期阶段列表 `should_be` 的长度：
                - 如果长度相等，则直接比较两个列表是否完全相同。
                - 如果当前列表长度小于预期列表长度，则直接返回 `False`，因为当前列表无法包含所有预期的阶段。
            3. 如果当前列表长度大于预期列表长度，则通过两个指针 `ptr_should` 和 `ptr_cur` 遍历列表：
                - 指针 `ptr_should` 用于遍历预期列表 `should_be`。
                - 指针 `ptr_cur` 用于遍历当前列表 `cur`。
                - 如果当前列表中的某个阶段与预期列表中的当前阶段匹配，移动 `ptr_should` 以检查下一个预期阶段。
                - 如果所有预期阶段都在当前列表中按顺序匹配，则返回 `True`。
            4. 如果遍历完当前列表后，仍未匹配所有预期阶段，则返回 `False`。

        使用场景:
            该方法适用于需要验证当前对象的阶段顺序是否与预期顺序相匹配的场景，特别是在阶段列表可能包含额外元素的情况下。

        注意事项:
            - 该方法假定 `should_be` 列表是严格按照顺序排列的。
            - 当前阶段列表 `cur` 中可能包含不在 `should_be` 列表中的额外阶段，这些阶段会被忽略，方法关注的仅仅是 `should_be` 列表中元素的顺序匹配情况。
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

    def __init__(
        self,
        compress_rate: float = None,
        target_size: tuple[int, int] = None,
        *args,
        **kwargs,
    ):

        if (not compress_rate) and (not target_size):
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
        self._hook_list.append(new_hook)
        # logger.debug(f"add hook: {new_hook.__class__.__name__}")

    def load(self, data: typing.Union[str, list["VideoCutRange"], None], *args, **kwargs) -> None:
        if isinstance(data, str):
            return self.load_from_dir(data, *args, **kwargs)
        if isinstance(data, list):
            return self.load_from_list(data, *args, **kwargs)
        raise TypeError(f"data type error, should be str or typing.List[VideoCutRange]")

    def load_from_list(self, data: list["VideoCutRange"], frame_count: int = None, *_, **__) -> None:
        """
        将一组 `VideoCutRange` 数据加载到 `_data` 字典中，并提取指定数量的帧数。

        参数:
            - data (list["VideoCutRange"]): `VideoCutRange` 对象的列表，每个对象代表一个视频剪辑范围。
            - frame_count (int, 可选): 指定要从每个 `VideoCutRange` 中提取的帧的数量。如果未提供，将使用默认数量。
            - *_, **__: 预留的额外位置参数和关键字参数，当前未使用，但可用于扩展。

        方法说明:
            1. 遍历传入的 `data` 列表，并使用 `enumerate` 为每个 `VideoCutRange` 分配一个唯一的阶段名称 `stage_name`。
            2. 对于每个 `VideoCutRange` 对象 `stage_data`，调用其 `pick` 方法，提取指定数量的帧，并将结果保存到 `_data` 字典中。
            3. `_data` 字典的键为 `stage_name` 的字符串形式，值为提取的帧列表。

        返回值:
            - 无返回值。该方法直接修改实例的 `_data` 字典，将每个阶段提取的帧列表存储其中。

        使用场景:
            该方法用于从 `VideoCutRange` 对象列表中提取并存储帧数据，以便后续处理或分析。

        注意事项:
            - `data` 参数必须是包含 `VideoCutRange` 对象的列表。
            - `frame_count` 参数决定了从每个 `VideoCutRange` 中提取的帧数。
            - `_data` 字典的键为阶段名称的字符串，值为对应的帧数据列表。
            - 如果 `data` 列表中的 `VideoCutRange` 对象数量较多，且 `frame_count` 设置较大，提取操作可能耗时较长。
        """
        for stage_name, stage_data in enumerate(data):
            target_frame_list = stage_data.pick(frame_count)
            self._data[str(stage_name)] = target_frame_list

    def load_from_dir(self, dir_path: str, *_, **__) -> None:
        """
        从指定目录加载阶段数据，并将其存储到实例的数据字典 `_data` 中。

        参数:
            - dir_path (str): 需要加载的目录路径。
            - *_, **__: 忽略的其他位置参数和关键字参数。

        方法说明:
            1. 将 `dir_path` 转换为 `pathlib.Path` 对象 `p`，以便使用面向对象的方式操作路径。
            2. 使用 `iterdir()` 方法遍历 `dir_path` 目录下的所有文件和子目录。
            3. 对每个遍历到的子目录 `each`：
                - 如果 `each` 是文件，则跳过不处理。
                - 如果 `each` 是目录，获取该目录的名称 `stage_name` 作为阶段的标识。
                - 使用 `iterdir()` 获取该阶段目录中的所有文件路径，并将它们的绝对路径保存在 `stage_pic_list` 列表中。
                - 将 `stage_name` 作为键，`stage_pic_list` 作为值，存储到实例的 `_data` 字典中。

        返回值:
            - 无返回值。该方法直接对实例的 `_data` 属性进行修改。

        使用场景:
            该方法适用于从文件系统中加载分阶段存储的图片或文件路径，将这些路径映射到各自的阶段名上，方便后续操作。

        注意事项:
            - 该方法假设 `dir_path` 中的子目录命名是有意义的，且代表不同的阶段。
            - 文件名将被忽略，只有目录中的文件路径会被加载。
            - `stage_name` 必须是目录名，并且 `_data` 字典中相同 `stage_name` 的数据会被覆盖。
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
        读取实例数据字典 `_data` 中的每个阶段数据，并通过生成器逐个返回阶段名称和对应的数据。

        参数:
            - *args, **kwargs: 传递给 `read_from_path` 方法的额外位置参数和关键字参数。

        方法说明:
            1. 遍历实例的 `_data` 字典，对于每个 `stage_name` 和 `stage_data`：
                - 检查 `stage_data` 列表的第一个元素是否为 `pathlib.Path` 类型。
                - 如果是 `pathlib.Path` 类型，调用 `self.read_from_path` 方法处理该阶段数据，并使用生成器 `yield` 返回 `stage_name` 和处理后的数据。
                - 如果数据类型不是 `pathlib.Path`，则抛出 `TypeError`，提示数据类型错误。

        返回值:
            - 生成器对象：逐个返回 `(stage_name, processed_data)` 对，`processed_data` 是通过 `read_from_path` 方法处理的结果。

        使用场景:
            该方法用于从 `_data` 字典中读取并处理不同阶段的数据。每个阶段的数据通过 `read_from_path` 方法进行处理，并按阶段名称与处理结果成对返回。

        注意事项:
            - `_data` 字典中每个阶段的数据必须是 `pathlib.Path` 对象列表。
            - 如果数据类型不符合预期（即不是 `pathlib.Path`），将抛出 `TypeError` 异常。
            - `read_from_path` 方法需要被正确实现，以确保对路径数据的正确处理。
        """
        for stage_name, stage_data in self._data.items():
            if isinstance(stage_data[0], pathlib.Path):
                yield stage_name, self.read_from_path(stage_data, *args, **kwargs)
            else:
                raise TypeError(
                    f"data type error, should be str or typing.List[VideoCutRange]"
                )

    @staticmethod
    def read_from_path(data: list[pathlib.Path], *_, **__) -> typing.Generator:
        return (toolbox.imread(each.as_posix()) for each in data)

    def read_from_list(self, data: list[int], video_cap: cv2.VideoCapture = None, *_, **__):
        raise DeprecationWarning("this function already deprecated")

    def _classify_frame(self, frame: "VideoFrame", *args, **kwargs) -> str:
        raise NotImplementedError

    def _apply_hook(self, frame: "VideoFrame", *args, **kwargs) -> "VideoFrame":
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
        分类视频帧，并根据给定参数返回分类结果。

        本方法接受视频文件路径或视频对象，并遍历视频帧以进行分类。可通过valid_range指定只分类特定范围内的帧。
        可选地，可以增强性能通过跳过帧(step)和利用之前的分类结果(boost_mode)。

        Parameters:
            video (Union[str, VideoObject]): 要分类的视频路径或视频对象。
            valid_range (list[VideoCutRange], optional): 一个包含VideoCutRange对象的列表，这些对象定义了需要分类的帧的范围。
            step (int, optional): 分类时的帧步长，默认为1，表示逐帧处理。
            keep_data (bool, optional): 是否在返回的结果中保留原始帧数据。
            boost_mode (bool, optional): 是否启用增强模式，复用前一帧的结果来提高性能。需要valid_range来工作。
            *args, **kwargs: 传递给钩子函数和帧分类函数的额外参数。

        Returns:
            ClassifierResult: 包含分类结果的对象，每个结果为一个SingleClassifierResult对象，包括视频路径、帧ID、时间戳和分类结果。

        Raises:
            AssertionError: 如果启用了boost_mode但没有提供valid_range，将抛出异常。
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
    `BaseModelClassifier` 类简要说明：
        1.`BaseModelClassifier` 是一个基于 `BaseClassifier` 的抽象基类，专门用于模型分类器的实现。
        2.该类定义了一组必须在子类中实现的关键方法，这些方法涵盖了模型的保存、加载、清理、训练以及预测等操作。

    ### 主要功能：
        - **save_model**: 用于保存训练后的模型。
        - **load_model**: 用于从指定路径加载模型。
        - **clean_model**: 用于清理或重置模型资源。
        - **train**: 用于训练模型。
        - **predict**: 用于对单个图像路径进行预测，并返回分类结果。
        - **predict_with_object**: 用于对图像数据进行预测，并返回分类结果。
        - **read_from_list**: 该方法不支持从列表中读取数据，只支持从文件加载数据，因此在调用时会抛出 `ValueError`。

    ### 设计意图：
        该类通过抛出 `NotImplementedError` 强制要求所有子类必须实现这些方法，从而确保不同的模型分类器具备一致的接口和行为。
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
