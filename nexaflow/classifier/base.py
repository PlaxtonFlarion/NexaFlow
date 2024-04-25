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

    def to_video_frame(self, *args, **kwargs) -> VideoFrame:
        if self.data is not None:
            return VideoFrame(self.frame_id, self.timestamp, self.data)

        with toolbox.video_capture(self.video_path) as cap:
            frame = toolbox.get_frame(cap, self.frame_id)
            compressed = toolbox.compress_frame(frame, *args, **kwargs)
        return VideoFrame(self.frame_id, self.timestamp, compressed)

    def get_data(self) -> np.ndarray:
        return self.to_video_frame().data

    def is_stable(self) -> bool:
        return self.stage not in (
            const.UNSTABLE_FLAG,
            const.IGNORE_FLAG,
            const.UNKNOWN_STAGE_FLAG,
        )

    def contain_image(
        self, *, image_path: str = None, image_object: np.ndarray = None, **kwargs
    ) -> typing.Dict[str, typing.Any]:
        return self.to_video_frame().contain_image(
            image_path=image_path, image_object=image_object, **kwargs
        )

    def to_dict(self) -> typing.Dict:
        return self.__dict__

    def __str__(self):
        return f"<ClassifierResult stage={self.stage} frame_id={self.frame_id} timestamp={self.timestamp}>"

    __repr__ = __str__


class DiffResult(object):

    def __init__(
        self, origin_data: "ClassifierResult", another_data: "ClassifierResult"
    ):
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

    def __init__(self, data: typing.List[SingleClassifierResult]):
        self.video_path: str = data[0].video_path
        self.data: typing.List[SingleClassifierResult] = data

    def get_timestamp_list(self) -> typing.List[float]:
        return [each.timestamp for each in self.data]

    def get_stage_list(self) -> typing.List[str]:
        return [each.stage for each in self.data]

    def get_length(self) -> int:
        return len(self.data)

    def get_offset(self) -> float:
        return self.data[1].timestamp - self.data[0].timestamp

    def get_ordered_stage_set(self) -> typing.List[str]:
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

    def to_dict(
        self,
    ) -> typing.Dict[str, typing.List[typing.List["SingleClassifierResult"]]]:
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

    def first(self, stage_name: str) -> SingleClassifierResult:
        for each in self.data:
            if each.stage == stage_name:
                # logger.debug(f"first frame of {stage_name}: {each}")
                return each
        logger.warning(f"no stage named {stage_name} found")

    def last(self, stage_name: str) -> SingleClassifierResult:
        for each in self.data[::-1]:
            if each.stage == stage_name:
                # logger.debug(f"last frame of {stage_name}: {each}")
                return each
        logger.warning(f"no stage named {stage_name} found")

    def get_stage_range(self) -> typing.List[typing.List[SingleClassifierResult]]:
        result: typing.List[typing.List[SingleClassifierResult]] = []

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

    def get_specific_stage_range(
        self, stage_name: str
    ) -> typing.List[typing.List[SingleClassifierResult]]:
        ret = list()
        for each_range in self.get_stage_range():
            cur = each_range[0]
            if cur.stage == stage_name:
                ret.append(each_range)
        return ret

    def get_not_stable_stage_range(
        self,
    ) -> typing.List[typing.List[SingleClassifierResult]]:
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

    def get_important_frame_list(self) -> typing.List[SingleClassifierResult]:
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

    def calc_changing_cost(
        self,
    ) -> typing.Dict[str, typing.Tuple[SingleClassifierResult, SingleClassifierResult]]:

        cost_dict: typing.Dict[str, typing.Tuple[SingleClassifierResult, SingleClassifierResult]] = {}
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

    def is_order_correct(self, should_be: typing.List[str]) -> bool:
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
        target_size: typing.Tuple[int, int] = None,
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

        self._data: typing.Dict[str, typing.Union[typing.List[pathlib.Path]]] = dict()

        self._hook_list: typing.List[BaseHook] = list()
        # compress_hook = FrameSizeHook(
        #     overwrite=True, compress_rate=compress_rate, target_size=target_size
        # )
        # grey_hook = GreyHook(overwrite=True)
        # self.add_hook(compress_hook)
        # self.add_hook(grey_hook)

    def add_hook(self, new_hook: BaseHook):
        self._hook_list.append(new_hook)
        # logger.debug(f"add hook: {new_hook.__class__.__name__}")

    def load(
        self, data: typing.Union[str, typing.List[VideoCutRange], None], *args, **kwargs
    ):

        if isinstance(data, str):
            return self.load_from_dir(data, *args, **kwargs)
        if isinstance(data, list):
            return self.load_from_list(data, *args, **kwargs)
        raise TypeError(f"data type error, should be str or typing.List[VideoCutRange]")

    def load_from_list(
        self, data: typing.List[VideoCutRange], frame_count: int = None, *_, **__
    ):
        for stage_name, stage_data in enumerate(data):
            target_frame_list = stage_data.pick(frame_count)
            self._data[str(stage_name)] = target_frame_list

    def load_from_dir(self, dir_path: str, *_, **__):
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

    def read(self, *args, **kwargs):
        for stage_name, stage_data in self._data.items():
            if isinstance(stage_data[0], pathlib.Path):
                yield stage_name, self.read_from_path(stage_data, *args, **kwargs)
            else:
                raise TypeError(
                    f"data type error, should be str or typing.List[VideoCutRange]"
                )

    @staticmethod
    def read_from_path(data: typing.List[pathlib.Path], *_, **__):
        return (toolbox.imread(each.as_posix()) for each in data)

    def read_from_list(
        self, data: typing.List[int], video_cap: cv2.VideoCapture = None, *_, **__
    ):
        raise DeprecationWarning("this function already deprecated")

    def _classify_frame(self, frame: VideoFrame, *args, **kwargs) -> str:
        raise NotImplementedError

    def _apply_hook(self, frame: VideoFrame, *args, **kwargs) -> VideoFrame:
        for each_hook in self._hook_list:
            frame = each_hook.do(frame, *args, **kwargs)
        return frame

    def classify(
        self,
        video: typing.Union[str, "VideoObject"],
        valid_range: typing.List["VideoCutRange"] = None,
        step: int = None,
        keep_data: bool = None,
        boost_mode: bool = None,
        *args,
        **kwargs,
    ) -> "ClassifierResult":

        # logger.debug(f"classify with {self.__class__.__name__}")
        step = step or 1
        boost_mode = boost_mode or True

        assert (boost_mode and valid_range) or (
            not (boost_mode or valid_range)
        ), "boost_mode required valid_range"

        final_result: typing.List[SingleClassifierResult] = list()
        if isinstance(video, str):
            video = VideoObject(video)

        operator = video.get_operator()
        frame = operator.get_frame_by_id(1)

        prev_result: typing.Optional[str] = None
        pbar = toolbox.show_progress(total=video.frame_count, color=38)
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
            pbar.update(1)
        pbar.close()

        return ClassifierResult(final_result)


class BaseModelClassifier(BaseClassifier):

    def save_model(self, model_path: str, overwrite: bool = None):
        raise NotImplementedError

    def load_model(self, model_path: str, overwrite: bool = None):
        raise NotImplementedError

    def clean_model(self):
        raise NotImplementedError

    def train(self, data_path: str = None, *_, **__):
        raise NotImplementedError

    def predict(self, pic_path: str, *_, **__) -> str:
        raise NotImplementedError

    def predict_with_object(self, frame: np.ndarray) -> str:
        raise NotImplementedError

    def read_from_list(
        self, data: typing.List[int], video_cap: cv2.VideoCapture = None, *_, **__
    ):
        raise ValueError("model-like classifier only support loading data from files")


if __name__ == '__main__':
    pass
