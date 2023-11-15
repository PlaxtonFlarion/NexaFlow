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
            video: VideoObject,
            range_list: typing.List[VideoCutRange],
            cut_kwargs: typing.Dict = None,
    ):
        self.video = video
        self.range_list = range_list
        self.cut_kwargs = cut_kwargs or {}

    def get_target_range_by_id(self, frame_id: int) -> VideoCutRange:
        for each in self.range_list:
            if each.contain(frame_id):
                return each
        raise RuntimeError(f"frame {frame_id} not found in video")

    @staticmethod
    def _length_filter(
            range_list: typing.List[VideoCutRange], limit: int
    ) -> typing.List[VideoCutRange]:
        after = list()
        for each in range_list:
            if each.get_length() >= limit:
                after.append(each)
        return after

    def get_unstable_range(
            self, limit: int = None, range_threshold: float = None, **kwargs
    ) -> typing.List[VideoCutRange]:

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
    ) -> typing.Tuple[typing.List[VideoCutRange], typing.List[VideoCutRange]]:

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
            logger.info("稳定阶段开始 ...")
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
            logger.info("不稳定阶段开始 ...")

        # stable end
        if end_stable_range_start_id <= video_end_frame_id:
            # logger.debug("stable end")
            logger.info("稳定阶段结束 ...")
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
            logger.info("不稳定阶段结束 ...")

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

    def get_stable_range(
            self, limit: int = None, **kwargs
    ) -> typing.List[VideoCutRange]:

        return self.get_range(limit, **kwargs)[0]

    def get_range_dynamic(
            self,
            stable_num_limit: typing.List[int],
            threshold: float,
            step: float = 0.005,
            max_retry: int = 10,
            **kwargs,
    ) -> typing.Tuple[typing.List[VideoCutRange], typing.List[VideoCutRange]]:

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
            target_range: VideoCutRange,
            to_dir: str = None,
            compress_rate: float = None,
            is_vertical: bool = None,
            *_,
            **__,
    ) -> np.ndarray:

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
            range_list: typing.List[VideoCutRange],
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

        stage_list = list()
        for index, each_range in enumerate(range_list):
            picked = each_range.pick(frame_count, *args, **kwargs)
            picked_frames = each_range.get_frames(picked)
            logger.info(f"pick {picked} in range {each_range}")
            stage_list.append((str(index), picked_frames))

        if prune:
            stage_list = self._prune(prune, stage_list)

        if not to_dir:
            to_dir = toolbox.get_timestamp_str()
        logger.debug(f"try to make dirs: {to_dir}")
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
            stages: typing.List[typing.Tuple[str, typing.List[VideoFrame]]],
    ) -> typing.List[typing.Tuple[str, typing.List[VideoFrame]]]:
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
            pre_hooks: typing.List[BaseHook] = None,
            output_path: str = None,
            *args,
            **kwargs,
    ) -> "VideoCutResultDiff":

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
            range_list_1: typing.List[VideoCutRange],
            range_list_2: typing.List[VideoCutRange],
            *args,
            **kwargs,
    ) -> typing.Dict[int, typing.Dict[int, typing.List[float]]]:

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

    threshold: float = 0.7
    default_stage_id: int = -1
    default_score: float = -1.0

    def __init__(
            self, origin: typing.List[VideoCutRange], another: typing.List[VideoCutRange]
    ):
        self.origin = origin
        self.another = another
        self.data: typing.Optional[
            typing.Dict[int, typing.Dict[int, typing.List[float]]]
        ] = None

    def apply_diff(self, pre_hooks: typing.List[BaseHook] = None):
        self.data = VideoCutResult.range_diff(self.origin, self.another, pre_hooks)

    def most_common(self, stage_id: int) -> (int, float):
        assert stage_id in self.data
        ret_k, ret_v = self.default_stage_id, self.default_score
        for k, v in self.data[stage_id].items():
            cur = max(v)
            if cur > ret_v:
                ret_k = k
                ret_v = cur
        return ret_k, ret_v

    def is_stage_lost(self, stage_id: int) -> bool:
        _, v = self.most_common(stage_id)
        return v < self.threshold

    def any_stage_lost(self) -> bool:
        return all((self.is_stage_lost(each) for each in self.data.keys()))

    def stage_shift(self) -> typing.List[int]:
        ret = list()
        for k in self.data.keys():
            new_k, score = self.most_common(k)
            if score > self.threshold:
                ret.append(new_k)
        return ret

    def stage_diff(self):
        return difflib.Differ().compare(
            [str(each) for each in self.stage_shift()],
            [str(each) for each in range(len(self.another))],
        )


if __name__ == '__main__':
    pass
