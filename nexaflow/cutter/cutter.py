import os
import time
import typing
import numpy as np
from loguru import logger
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from nexaflow import toolbox
from nexaflow.cutter.cut_range import VideoCutRange
from nexaflow.cutter.cut_result import VideoCutResult
from nexaflow.video import VideoObject, VideoFrame
from nexaflow.hook import BaseHook


class Window(object):

    def __init__(self, video: "VideoObject", *args):
        self.video = video
        assert len(args) == 7, "需要7个参数"
        (self.step, self.block, self.window_size, self.window_coefficient,
         self.start, self.video_length, self.frame_total) = args
        self.end = self.start + self.window_size * self.step

    def load_data(self) -> typing.List[VideoFrame]:
        cur = self.start
        result = []
        video_operator = self.video.get_operator()
        while cur <= self.end:
            frame = video_operator.get_frame_by_id(cur)
            result.append(frame)
            cur += self.step

        if len(result) < 2:
            last = video_operator.get_frame_by_id(self.end)
            result.append(last)
        return result

    def shift(self) -> bool:
        self.start += self.step
        self.end += self.step
        if self.start >= self.video_length:
            return False

        if self.end >= self.video_length:
            self.end = self.video_length
        return True

    def float_merge(self, float_list: typing.List[float]) -> float:
        length = len(float_list)
        result = 0.0
        denominator = 0.0
        for i, each in enumerate(float_list):
            weight = pow(length - i, self.window_coefficient)
            denominator += weight
            result += each * weight
        final = result / denominator
        return final


class VideoCutter(object):

    def __init__(
        self,
        step: int = None,
        compress_rate: float = None,
        target_size: typing.Tuple[int, int] = None,
    ):

        self.step = step or 1
        self.compress_rate = compress_rate
        self.target_size = target_size

        self._hook_list: typing.List[BaseHook] = list()
        # compress_hook = CompressHook(
        #     overwrite=True, compress_rate=compress_rate, target_size=target_size
        # )
        # grey_hook = GreyHook(overwrite=True)
        # self.add_hook(compress_hook)
        # self.add_hook(grey_hook)

    def add_hook(self, new_hook: BaseHook):
        self._hook_list.append(new_hook)
        # logger.debug(f"add hook: {new_hook.__class__.__name__}")

    @staticmethod
    def pic_split(origin: np.ndarray, block: int) -> typing.List[np.ndarray]:
        result: typing.List[np.ndarray] = list()
        for each_block in np.array_split(origin, block, axis=0):
            sub_block = np.array_split(each_block, block, axis=1)
            result += sub_block
        return result

    def _apply_hook(self, frame: VideoFrame, *args, **kwargs) -> VideoFrame:
        for each_hook in self._hook_list:
            frame = each_hook.do(frame, *args, **kwargs)
        return frame

    @staticmethod
    def compare_frame_list(
        src: typing.List[np.ndarray], target: typing.List[np.ndarray]
    ) -> typing.List[float]:

        ssim = 1.0
        mse = 0.0
        psnr = 0.0

        for part_index, (each_start, each_end) in enumerate(zip(src, target)):
            part_ssim = toolbox.compare_ssim(each_start, each_end)
            if part_ssim < ssim:
                ssim = part_ssim

            part_mse = toolbox.calc_mse(each_start, each_end)
            if part_mse > mse:
                mse = part_mse

            part_psnr = toolbox.calc_psnr(each_start, each_end)
            if part_psnr > psnr:
                psnr = part_psnr
            # logger.debug(
            #     f"part {part_index}: ssim={part_ssim}; mse={part_mse}; psnr={part_psnr}"
            # )
        return [ssim, mse, psnr]

    @staticmethod
    def split_into_parts(value: int, parts: int) -> List[Tuple[int, int, int]]:
        division, remainder = value // parts, value % parts
        result, current_start = [], 1

        for i in range(parts):
            current_end = current_start + division - 1
            if i == parts - 1:  # 处理最后一部分，加上余数
                current_end += remainder
            result.append((current_start, current_end, current_end - current_start))

            if i < parts - 1:  # 不是最后一部分时，添加断开部分
                gap_start = current_end
                gap_end = current_end + 1
                result.append((gap_start, gap_end, gap_end - gap_start))
            current_start = current_end + 1

        return result

    def handler_frames(self, window: Window) -> typing.List[VideoCutRange]:
        range_list_part = []

        def technique():
            frame_list = window.load_data()
            frame_list = [self._apply_hook(each) for each in frame_list]

            ssim_list, mse_list, psnr_list = [], [], []

            cur_frame = frame_list[0]
            first_target_frame = frame_list[1]
            cur_frame_list = self.pic_split(cur_frame.data, window.block)
            for each in frame_list[1:]:
                each_frame_list = self.pic_split(each.data, window.block)
                ssim, mse, psnr = self.compare_frame_list(
                    cur_frame_list, each_frame_list
                )
                ssim_list.append(ssim)
                mse_list.append(mse)
                psnr_list.append(psnr)

            ssim = window.float_merge(ssim_list)
            mse = window.float_merge(mse_list)
            psnr = window.float_merge(psnr_list)

            range_list_part.append(
                VideoCutRange(
                    window.video,
                    start=cur_frame.frame_id, end=first_target_frame.frame_id,
                    ssim=[ssim], mse=[mse], psnr=[psnr],
                    start_time=cur_frame.timestamp, end_time=first_target_frame.timestamp,
                )
            )

        pbar = toolbox.show_progress(window.frame_total, 174, "Cutter")
        while True:
            technique()
            pbar.update(1)

            continue_flag = window.shift()
            if not continue_flag:
                pbar.close()
                break

        return range_list_part

    def _convert_video_into_range_list(
        self, video: VideoObject, block: int, window_size: int, window_coefficient: int
    ) -> typing.List[VideoCutRange]:

        step = self.step
        video_length = video.frame_count
        range_list: typing.List[VideoCutRange] = list()
        logger.info(f"总帧数: {video_length} 片段数: {video_length - 1} 分辨率: {video.frame_size}")

        window_list: List["Window"] = []
        for index, parts in enumerate(self.split_into_parts(video_length, 2)):
            start, end, size = parts
            logger.info(f"帧片段: {index + 1:02} Start: {start:03} End: {end:03} Length: {size:03}")
            window = Window(video, step, block, window_size, window_coefficient, start, end, size)
            window_list.append(window)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.handler_frames, w) for w in window_list]
            for future in futures:
                range_list.extend(future.result())

        return range_list

    def cut(
        self,
        video: typing.Union[str, VideoObject],
        block: int = None,
        window_size: int = None,
        window_coefficient: int = None,
        *_,
        **kwargs,
    ) -> VideoCutResult:

        if not block:
            block = 3
        if not window_size:
            window_size = 1
        if not window_coefficient:
            window_coefficient = 2

        start_time = time.time()
        if isinstance(video, str):
            video = VideoObject(video)

        logger.info(f"开始压缩视频: {os.path.basename(video.path)}")
        range_list = self._convert_video_into_range_list(
            video, block, window_size, window_coefficient
        )
        logger.info(f"视频压缩完成: {os.path.basename(video.path)}")
        logger.info(f"视频压缩耗时: {(time.time() - start_time):.2f}秒")

        return VideoCutResult(video, range_list, cut_kwargs=kwargs)


if __name__ == '__main__':
    pass
