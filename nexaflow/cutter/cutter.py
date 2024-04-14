import os
import time
import typing
import numpy as np
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from nexaflow import toolbox
from nexaflow.hook import BaseHook
from nexaflow.video import VideoObject, VideoFrame
from nexaflow.cutter.cut_range import VideoCutRange
from nexaflow.cutter.cut_result import VideoCutResult


class Window(object):

    def __init__(self, video: "VideoObject", *args):
        self.video = video
        self.step, self.block, self.window_size, self.window_coefficient, *_ = args
        *_, self.start, self.video_length, self.frame_total = args
        self.end = self.start + self.window_size * self.step

    def load_data(self) -> list["VideoFrame"]:
        current = self.start
        result = []
        video_operator = self.video.get_operator()
        while current <= self.end:
            frame = video_operator.get_frame_by_id(current)
            result.append(frame)
            current += self.step

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

    def float_merge(self, float_list: list[float]) -> float:
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
        target_size: tuple[int, int] = None,
    ):

        self.step = step or 1
        self.compress_rate = compress_rate
        self.target_size = target_size

        self._hook_list: list["BaseHook"] = list()
        # compress_hook = CompressHook(
        #     overwrite=True, compress_rate=compress_rate, target_size=target_size
        # )
        # grey_hook = GreyHook(overwrite=True)
        # self.add_hook(compress_hook)
        # self.add_hook(grey_hook)

    def add_hook(self, new_hook: "BaseHook"):
        self._hook_list.append(new_hook)
        # logger.debug(f"add hook: {new_hook.__class__.__name__}")

    @staticmethod
    def pic_split(origin: np.ndarray, block: int) -> list[np.ndarray]:
        result: list[np.ndarray] = list()
        for each_block in np.array_split(origin, block, axis=0):
            sub_block = np.array_split(each_block, block, axis=1)
            result += sub_block
        return result

    def _apply_hook(self, frame: "VideoFrame", *args, **kwargs) -> "VideoFrame":
        for each_hook in self._hook_list:
            frame = each_hook.do(frame, *args, **kwargs)
        return frame

    @staticmethod
    def compare_frame_list(
        src: list[np.ndarray], target: list[np.ndarray]
    ) -> list[float]:

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
    def split_range(value: int, parts: int) -> list[tuple[int, int, int]]:
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

    def window_slice(self, window: "Window") -> list["VideoCutRange"]:

        def cutting():
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

        range_list_part = []
        pbar = toolbox.show_progress(window.frame_total, 174, "Cutter")
        while True:
            cutting()
            pbar.update(1)

            continue_flag = window.shift()
            if not continue_flag:
                pbar.close()
                break

        return range_list_part

    def magic_frame_range(
        self, video: "VideoObject", block: int, window_size: int, window_coefficient: int
    ) -> list[VideoCutRange]:

        video_length = video.frame_count
        logger.info(f"总帧数: {video_length} 片段数: {video_length - 1} 分辨率: {video.frame_size}")

        window_list = []
        for index, parts in enumerate(self.split_range(video_length, 1 if video_length < 500 else 2)):
            start, end, size = parts
            logger.info(f"帧片段: {index + 1:02} Start: {start:03} End: {end:03} Length: {size:03}")
            window = Window(
                video, self.step, block, window_size, window_coefficient, start, end, size
            )
            window_list.append(window)

        with ThreadPoolExecutor() as exe:
            futures = [
                exe.submit(self.window_slice, window) for window in window_list
            ]
            range_list = [
                part for future in futures for part in future.result()
            ]

        return range_list

    def cut(
        self,
        video: typing.Union[str, "VideoObject"],
        block: int = None, window_size: int = None, window_coefficient: int = None,
        *_, **kwargs,
    ) -> "VideoCutResult":

        block = block or 3
        window_size = window_size or 1
        window_coefficient = window_coefficient or 2

        start_time = time.time()
        video = VideoObject(video) if isinstance(video, str) else video

        logger.info(f"开始压缩视频: {os.path.basename(video.path)}")
        # 如果视频包含 100 帧
        # 从1开始，列表长度是99，而不是100
        # [范围(1-2)、范围(2-3)、范围(3-4) ... 范围(99-100)]
        range_list = self.magic_frame_range(
            video, block, window_size, window_coefficient
        )
        logger.info(f"视频压缩完成: {os.path.basename(video.path)}")
        logger.info(f"视频压缩耗时: {(time.time() - start_time):.2f}秒")

        return VideoCutResult(video, range_list, cut_kwargs=kwargs)


if __name__ == '__main__':
    pass
