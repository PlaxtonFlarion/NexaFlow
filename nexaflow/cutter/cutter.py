import numpy
import typing
from nexaflow import toolbox
from nexaflow.hook import BaseHook
from nexaflow.video import VideoObject, VideoFrame
from nexaflow.cutter.cut_range import VideoCutRange
from nexaflow.cutter.cut_result import VideoCutResult


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
        # compress_hook = FrameSizeHook(
        #     overwrite=True, compress_rate=compress_rate, target_size=target_size
        # )
        # grey_hook = GreyHook(overwrite=True)
        # self.add_hook(compress_hook)
        # self.add_hook(grey_hook)

    def add_hook(self, new_hook: "BaseHook"):
        self._hook_list.append(new_hook)
        # logger.debug(f"add hook: {new_hook.__class__.__name__}")

    @staticmethod
    def pic_split(origin: numpy.ndarray, block: int) -> list[numpy.ndarray]:
        """
        函数 `pic_split` 的作用是将一个输入的二维数组（例如图片数据）分割成多个小块。
        参数 `block` 指定了在每个维度（即高度和宽度）上将数组分割成多少块。
        这意味着整个数组会被分割成 `block * block` 个子块。
        @param origin: 一个 `numpy.ndarray`，代表要被分割的图像或任何二维数据。
        @param block: 一个整数，指定在每个轴（水平和垂直）上要分割成多少块。
        @return: 函数返回包含所有小块的列表，每个小块仍然是一个 `numpy.ndarray` 对象。
        """
        result: list[numpy.ndarray] = list()
        for each_block in numpy.array_split(origin, block, axis=0):
            sub_block = numpy.array_split(each_block, block, axis=1)
            result += sub_block
        return result

    def _apply_hook(self, frame: "VideoFrame", *args, **kwargs) -> "VideoFrame":
        for each_hook in self._hook_list:
            frame = each_hook.do(frame, *args, **kwargs)
        return frame

    @staticmethod
    def compare_frame_list(src: list[numpy.ndarray], target: list[numpy.ndarray]) -> list[float]:
        """
        关于如何比较两个 ndarray 列表并获取它们的 ssim/mse/psnr 的核心方法
        你可以重写这个方法来实现你自己的算法
        """
        # 找到最小 ssim 和最大 mse / psnr
        ssim, mse, psnr = 1.0, 0.0, 0.0

        for part_index, (each_start, each_end) in enumerate(zip(src, target)):
            part_ssim = toolbox.compare_ssim(each_start, each_end)
            if part_ssim < ssim:
                ssim = part_ssim

            # mse 非常敏感
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

    def magic_frame_range(
        self,
        video: "VideoObject",
        block: int,
        window_size: int,
        window_coefficient: int
    ) -> list["VideoCutRange"]:

        def slice_frame():
            range_list: typing.List["VideoCutRange"] = list()
            progress_bar = toolbox.show_progress(total=video_length, color=174)
            while True:
                frame_list = window.load_data()
                frame_list = [self._apply_hook(each) for each in frame_list]

                ssim_list, mse_list, psnr_list = [], [], []

                cur_frame = frame_list[0]
                first_target_frame = frame_list[1]
                cur_frame_list = self.pic_split(cur_frame.data, block)
                for each in frame_list[1:]:
                    each_frame_list = self.pic_split(each.data, block)
                    ssim, mse, psnr = self.compare_frame_list(
                        cur_frame_list, each_frame_list
                    )
                    ssim_list.append(ssim)
                    mse_list.append(mse)
                    psnr_list.append(psnr)

                ssim = float_merge(ssim_list)
                mse = float_merge(mse_list)
                psnr = float_merge(psnr_list)

                range_list.append(
                    VideoCutRange(
                        video,
                        start=cur_frame.frame_id, end=first_target_frame.frame_id,
                        ssim=[ssim], mse=[mse], psnr=[psnr],
                        start_time=cur_frame.timestamp, end_time=first_target_frame.timestamp,
                    )
                )
                progress_bar.update(1)

                continue_flag = window.shift()
                if not continue_flag:
                    progress_bar.close()
                    break

            return range_list

        def float_merge(float_list: list[float]) -> float:
            # 第一个，最大的
            length = len(float_list)
            result = 0.0
            denominator = 0.0
            for i, each in enumerate(float_list):
                weight = pow(length - i, window_coefficient)
                denominator += weight
                result += each * weight
                # logger.debug(f"calc: {each} x {weight}")
            final = result / denominator
            # logger.debug(f"calc final: {final} from {result} / {denominator}")
            return final

        class Window(object):

            def __init__(self):
                self.start = 1
                self.size = window_size
                self.end = self.start + window_size * step

            def load_data(self) -> typing.List[VideoFrame]:
                cur = self.start
                result = []
                video_operator = video.get_operator()
                while cur <= self.end:
                    frame = video_operator.get_frame_by_id(cur)
                    result.append(frame)
                    cur += step
                # 至少2个
                if len(result) < 2:
                    last = video_operator.get_frame_by_id(self.end)
                    result.append(last)
                return result

            def shift(self) -> bool:
                # logger.debug(f"window before: {self.start}, {self.end}")
                self.start += step
                self.end += step
                if self.start >= video_length:
                    # 超出范围
                    return False
                # 窗端
                if self.end >= video_length:
                    self.end = video_length
                # logger.debug(f"window after: {self.start}, {self.end}")
                return True

        video_length = video.frame_count
        step = self.step

        window = Window()

        return slice_frame()

    def cut(
        self,
        video: typing.Union[str, "VideoObject"],
        block: int = None,
        window_size: int = None,
        window_coefficient: int = None,
        *_,
        **kwargs,
    ) -> "VideoCutResult":

        video = VideoObject(video) if isinstance(video, str) else video

        block = block or 3
        window_size = window_size or 1
        window_coefficient = window_coefficient or 2

        # 如果视频包含 100 帧
        # 从1开始，列表长度是99，而不是100
        # [范围(1-2)、范围(2-3)、范围(3-4) ... 范围(99-100)]
        range_list = self.magic_frame_range(
            video, block, window_size, window_coefficient
        )

        return VideoCutResult(video, range_list, cut_kwargs=kwargs)


if __name__ == '__main__':
    pass
