#
#    ____      _   _
#   / ___|   _| |_| |_ ___ _ __
#  | |  | | | | __| __/ _ \ '__|
#  | |__| |_| | |_| ||  __/ |
#   \____\__,_|\__|\__\___|_|
#

import numpy
import typing
from nexaflow import toolbox
from nexaflow.hook import BaseHook
from nexaflow.video import VideoObject, VideoFrame
from nexaflow.cutter.cut_range import VideoCutRange
from nexaflow.cutter.cut_result import VideoCutResult


class VideoCutter(object):
    """
    一个用于视频帧处理的工具类，支持视频帧的压缩、灰度处理及其他自定义钩子的添加。
    """

    def __init__(
        self,
        step: int = None,
        compress_rate: float = None,
        target_size: tuple[int, int] = None,
    ):
        """
        初始化方法。

        参数:
            - `step (int)`: 每次处理视频时的帧步长。默认为1，即逐帧处理。
            - `compress_rate (float)`: 视频帧的压缩比例。用于减少视频帧的尺寸以提高处理效率。默认不压缩。
            - `target_size (tuple[int, int])`: 目标视频帧的尺寸。用于将视频帧调整到指定大小。

        说明:
            该构造方法初始化了 `VideoCutter` 对象的基本属性，包括帧步长、压缩率和目标尺寸。还初始化了一个钩子列表，用于存储添加的帧处理钩子。
        """

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
        """
        添加钩子对象。

        参数:
            - `new_hook (BaseHook)`: 一个实现了 `BaseHook` 接口的钩子对象，用于在视频帧处理中执行特定操作。

        说明:
            该方法将新的钩子对象添加到 `_hook_list` 列表中，以便在视频帧处理时调用这些钩子。
        """
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
        """
        应用钩子处理方法到视频帧上。

        参数:
            - frame (`VideoFrame`): 需要处理的视频帧对象。
            - *args: 传递给钩子处理方法的可变参数。
            - **kwargs: 传递给钩子处理方法的关键字参数。

        返回:
            - `VideoFrame`: 经过所有钩子处理后返回的视频帧对象。

        说明:
            该方法遍历 `self._hook_list` 中的每个钩子对象，并依次将 `frame` 传递给每个钩子的 `do` 方法进行处理。
            处理后的 `frame` 会被更新，并作为输入传递给下一个钩子处理，最后返回最终处理完毕的视频帧。

        异常处理:
            - 该方法假设 `self._hook_list` 中的每个钩子对象都实现了 `do` 方法，并且该方法接受一个 `VideoFrame` 对象及其他可选参数。
            - 如果钩子对象未正确实现 `do` 方法，可能会引发 `AttributeError` 或 `TypeError` 异常。
        """
        for each_hook in self._hook_list:
            frame = each_hook.do(frame, *args, **kwargs)
        return frame

    @staticmethod
    def compare_frame_list(src: list[numpy.ndarray], target: list[numpy.ndarray]) -> list[float]:
        """
        比较两个 ndarray 列表并获取它们的 SSIM、MSE 和 PSNR 值的核心方法。

        参数:
            - src (list[numpy.ndarray]): 源帧列表。
            - target (list[numpy.ndarray]): 目标帧列表。

        返回:
            - list[float]: 包含 SSIM、MSE 和 PSNR 值的列表。

        方法:
            1. 初始化 SSIM、MSE 和 PSNR 的默认值。
            2. 遍历源帧和目标帧的对应对。
            3. 计算每对帧的 SSIM，并更新最小 SSIM 值。
            4. 计算每对帧的 MSE，并更新最大 MSE 值。
            5. 计算每对帧的 PSNR，并更新最大 PSNR 值。
            6. 返回包含最小 SSIM、最大 MSE 和最大 PSNR 的列表。

        注意:
            - 你可以重写这个方法来实现你自己的算法。
            - SSIM 表示结构相似性指数，数值范围在 0 到 1 之间，越接近 1 说明两张图像越相似。
            - MSE 表示均方误差，数值越小说明两张图像越相似。
            - PSNR 表示峰值信噪比，数值越大说明两张图像越相似。
        """

        # 初始化 SSIM 为最大值，MSE 和 PSNR 为最小值
        ssim, mse, psnr = 1.0, 0.0, 0.0

        # 遍历源帧和目标帧的对应对
        for part_index, (each_start, each_end) in enumerate(zip(src, target)):
            # 计算每对帧的 SSIM
            part_ssim = toolbox.compare_ssim(each_start, each_end)
            if part_ssim < ssim:
                ssim = part_ssim

            # 计算每对帧的 MSE，MSE 非常敏感
            part_mse = toolbox.calc_mse(each_start, each_end)
            if part_mse > mse:
                mse = part_mse

            # 计算每对帧的 PSNR
            part_psnr = toolbox.calc_psnr(each_start, each_end)
            if part_psnr > psnr:
                psnr = part_psnr

            # 调试日志
            # logger.debug(
            #     f"part {part_index}: ssim={part_ssim}; mse={part_mse}; psnr={part_psnr}"
            # )

        # 返回包含最小 SSIM、最大 MSE 和最大 PSNR 的列表
        return [ssim, mse, psnr]

    @staticmethod
    def split_range(value: int, parts: int) -> list[tuple[int, int, int]]:
        """
        将一个范围分割成多个部分，并在每个部分之间插入一个断开区间。

        参数:
            - value (`int`): 需要分割的总范围长度。
            - parts (`int`): 将范围分割成的部分数量。

        返回:
            - `list[tuple[int, int, int]]`: 返回一个列表，每个元素是一个三元组，表示分割后的部分或断开区间。
              每个三元组包含三个整数，分别表示该区间的起始位置、结束位置及区间长度。

        说明:
            该方法首先计算分割后的每个部分的长度 `division` 及余数 `remainder`。然后，根据指定的 `parts` 数量，将范围均匀分割，并在每个部分之间插入一个长度为1的断开区间。最后返回包含所有部分及断开区间的列表。

        例子:
            假设 `value=10` 且 `parts=3`，则返回的列表可能包含以下区间:
            ```
            [(1, 3, 2), (3, 4, 1), (4, 6, 2), (6, 7, 1), (7, 10, 3)]
            ```

        异常处理:
            - `ZeroDivisionError`: 如果 `parts` 为 0，会导致除零异常。在调用该方法时应确保 `parts` 大于 0。
            - 该方法假设 `value` 和 `parts` 均为正整数，否则可能导致不可预期的结果。

        注意:
            - 如果 `value` 不能被 `parts` 整除，则最后一个部分将包含余数，以确保所有部分的总长度等于 `value`。
        """

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
        """
        根据视频对象生成视频帧范围的核心方法，使用滑动窗口技术对帧进行分割和分析。

        参数:
            - video (VideoObject): 视频对象，包含视频帧数据。
            - block (int): 用于图像分割的块大小。
            - window_size (int): 滑动窗口的大小，以帧为单位。
            - window_coefficient (int): 用于权重计算的窗口系数。

        返回:
            - list[VideoCutRange]: 生成的视频帧范围列表。
        """

        def slice_frame():
            """
            使用滑动窗口对视频帧进行分割，并计算每个窗口的SSIM、MSE和PSNR值。

            返回:
                - list[VideoCutRange]: 生成的视频帧范围列表。
            """
            range_list: list["VideoCutRange"] = list()
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
            """
            根据指定的权重系数，对浮点数列表进行加权平均计算。

            参数:
                - float_list (list[float]): 包含浮点数的列表，每个浮点数代表一个计算结果。

            返回:
                - float: 加权平均后的浮点数结果。

            具体流程:
                1. 初始化变量 `result` 和 `denominator` 为0，用于存储计算结果和权重系数的总和。
                2. 遍历 `float_list`，对每个浮点数 `each` 进行处理：
                    - 根据窗口的长度和当前索引计算权重 `weight`，权重是基于窗口系数的幂次方。
                    - 将权重和浮点数的乘积累加到 `result`。
                    - 将权重累加到 `denominator`。
                3. 使用 `result` 除以 `denominator` 计算加权平均值。
                4. 返回计算的加权平均值。

            应用场景:
                - 在视频处理和分析过程中，针对每个视频帧或帧段计算一系列指标（如SSIM、MSE、PSNR）。
                - 使用滑动窗口技术和权重系数，对这些指标进行加权平均，以便更好地反映整个视频段的质量变化。
            """

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
            """
            窗口类用于在视频帧中创建一个滑动窗口，用于提取和处理视频帧的子集。

            属性:
                - start (int): 当前窗口的起始帧ID，初始值为1。
                - size (int): 窗口的大小（包含的帧数），初始值为给定的 `window_size`。
                - end (int): 当前窗口的结束帧ID，初始值为起始帧加上窗口大小乘以步长 `step`。

            工作流程:
                1. 初始化窗口的起始帧 `start` 为1，窗口大小 `size` 为给定的 `window_size`，结束帧 `end` 为起始帧加上窗口大小乘以步长。
                2. 在 `load_data` 方法中，从起始帧 `start` 开始逐帧加载视频帧数据，直到结束帧 `end`。如果帧数不足，则补充到至少两个帧。
                3. 在 `shift` 方法中，将窗口的起始帧和结束帧向前移动一个步长 `step`。如果移动后起始帧超过视频总帧数，返回 `False` 表示窗口无法继续移动；否则，更新窗口位置并返回 `True` 表示窗口可以继续移动。

            应用场景:
                - 在视频分析和处理过程中，使用滑动窗口技术对视频进行分段处理。
                - 通过窗口类，可以方便地在视频帧之间滑动，提取特定范围的帧进行进一步处理，如计算质量指标、应用图像处理算法等。
            """

            def __init__(self):
                self.start = 1
                self.size = window_size
                self.end = self.start + window_size * step

            def load_data(self) -> list["VideoFrame"]:
                """
                从当前窗口位置加载视频帧数据。

                返回:
                    - list[VideoFrame]: 窗口内视频帧的列表。

                工作流程:
                    1. 从当前 `start` 开始逐帧加载视频帧数据，直到 `end`。
                    2. 如果帧数不足 `size`，则补充到至少两个帧。
                """
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
                """
                将窗口的位置向前移动一个步长。

                返回:
                    - bool: 窗口是否可以继续移动（未超出视频帧的范围）。

                工作流程:
                    1. 将窗口的起始帧 `start` 和结束帧 `end` 向前移动一个步长 `step`。
                    2. 如果移动后起始帧超过视频总帧数 `video_length`，返回 `False` 表示窗口无法继续移动；否则，更新窗口位置并返回 `True` 表示窗口可以继续移动。
                """

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

        # 获取视频总帧数
        video_length = video.frame_count

        """
        ### step 参数具体作用和使用场景
        
        #### 作用
        - `step` 参数用于控制窗口在视频帧序列上的移动步长。
        - 在每次调用 `shift()` 方法时，窗口的起始和结束位置都会根据 `step` 的值进行调整。
        - `step` 决定了每次窗口移动时跳过的帧数。例如，`step=1` 时，窗口每次移动一帧；`step=2` 时，窗口每次移动两帧。
        
        #### 使用场景
        - **细粒度分析**: 当需要对视频帧进行精细分析时，可以设置较小的 `step` 值（例如 `step=1` 或 `step=2`），确保每次窗口移动时覆盖更多的帧，提供更高的分析精度。
        - **快速处理**: 在需要快速处理视频，且对精度要求不高的情况下，可以设置较大的 `step` 值（例如 `step=5` 或 `step=10`），这样窗口每次移动时跳过更多帧，从而提高处理效率。
        - **数据平滑**: 在进行视频帧数据的平滑处理时，通过调整 `step` 值，可以控制窗口的重叠程度，从而平衡处理速度和数据平滑效果。较小的 `step` 值可以增加帧间重叠，提供更平滑的数据结果。
        
        通过调整 `step` 参数，可以灵活控制窗口的移动步长，以适应不同的处理需求和应用场景。
        """
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
        """
        对视频进行分段处理，生成视频的剪辑结果。

        参数:
            - video (typing.Union[str, "VideoObject"]): 视频对象或视频文件路径。
            - block (int, 可选): 视频帧划分的块大小。默认值为3。
            - window_size (int, 可选): 滑动窗口的大小。默认值为1。
            - window_coefficient (int, 可选): 窗口系数，用于加权平均计算。默认值为2。
            - *_: 忽略位置参数。
            - **kwargs: 其他关键字参数。

        返回:
            - VideoCutResult: 视频剪辑结果对象，包含剪辑的范围列表和相关参数。

        工作流程:
            1. 如果 `video` 是字符串类型，则将其转换为 `VideoObject` 对象。
            2. 如果未提供 `block`、`window_size` 或 `window_coefficient` 参数，则使用默认值。
            3. 调用 `magic_frame_range` 方法，根据指定的块大小、窗口大小和窗口系数，对视频进行帧的分段处理，生成帧的范围列表 `range_list`。
            4. 返回 `VideoCutResult` 对象，包含视频对象、帧的范围列表和剪辑参数 `kwargs`。

        说明:
            - `block` 参数用于定义视频帧划分的块大小，默认值为3。
            - `window_size` 参数用于定义滑动窗口的大小，默认值为1。
            - `window_coefficient` 参数用于定义窗口系数，用于加权平均计算，默认值为2。
            - `magic_frame_range` 方法根据指定的参数，对视频帧进行分段处理，生成帧的范围列表。
            - `VideoCutResult` 对象包含视频对象、帧的范围列表和剪辑参数，表示视频的剪辑结果。

        block 参数:
            这个参数用于定义在视频帧分析时分割图像的块大小。
            图像块是一种将图像分割成小块的方法，每个块独立处理。这种方法可以有效减少计算复杂度，同时保持对图像局部特征的敏感度。
            块大小越大，每个块包含的像素越多，处理时的计算量越大；块大小越小，每个块包含的像素越少，计算量减少，但对细节的敏感度增加。

        应用场景:
            在视频分析中，使用块来处理帧有助于发现局部变化和细节。
            例如，在运动检测或场景切换检测中，小块可以更精确地捕捉到图像中的细微变化。选择合适的块大小可以平衡计算性能和检测精度。

        示例:
            如果 block 为 3，则每个图像将被分割成 3x3 的小块进行处理。这意味着每个图像将被分割成 9 个小块，每个小块独立进行相似度、均方误差和峰值信噪比的计算。

        window_size 参数:
            这个参数定义了滑动窗口的大小。滑动窗口用于从视频帧序列中提取一组连续帧进行处理。
            窗口大小越大，每次处理的帧数越多，可以捕捉到更长时间范围内的变化；窗口大小越小，每次处理的帧数越少，更关注局部的短时变化。

        应用场景:
            使用滑动窗口可以在视频中找到连续帧之间的变化，适用于场景切换检测、运动检测等场景。窗口大小的选择取决于具体应用场景和对时间连续性的要求。例如，在场景切换检测中，较大的窗口可以更好地捕捉到较长时间内的渐变；在运动检测中，较小的窗口可以更敏感地捕捉到细微的变化。

        示例:
            如果 window_size 为 5，则每次从视频中提取 5 帧进行处理。这意味着每次处理的帧组包含 5 个连续帧，计算这些帧之间的相似度、均方误差和峰值信噪比。

        window_coefficient 参数:
            这个参数用于在加权平均计算中确定每个值的权重。权重系数越大，窗口内较新帧的权重越高。
            这意味着在计算相似度、均方误差和峰值信噪比时，较新帧对结果的影响更大。具体地，权重按窗口内帧的倒序指数增加。

        应用场景:
            当分析视频变化时，较新帧可能比较旧帧更为重要。例如，在检测视频中的突然变化或剪辑时，较新帧的变化可能更显著。
            通过调整 window_coefficient，可以更灵活地控制每个帧对最终计算结果的贡献。

        示例:
            如果 window_size 为 5，window_coefficient 为 2，则窗口内的权重依次为 1, 4, 9, 16, 25。权重最高的帧对结果的影响最大。
        """

        video = VideoObject(video) if isinstance(video, str) else video

        block = block or 3
        window_size = window_size or 1
        window_coefficient = window_coefficient or 2

        """
        如果视频包含 100 帧
        从1开始，列表长度是99，而不是100
        [范围(1-2)、范围(2-3)、范围(3-4) ... 范围(99-100)]
        """
        range_list = self.magic_frame_range(
            video, block, window_size, window_coefficient
        )

        return VideoCutResult(video, range_list, cut_kwargs=kwargs)


if __name__ == '__main__':
    pass
