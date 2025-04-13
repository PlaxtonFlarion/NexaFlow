#
#    ____      _   _
#   / ___|   _| |_| |_ ___ _ __
#  | |  | | | | __| __/ _ \ '__|
#  | |__| |_| | |_| ||  __/ |
#   \____\__,_|\__|\__\___|_|
#

"""
版权所有 (c) 2024  Framix(画帧秀)
此文件受 Framix(画帧秀) 许可证的保护。您可以在 LICENSE.md 文件中查看详细的许可条款。

Copyright (c) 2024  Framix(画帧秀)
This file is licensed under the Framix(画帧秀) License. See the LICENSE.md file for more details.

# Copyright (c) 2024  Framix(画帧秀)
# このファイルは Framix(画帧秀) ライセンスの下でライセンスされています。詳細は LICENSE.md ファイルを参照してください。
"""

import numpy
import typing
from nexaflow import toolbox
from nexaflow.hook import BaseHook
from nexaflow.video import (
    VideoObject, VideoFrame
)
from nexaflow.cutter.cut_range import VideoCutRange
from nexaflow.cutter.cut_result import VideoCutResult


class VideoCutter(object):
    """
    VideoCutter 类用于对视频帧进行滑动窗口分析与智能剪辑。

    该类提供了基于 SSIM（结构相似性）、MSE（均方误差）和 PSNR（峰值信噪比）指标的帧间相似性评估方法，
    支持滑动窗口、图像块分割、帧剪辑和剪辑结果生成，适用于视频中的动态检测、阶段划分、训练样本提取等应用场景。

    本类核心功能包括帧分块处理、滑动窗口遍历、帧相似度计算、剪辑范围标记与保存。

    Notes
    -----
    - 默认以逐帧方式遍历视频帧序列，也可通过 step 参数设置跳帧间隔。
    - 支持通过 add_hook 方法添加帧级预处理逻辑，常用于模型推理或对比增强。
    - 核心分析逻辑集中于 magic_frame_range 方法，利用滑动窗口计算帧间差异值。
    - 适用于自动剪辑、稳定段检测、视频事件抽取等智能分析任务。
    """

    def __init__(
        self,
        step: int = None,
        compress_rate: float = None,
        target_size: tuple[int, int] = None,
    ):
        """
        初始化 VideoCutter 实例。

        该方法用于配置视频剪辑分析器的基础参数，包括滑动窗口步长、帧压缩比例、目标尺寸等。
        同时初始化帧处理钩子列表，用于后续预处理操作。

        Parameters
        ----------
        step : int, optional
            滑动窗口的步长，用于控制每次窗口滑动所跨越的帧数。默认值为 1，表示逐帧处理。

        compress_rate : float, optional
            图像压缩比例（0-1），用于在处理帧数据前降低分辨率，提高计算效率。若指定了 target_size，将被覆盖。

        target_size : tuple[int, int], optional
            压缩图像时的目标尺寸，格式为 (width, height)。优先生效于 compress_rate。

        Notes
        -----
        - 若未显式提供 `step`，则默认设为 1。
        - 若未设置压缩参数，系统不会自动缩放图像尺寸。
        - 可通过 add_hook 方法手动添加帧处理流程（如灰度化、裁剪等）。
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

    def add_hook(self, new_hook: "BaseHook") -> None:
        """
        向 VideoCutter 实例添加帧处理钩子。

        该方法将自定义的帧处理钩子添加到内部钩子列表中，钩子将在每帧处理前依次执行，
        可用于实现帧的裁剪、灰度转换、特征提取等预处理操作。

        Parameters
        ----------
        new_hook : BaseHook
            实现 `do(frame)` 方法的钩子对象，需继承自 `BaseHook` 类。

        Notes
        -----
        - 所有添加的钩子会在 `cut()` 或其他处理过程中依次作用于每一帧。
        - 添加顺序即为钩子执行顺序。
        - 常用钩子包括 `FrameSizeHook`（缩放图像）、`GreyHook`（灰度化图像）等。
        """
        self._hook_list.append(new_hook)
        # logger.debug(f"add hook: {new_hook.__class__.__name__}")

    @staticmethod
    def pic_split(origin: "numpy.ndarray", block: int) -> list["numpy.ndarray"]:
        """
        将图像均匀切分为指定数量的块。

        该方法将输入图像分别按高度和宽度方向平均分成 `block` × `block` 个子块，
        每个子块为一个独立的 `ndarray` 对象，适用于局部图像分析、块级相似度计算等场景。

        Parameters
        ----------
        origin : numpy.ndarray
            原始图像数组，通常为 OpenCV 加载的 BGR 图像或灰度图。

        block : int
            图像切分块的数量，将图像在两个维度上各分为 `block` 块，共 `block ** 2` 个子块。

        Returns
        -------
        list of numpy.ndarray
            切分后得到的所有图像子块列表，顺序按行优先（从上到下，从左到右）。

        Notes
        -----
        - 如果图像尺寸不能被 `block` 整除，OpenCV 的 `array_split` 将自动进行补齐分块。
        - 子块可用于后续局部区域的结构相似度（SSIM）、MSE、PSNR 等指标计算。
        - 对于较大图像和细粒度分析，推荐设置较大的 block 数量（例如 5 或 6）。
        """
        result: list["numpy.ndarray"] = list()
        for each_block in numpy.array_split(origin, block, axis=0):
            sub_block = numpy.array_split(each_block, block, axis=1)
            result += sub_block
        return result

    def _apply_hook(self, frame: "VideoFrame", *args, **kwargs) -> "VideoFrame":
        """
        应用所有挂载的图像处理钩子函数到当前帧。

        该方法将已注册的所有预处理 Hook 按顺序依次作用于输入帧，实现图像处理流程的模块化扩展。
        每个 Hook 必须实现 `do()` 方法，并返回处理后的帧对象。

        Parameters
        ----------
        frame : VideoFrame
            原始视频帧对象，包含帧数据和元信息（如帧编号和时间戳）。

        *args :
            传递给 Hook 的其他位置参数。

        **kwargs :
            传递给 Hook 的其他关键字参数。

        Returns
        -------
        VideoFrame
            经过所有挂载 Hook 处理后的帧对象。

        Notes
        -----
        - 所有 Hook 应继承自 `BaseHook`，并实现统一接口。
        - 如果某个 Hook 返回 None，可能导致链式处理中断，应在 Hook 内部处理异常逻辑。
        - 支持动态添加 Hook，可用于图像裁剪、降噪、尺寸压缩、灰度转换等操作。
        """
        for each_hook in self._hook_list:
            frame = each_hook.do(frame, *args, **kwargs)
        return frame

    @staticmethod
    def compare_frame_list(src: list["numpy.ndarray"], target: list["numpy.ndarray"]) -> list[float]:
        """
        比较两组图像块的相似性指标，包括 SSIM、MSE 和 PSNR。

        本方法逐一对比两组图像块中对应位置的图像，计算结构相似度（SSIM）、均方误差（MSE）和峰值信噪比（PSNR）。
        对于 SSIM 取所有对比结果中的最小值，MSE 和 PSNR 分别取最大值，以体现最“差”匹配的特征。

        Parameters
        ----------
        src : list of numpy.ndarray
            第一组图像块（源图像），通常由 pic_split 拆分得到。

        target : list of numpy.ndarray
            第二组图像块（目标图像），与源图像块一一对应。

        Returns
        -------
        list of float
            返回一个包含三项指标的列表：
            - 最小结构相似度 SSIM（越小表示差异越大）；
            - 最大均方误差 MSE（越大表示差异越大）；
            - 最大峰值信噪比 PSNR（越大表示差异越小，但仅取最大值以辅助分析）。

        Notes
        -----
        - 两组图像块的长度必须一致，且应逐块对应。
        - SSIM 越接近 1 表示图像越相似；MSE 越低表示误差越小；PSNR 越高表示质量越高。
        - 该方法强调异常值敏感性，用于捕捉帧间细粒度差异。
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
        将一个整数范围平均划分为若干段，并在每段之间插入间隔区间。

        本方法将整数范围 `[1, value]` 拆分为指定数量的 `parts` 份，并在每一份之间插入一个长度为 1 的间隔区域。
        返回的结果包含数据段和间隔段，用于多段数据分布场景下的任务拆分、分析或可视化。

        Parameters
        ----------
        value : int
            要拆分的整数总范围，例如视频的帧数。

        parts : int
            拆分的份数，例如期望的稳定阶段数量。

        Returns
        -------
        list of tuple[int, int, int]
            每个元组表示一个范围 (start, end, length)，包含原始数据段与间隔段。
            - 原始数据段为等长（最后一段可能稍长以容纳余数）；
            - 间隔段长度固定为 1（表示两个段之间的“断点”）。

        Notes
        -----
        - 最终返回的段数量为 `2 * parts - 1`，即包含 `parts` 个主段和 `parts - 1` 个间隔段。
        - 通常用于稳定片段划分、采样任务分批执行等需求。
        - 所有区间从 `1` 开始计数，包含左右边界。
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
        基于滑动窗口和图像分块的方式对视频帧进行特征分析，生成连续帧段的相似性评估范围。

        该方法通过遍历视频帧构建滑动窗口，并在窗口内对图像帧进行分块比对，计算 SSIM、MSE 和 PSNR 指标，
        最终生成若干 `VideoCutRange` 对象，表示视频中连续帧之间的变化程度。

        Parameters
        ----------
        video : VideoObject
            视频对象，用于提取帧和元信息。

        block : int
            图像划分的分块数，例如 `3` 表示划分为 3x3 小块。用于提升变化检测的局部敏感性。

        window_size : int
            滑动窗口的帧数。窗口中每帧都将与起始帧进行对比，用于反映短时变化趋势。

        window_coefficient : int
            加权系数，用于平滑窗口内的相似性评分，靠近窗口尾部的帧权重更高。

        Returns
        -------
        list of VideoCutRange
            返回一个由 `VideoCutRange` 对象组成的列表，每个对象代表一个帧段，并附带其 SSIM/MSE/PSNR 指标。

        Notes
        -----
        - SSIM 用于衡量图像结构相似性，数值越高表示图像越相似。
        - MSE 表示均方误差，越低越好，越高则说明变化大。
        - PSNR 表示峰值信噪比，越高代表图像越清晰。
        - 该方法会自动跳过超过视频总帧数的区域，确保窗口合法。

        Workflow
        --------
        1. 初始化滑动窗口，设置起始帧索引。
        2. 每次窗口内加载若干帧，应用所有注册的 hook。
        3. 将帧图像分块后，对每对相邻帧计算 SSIM、MSE、PSNR。
        4. 对窗口内的所有对比结果应用加权平均（根据 `window_coefficient`）。
        5. 构造 `VideoCutRange`，记录起始帧、结束帧、时间戳及相似性指标。
        6. 窗口右移，重复上述过程，直到处理完整个视频。
        7. 返回所有构造的 `VideoCutRange` 列表，用于后续分析或剪辑。
        """

        def slice_frame() -> list["VideoCutRange"]:
            """
            执行滑动窗口的视频帧分析流程，将连续帧之间的相似性评估结果封装为 VideoCutRange 对象。

            该函数内部使用滑动窗口从视频中按顺序读取帧，并应用图像处理 hook 和图像分块（block split）策略，
            依次比较当前帧与窗口内其他帧之间的结构相似性（SSIM）、均方误差（MSE）与峰值信噪比（PSNR）。
            计算完成后，将分析结果封装为 `VideoCutRange`，用于后续剪辑分析或阶段识别。

            Returns
            -------
            list of VideoCutRange
                视频范围段列表，每段对应一对连续帧间的图像相似性分析结果，包含起止帧 ID、时间戳、SSIM、MSE 和 PSNR。

           Notes
            -----
            - 每个 `VideoCutRange` 表示两个时间点间的视频视觉变化程度。
            - 加权策略使得距离窗口末尾的帧对结果影响更大（通过 window_coefficient）。
            - 该函数是 `magic_frame_range` 的核心组成部分。

            Workflow
            --------
            1. 初始化空的范围列表 `range_list`，准备用于存储分析结果。
            2. 启动进度条，覆盖整个视频帧数。
            3. 每轮循环通过滑动窗口 `window.load_data()` 加载一组帧。
            4. 对窗口中每帧应用图像增强 Hook（如裁剪、灰度处理、压缩等）。
            5. 将参考帧与窗口内后续帧逐对进行分块比对，计算 SSIM、MSE、PSNR。
            6. 使用 `float_merge()` 方法对每项指标进行加权平均。
            7. 构造 `VideoCutRange`，记录当前帧段的起止帧、时间、指标等，并加入结果列表。
            8. 通过 `window.shift()` 向后滑动窗口，重复分析流程。
            9. 所有帧处理完成后，关闭进度条，返回完整分析结果列表。
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
            对滑动窗口中的相似性度量值进行加权平均处理。

            此方法用于融合滑动窗口中多个帧对的图像指标（如 SSIM、MSE、PSNR），
            并根据其在窗口中的位置给予不同权重，靠近窗口末尾的帧将获得更高的权重，
            从而提升对后续变化趋势的响应能力。

            Parameters
            ----------
            float_list : list of float
                来自滑动窗口中连续帧之间计算出的某类指标值（例如 SSIM）。

            Returns
            -------
            float
                加权平均后的融合值，代表当前窗口段落的整体指标表现。

            Notes
            -----
            - 加权权重计算方式为：(length - index)^window_coefficient，指数型增强后端帧影响力。
            - 权重分母 `denominator` 为所有权重之和，用于归一化。
            - 若 `float_list` 趋于一致，输出值近似于平均值；若变化剧烈，输出更偏向后端值。
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
            视频滑动窗口类，用于逐段加载视频帧以进行局部分析。

            该类用于模拟滑动窗口机制，通过控制 `start` 和 `end` 字段，
            实现按帧推进并动态截取视频帧范围，支撑 `magic_frame_range` 中的窗口处理逻辑。

            Attributes
            ----------
            start : int
                当前窗口的起始帧 ID（包含）。

            end : int
                当前窗口的结束帧 ID（包含），受 step 和 window_size 控制。

            size : int
                窗口包含的帧数，通常为 `window_size`。

            Notes
            -----
            - `Window` 封装了窗口帧加载逻辑，是局部时序建模的基础单元。
            - 当窗口滑动到末端，会自动限制尾部帧 ID 不超过视频长度。
            """

            def __init__(self):
                self.start = 1
                self.size = window_size
                self.end = self.start + window_size * step

            def load_data(self) -> list["VideoFrame"]:
                """
                载入当前窗口中的帧列表，按 `step` 步长从起始帧逐帧读取。
                保证窗口至少包含两个帧（起始帧 + 目标帧），用于区段构造。
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
                将窗口向后平移一个单位（即步长），更新 `start` 和 `end`。
                返回 False 表示滑动结束（已超出视频范围）。
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

        # Note
        # ## `step` 参数具体作用和使用场景
        # - 作用
        #   - `step` 参数用于控制窗口在视频帧序列上的移动步长。
        #   - 在每次调用 `shift()` 方法时，窗口的起始和结束位置都会根据 `step` 的值进行调整。
        #   - `step` 决定了每次窗口移动时跳过的帧数。例如，`step=1` 时，窗口每次移动一帧；`step=2` 时，窗口每次移动两帧。
        # - 使用场景
        #   - 当需要对视频帧进行精细分析时，可以设置较小的 `step` 值（例如 `step=1` 或 `step=2`），确保每次窗口移动时覆盖更多的帧，提供更高的分析精度。
        #   - 在需要快速处理视频，且对精度要求不高的情况下，可以设置较大的 `step` 值（例如 `step=5` 或 `step=10`），这样窗口每次移动时跳过更多帧，从而提高处理效率。
        #   - 在进行视频帧数据的平滑处理时，通过调整 `step` 值，可以控制窗口的重叠程度，从而平衡处理速度和数据平滑效果。较小的 `step` 值可以增加帧间重叠，提供更平滑的数据结果。
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
        对输入视频执行滑动窗口剪辑处理，输出帧分段结果。

        该方法用于将视频帧序列切分为若干段落（`VideoCutRange`），每段表示两个关键帧之间的相似性变化，
        主要用于检测帧间稳定性、运动段落、剪辑边界等结构。其核心逻辑基于滑动窗口与图像块的对比分析。

        Parameters
        ----------
        video : str or VideoObject
            输入的视频对象或视频文件路径。若为字符串，将自动构造为 `VideoObject`。

        block : int, optional
            图像块划分数（默认 3），用于将每帧分割成 `block x block` 的子块进行局部相似性分析。

        window_size : int, optional
            滑动窗口大小，控制每次处理的帧数量（默认 1）。窗口中第一个帧与其余帧逐对比对。

        window_coefficient : int, optional
            窗口权重指数（默认 2），用于加权平均时控制窗口中各帧对的贡献程度。

        **kwargs :
            传递给 `VideoCutResult` 的剪辑配置，例如压缩率、灰度转换等钩子参数。

        Returns
        -------
        VideoCutResult
            视频剪辑结果对象，包含切割区间列表（`VideoCutRange`）和处理参数。

        Workflow
        --------
        1. 若输入是字符串路径，则使用 `VideoObject` 封装。
        2. 设置默认参数（block=3，window_size=1，window_coefficient=2）。
        3. 调用 `magic_frame_range()` 方法执行剪辑主逻辑，返回区间列表。
        4. 构建并返回 `VideoCutResult` 对象，封装所有结果及配置信息。

        Notes
        -----
        - 若视频总帧数为 N，则生成的帧段落为 N-1。
        - 每个 `VideoCutRange` 表示两帧之间的图像差异，包含 SSIM/MSE/PSNR 指标。
        - 本方法适用于帧级动态分析、剪辑检测、稳定性建模等场景。
        - 可结合 `VideoCutResult.get_range()` 自动筛选稳定与不稳定段落。
        """
        video = VideoObject(video) if isinstance(video, str) else video

        block = block or 3
        window_size = window_size or 1
        window_coefficient = window_coefficient or 2

        # Note
        # 如果视频包含 100 帧
        # 从1开始，列表长度是99，而不是100
        # [范围(1-2)、范围(2-3)、范围(3-4) ... 范围(99-100)]
        range_list = self.magic_frame_range(
            video, block, window_size, window_coefficient
        )

        return VideoCutResult(video, range_list, cut_kwargs=kwargs)


if __name__ == '__main__':
    pass
