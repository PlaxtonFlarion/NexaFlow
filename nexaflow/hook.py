#   _   _             _
#  | | | | ___   ___ | | __
#  | |_| |/ _ \ / _ \| |/ /
#  |  _  | (_) | (_) |   <
#  |_| |_|\___/ \___/|_|\_\
#
# ==== Notes: License ====
# Copyright (c) 2024  Framix :: 画帧秀
# This file is licensed under the Framix :: 画帧秀 License. See the LICENSE.md file for more details.

import os
import cv2
import typing
# from loguru import logger
from nexaflow import toolbox
from nexaflow.video import VideoFrame


class BaseHook(object):
    """
    图像帧处理钩子基类。

    提供统一的接口结构，供所有图像帧处理钩子继承使用。该类主要负责基础结构初始化与 `do()` 方法定义，
    子类应重写 `do()` 方法以执行实际的图像处理任务。

    Attributes
    ----------
    result : dict
        用于记录钩子处理结果或中间状态信息的结果字典。
    """

    def __init__(self, *_, **__):
        """
        初始化钩子基类，创建结果记录字典。
        """
        # logger.debug(f"start initialing: {self.__class__.__name__} ...")
        self.result = dict()

    def do(self, frame: "VideoFrame", *_, **__) -> typing.Optional["VideoFrame"]:
        """
        钩子处理主方法，供子类重写。

        Parameters
        ----------
        frame : VideoFrame
            传入的图像帧对象。

        Returns
        -------
        Optional[VideoFrame]
            原始或处理后的图像帧对象。
        """
        # info = f"execute hook: {self.__class__.__name__}"

        frame_id = frame.frame_id
        if frame_id != -1:
            # logger.debug(f"{info}, frame id: {frame_id}")
            pass
        return frame


class ExampleHook(BaseHook):
    """
    示例钩子类：将帧图像转换为灰度图，并记录处理后的图像尺寸。

    该类继承自 BaseHook，并在 `do()` 方法中调用工具函数进行图像灰度转换，同时将结果记录到 result 字典中。
    """

    def __init__(self, *_, **__):
        """
        初始化 ExampleHook，调用基类构造方法。
        """
        super().__init__(*_, **__)

    def do(self, frame: "VideoFrame", *_, **__) -> typing.Optional["VideoFrame"]:
        """
        对帧图像执行灰度转换，并记录图像尺寸。

        Parameters
        ----------
        frame : VideoFrame
            待处理的图像帧对象。

        Returns
        -------
        Optional[VideoFrame]
            灰度处理后的图像帧对象。
        """
        super().do(frame, *_, **__)
        frame.data = toolbox.turn_grey(frame.data)
        self.result[frame.frame_id] = frame.data.shape

        return frame


class FrameSizeHook(BaseHook):
    """
    帧尺寸调整钩子。

    用于对视频帧进行压缩或重设尺寸，可配合是否转灰度处理。支持基于压缩比例或目标尺寸的灵活缩放方式。

    Parameters
    ----------
    compress_rate : float, optional
        缩放系数，范围通常在 (0, 1]，用于等比缩小图像。若指定该参数，将按比例压缩图像。

    target_size : tuple of int, optional
        指定输出图像的尺寸 (width, height)。优先级高于 `compress_rate`，用于将帧重设为固定尺寸。

    not_grey : bool, optional
        是否保持彩色输出。若为 `True`，保留彩色；若为 `False` 或未指定，则默认转换为灰度图像。

    Notes
    -----
    - 当 `target_size` 被指定时，`compress_rate` 将被忽略。
    - 该钩子常用于模型推理前的图像预处理或数据降采样任务。
    """

    def __init__(
        self,
        compress_rate: float = None,
        target_size: tuple[int, int] = None,
        not_grey: bool = None,
        *_,
        **__,
    ):
        super().__init__(*_, **__)
        self.compress_rate = compress_rate
        self.target_size = target_size
        self.not_grey = not_grey
        # logger.debug(f"compress rate: {compress_rate}")
        # logger.debug(f"target size: {target_size}")

    def do(self, frame: "VideoFrame", *_, **__) -> typing.Optional["VideoFrame"]:
        """
        对图像帧执行压缩或尺寸调整处理。

        Parameters
        ----------
        frame : VideoFrame
            输入帧对象，包含图像数据和元信息。

        Returns
        -------
        Optional[VideoFrame]
            调整后的帧对象。其 `data` 属性将根据指定的压缩率或目标尺寸被修改。
        """
        super().do(frame, *_, **__)
        frame.data = toolbox.compress_frame(
            frame.data, self.compress_rate, self.target_size, self.not_grey
        )
        return frame


class GreyHook(BaseHook):
    """
    灰度转换钩子。

    用于将视频帧或图像帧转换为灰度图。适用于图像压缩、边缘提取、特征提取等前处理步骤。

    Notes
    -----
    - 本钩子可作为图像预处理的一部分，常用于简化后续模型计算量。
    """

    def do(self, frame: "VideoFrame", *_, **__) -> typing.Optional["VideoFrame"]:
        """
        对图像帧进行灰度转换。

        Parameters
        ----------
        frame : VideoFrame
            当前处理的帧对象。

        Returns
        -------
        Optional[VideoFrame]
            处理后的帧对象，其 `data` 属性为灰度图像。
        """
        super().do(frame, *_, **__)
        frame.data = toolbox.turn_grey(frame.data)
        return frame


class RefineHook(BaseHook):
    """
    图像锐化钩子。

    使用边缘增强或高通滤波等技术提升图像清晰度，增强边缘信息。适合用于特征检测、轮廓分析等任务。

    Notes
    -----
    - 此钩子适合在图像模糊或低清晰度情况下作为增强手段。
    - 对图像噪声敏感，建议结合降噪操作使用。
    """

    def do(self, frame: "VideoFrame", *_, **__) -> typing.Optional["VideoFrame"]:
        """
        对图像帧进行锐化处理。

        Parameters
        ----------
        frame : VideoFrame
            当前处理的帧对象。

        Returns
        -------
        Optional[VideoFrame]
            处理后的帧对象，其 `data` 属性为增强锐度后的图像。
        """
        super().do(frame, *_, **__)
        frame.data = toolbox.sharpen_frame(frame.data)
        return frame


class _AreaBaseHook(BaseHook):
    """
    区域裁剪钩子基类，支持按比例或固定像素尺寸定义裁剪区域。

    该类用于派生具体的图像裁剪操作钩子，例如遮罩、清除、提取区域等。
    通过 `size` 和 `offset` 参数灵活指定作用区域，可根据图像尺寸自动换算为实际像素。

    Parameters
    ----------
    size : tuple[int | float, int | float]
        区域的高宽定义。支持整数（像素）或 0~1 之间的浮点数（相对比例）。

    offset : tuple[int | float, int | float], optional
        区域左上角偏移量。支持像素或比例，默认为 (0, 0) 表示从图像左上角开始。

    Notes
    -----
    - 如果 `size` 或 `offset` 中的值为小于等于 1 的浮点数，则会根据原图尺寸自动计算实际像素。
    - 该类通常配合 `VideoFrame` 或图像处理流水线使用。
    """

    def __init__(
        self,
        size: tuple[typing.Union[int, float], typing.Union[int, float]],
        offset: tuple[typing.Union[int, float], typing.Union[int, float]] = None,
        *_,
        **__,
    ):
        super().__init__(*_, **__)
        self.size = size
        self.offset = offset or (0, 0)
        # logger.debug(f"size: {self.size}")
        # logger.debug(f"offset: {self.offset}")

    @staticmethod
    def is_proportion(
        target: tuple[typing.Union[int, float], typing.Union[int, float]]
    ) -> bool:
        """
        判断目标尺寸是否为相对比例。

        Parameters
        ----------
        target : tuple[int | float, int | float]
            包含两个数值，表示高和宽。

        Returns
        -------
        bool
            若两个值都在 [0.0, 1.0] 范围内，则判定为比例定义，返回 True。
        """
        return len([i for i in target if 0.0 <= i <= 1.0]) == 2

    @staticmethod
    def convert(
        origin_h: int,
        origin_w: int,
        input_h: typing.Union[float, int],
        input_w: typing.Union[float, int],
    ) -> tuple[typing.Union[int, float], typing.Union[int, float]]:
        """
        将输入尺寸从比例或像素转换为实际像素单位。

        Parameters
        ----------
        origin_h : int
            原始图像高度（像素）。

        origin_w : int
            原始图像宽度（像素）。

        input_h : int | float
            输入高度，可能为像素或比例。

        input_w : int | float
            输入宽度，可能为像素或比例。

        Returns
        -------
        tuple[int | float, int | float]
            转换后的高宽值（像素单位）。
        """
        if _AreaBaseHook.is_proportion((input_h, input_w)):
            return origin_h * input_h, origin_w * input_w
        return input_h, input_w

    def convert_size_and_offset(
        self, *origin_size
    ) -> tuple[tuple, tuple]:
        """
        根据原始图像尺寸，计算实际裁剪区域的位置范围。

        Parameters
        ----------
        origin_size : tuple[int, int]
            图像的原始尺寸 (height, width)。

        Returns
        -------
        tuple[tuple[int, int], tuple[int, int]]
            返回两个元组，分别为高度范围 `(start_h, end_h)` 和宽度范围 `(start_w, end_w)`。
        """
        size_h, size_w = self.convert(*origin_size, *self.size)
        # logger.debug(f"size: ({size_h}, {size_w})")
        offset_h, offset_w = self.convert(*origin_size, *self.offset)
        # logger.debug(f"offset: {offset_h}, {offset_w}")
        height_range, width_range = (
            (int(offset_h), int(offset_h + size_h)),
            (int(offset_w), int(offset_w + size_w)),
        )
        # logger.debug(f"final range h: {height_range}, w: {width_range}")
        return height_range, width_range


class CropHook(_AreaBaseHook):

    def do(self, frame: "VideoFrame", *_, **__) -> typing.Optional["VideoFrame"]:
        super().do(frame, *_, **__)

        height_range, width_range = self.convert_size_and_offset(
            *frame.data.shape
        )
        frame.data[: height_range[0], :] = 0
        frame.data[height_range[1]:, :] = 0
        frame.data[:, : width_range[0]] = 0
        frame.data[:, width_range[1]:] = 0

        return frame


class OmitHook(_AreaBaseHook):

    def do(self, frame: "VideoFrame", *_, **__) -> typing.Optional["VideoFrame"]:
        super().do(frame, *_, **__)

        height_range, width_range = self.convert_size_and_offset(
            *frame.data.shape
        )
        frame.data[
            height_range[0]: height_range[1], width_range[0]: width_range[1]
        ] = 0

        return frame


class PaintCropHook(_AreaBaseHook):

    def do(self, frame: "VideoFrame", *_, **__) -> typing.Optional["VideoFrame"]:
        """
        执行裁剪操作，将指定区域外的图像部分置为零。

        Parameters
        ----------
        frame : VideoFrame
            视频帧对象，包含待处理的图像数据。

        Returns
        -------
        typing.Optional[VideoFrame]
            返回处理后的帧对象，图像数据中非裁剪区域以外的部分将被清除（置为 0）。

        Notes
        -----
        本方法会基于已配置的区域范围（宽高）裁剪图像，仅保留中心区域，边缘区域清零。

        Workflow
        --------
        1. 继承父类逻辑，记录当前帧状态。
        2. 计算裁剪区域坐标范围。
        3. 将图像中不在裁剪区域内的像素全部置为 0。
        """
        super().do(frame, *_, **__)

        height_range, width_range = self.convert_size_and_offset(
            *frame.data.shape[:2]
        )
        frame.data[: height_range[0], :] = 0
        frame.data[height_range[1]:, :] = 0
        frame.data[:, : width_range[0]] = 0
        frame.data[:, width_range[1]:] = 0

        return frame


class PaintOmitHook(_AreaBaseHook):

    def do(self, frame: "VideoFrame", *_, **__) -> typing.Optional["VideoFrame"]:
        """
        执行遮挡操作，将指定区域内的图像部分置为零。

        Parameters
        ----------
        frame : VideoFrame
            视频帧对象，包含待处理的图像数据。

        Returns
        -------
        typing.Optional[VideoFrame]
            返回处理后的帧对象，图像数据中指定区域将被清除（置为 0）。

        Notes
        -----
        本方法会根据指定的区域范围对图像做局部遮挡处理，仅清除该区域像素数据。

        Workflow
        --------
        1. 继承父类逻辑，记录当前帧状态。
        2. 计算遮挡区域的坐标范围。
        3. 将图像中该区域的像素全部置为 0。
        """
        super().do(frame, *_, **__)

        height_range, width_range = self.convert_size_and_offset(
            *frame.data.shape[:2]
        )
        frame.data[
            height_range[0]: height_range[1], width_range[0]: width_range[1]
        ] = 0

        return frame


class FrameSaveHook(BaseHook):

    def __init__(self, target_dir: str, *_, **__):
        super().__init__(*_, **__)

        self.target_dir = target_dir
        os.makedirs(target_dir, exist_ok=True)
        # logger.debug(f"target dir: {target_dir}")

    def do(self, frame: "VideoFrame", *_, **__) -> typing.Optional["VideoFrame"]:
        """
        保存视频帧图像到目标目录。

        Parameters
        ----------
        frame : VideoFrame
            当前处理的视频帧对象，包含帧 ID、时间戳和图像数据。
        *_ : Any
            保留的扩展位置参数。
        **__ : Any
            保留的扩展关键字参数。

        Returns
        -------
        typing.Optional[VideoFrame]
            返回当前视频帧对象，图像已被保存。

        Notes
        -----
        - 该方法将视频帧以 PNG 格式保存为图片文件，文件名为 `frame_id(timestamp).png`。
        - 为兼容中文路径，采用 `cv2.imencode(...).tofile(...)` 写入图片。
        - 帧图像保存路径为初始化传入的 `target_dir`。

        Workflow
        --------
        1. 调用父类的 `do` 方法。
        2. 构建帧文件名并转义时间戳中的小数点。
        3. 拼接目标路径。
        4. 使用 OpenCV 编码并保存为 PNG 文件。
        5. 返回原始帧对象。
        """
        super().do(frame, *_, **__)

        safe_timestamp = str(frame.timestamp).replace(".", "_")
        frame_name = f"{frame.frame_id}({safe_timestamp}).png"
        target_path = os.path.join(self.target_dir, frame_name)

        # 不能保存中文路径
        # cv2.imwrite(target_path, frame.data)
        # logger.debug(f"frame saved to {target_path}")

        # 保存中文路径
        cv2.imencode(".png", frame.data)[1].tofile(target_path)

        return frame


if __name__ == '__main__':
    pass
