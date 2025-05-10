#
#   _____           _ ____
#  |_   _|__   ___ | | __ )  _____  __
#    | |/ _ \ / _ \| |  _ \ / _ \ \/ /
#    | | (_) | (_) | | |_) | (_) >  <
#    |_|\___/ \___/|_|____/ \___/_/\_\
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
import math
import time
import shutil
import random
import typing
import contextlib
import subprocess
import numpy as np
from tqdm import tqdm
from loguru import logger
from findit import FindIt
from base64 import b64encode
from PIL import (
    Image, ImageDraw, ImageFont
)
from skimage.feature import (
    hog, local_binary_pattern
)
from skimage.metrics import (
    normalized_root_mse as compare_nrmse,
    peak_signal_noise_ratio as compare_psnr,
    structural_similarity as origin_compare_ssim
)
from nexaflow import const


@contextlib.contextmanager
def video_capture(video_path: str):
    """
    视频捕获上下文管理器。

    该方法基于 contextlib 提供一个简洁的方式，用于自动管理视频捕获对象（cv2.VideoCapture）的释放过程。

    Parameters
    ----------
    video_path : str
        视频文件的路径。

    Yields
    ------
    cv2.VideoCapture
        OpenCV 的视频捕获对象，可用于逐帧读取视频。

    Notes
    -----
    使用该上下文管理器可以避免手动调用 `.release()` 方法，在代码块结束时自动释放资源，防止资源泄露。
    常用于需要读取视频帧的分析任务或图像处理流程中。
    """
    video_cap = cv2.VideoCapture(video_path)
    try:
        yield video_cap
    finally:
        video_cap.release()


def video_jump(video_cap: "cv2.VideoCapture", frame_id: int) -> None:
    """
    跳转到指定帧位置以定位视频帧。

    本函数将 OpenCV 视频捕获对象跳转至指定帧位置，并执行一次预读取操作以确保缓冲同步。

    Parameters
    ----------
    video_cap : cv2.VideoCapture
        OpenCV 视频捕获对象，必须是已打开的视频流。

    frame_id : int
        目标帧的编号（从 1 开始计数）。

    Returns
    -------
    None

    Notes
    -----
    - 跳转时实际设置帧为 `frame_id - 2`，然后执行一次 `.read()`，这是为了避开部分播放器的预缓冲机制，确保定位准确。
    - 如果帧 ID 小于 2，可能会导致设置无效，建议调用前进行合法性检查。

    Workflow
    --------
    1. 使用 `cv2.CAP_PROP_POS_FRAMES` 设置帧位置。
    2. 立即读取一次帧，完成解码器状态同步。
    """
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1 - 1)
    video_cap.read()


def compare_ssim(pic1: "np.ndarray", pic2: "np.ndarray") -> float:
    """
    计算两张图片的结构相似性（SSIM）。

    本函数接收两张图像并转换为灰度图后，使用 SSIM 指标评估它们在结构上的相似程度，结果范围为 0~1，值越高相似度越高。

    Parameters
    ----------
    pic1 : np.ndarray
        第一张图片的数组形式，通常为三通道彩色图或单通道灰度图。

    pic2 : np.ndarray
        第二张图片的数组形式，要求尺寸与 pic1 一致。

    Returns
    -------
    float
        两张图像之间的结构相似度（Structural Similarity Index），范围 [0.0, 1.0]。

    Notes
    -----
    - 输入图像会被自动转换为灰度图再进行 SSIM 计算。
    - 图像尺寸需相同，否则可能引发运行时错误。
    - SSIM（结构相似性）可用于图像质量评估、变化检测等场景。

    Workflow
    --------
    1. 将两张图像转换为灰度图。
    2. 使用 `skimage.metrics.structural_similarity` 进行 SSIM 计算。
    """
    pic1, pic2 = [turn_grey(i) for i in [pic1, pic2]]
    return origin_compare_ssim(pic1, pic2)


def multi_compare_ssim(
    pic1_list: list, pic2_list: list, hooks: typing.Optional[list] = None
) -> list[float]:
    """
    对多个图像帧列表执行结构相似度（SSIM）批量比较。

    该函数支持两组图像帧之间的逐一配对比较，支持通过 hooks 对图像帧进行预处理操作（如裁剪、缩放等），最终返回每一对图像的 SSIM 相似度结果。

    Parameters
    ----------
    pic1_list : list
        第一组图像帧列表，元素为 `np.ndarray` 或 `VideoFrame`。

    pic2_list : list
        第二组图像帧列表，元素为 `np.ndarray` 或 `VideoFrame`，与 pic1_list 一一对应。

    hooks : list, optional
        可选的图像帧处理器列表，每个 hook 应包含 `do(frame)` 方法用于帧处理。

    Returns
    -------
    list of float
        每对图像帧之间的 SSIM 相似度得分列表。

    Notes
    -----
    - 若输入为 `VideoFrame` 类型，将提取其中的 `.data` 属性。
    - 若提供 hooks，则会在比较之前对图像帧进行预处理。
    - 输出列表中的每个值对应于一对输入帧的 SSIM 相似度。
    """
    from nexaflow.video import VideoFrame

    if isinstance(pic1_list[0], VideoFrame):
        if hooks:
            for each in hooks:
                pic1_list = [each.do(each_frame) for each_frame in pic1_list]
        pic1_list = [i.data for i in pic1_list]

    if isinstance(pic2_list[0], VideoFrame):
        if hooks:
            for each in hooks:
                pic2_list = [each.do(each_frame) for each_frame in pic2_list]
        pic2_list = [i.data for i in pic2_list]

    return [compare_ssim(a, b) for a, b in zip(pic1_list, pic2_list)]


def get_current_frame_id(video_cap: "cv2.VideoCapture") -> int:
    """
    获取当前帧的帧编号（Frame ID）。

    该函数用于获取当前帧在视频流中的索引位置，返回的是整数帧编号（从0开始计数）。

    Parameters
    ----------
    video_cap : cv2.VideoCapture
        已打开的视频捕获对象。

    Returns
    -------
    int
        当前帧的索引编号。
    """
    return int(video_cap.get(cv2.CAP_PROP_POS_FRAMES))


def get_current_frame_time(video_cap: "cv2.VideoCapture") -> float:
    """
    获取当前帧的时间戳（单位：秒）。

    该函数返回当前帧对应的时间戳，单位为秒。适用于对视频帧进行精确时间定位。

    Parameters
    ----------
    video_cap : cv2.VideoCapture
        已打开的视频捕获对象。

    Returns
    -------
    float
        当前帧的时间戳，单位为秒。
    """
    return video_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000


def imread(img_path: str, *_, **__) -> "np.ndarray":
    """
    读取图像文件，包装自 cv2.imread。

    该函数在调用 OpenCV 的 imread 读取图像前，先进行路径合法性判断，确保文件存在。

    Parameters
    ----------
    img_path : str
        图像文件的路径。

    *_
        传递给 cv2.imread 的可选位置参数。

    **__
        传递给 cv2.imread 的可选关键字参数。

    Returns
    -------
    np.ndarray
        读取的图像数据，格式为 NumPy 数组。如果读取失败，返回 None。

    Raises
    ------
    AssertionError
        如果提供的图像路径不存在，则抛出异常。
    """
    assert os.path.isfile(img_path), f"file {img_path} is not existed"
    return cv2.imread(img_path, *_, **__)


def get_frame_time(
    video_cap: "cv2.VideoCapture", frame_id: int, recover: bool = None
) -> float:
    """
    获取指定帧的时间戳。

    该函数将视频跳转到指定帧位置，读取该帧的时间戳信息，并在可选参数设置为 True 时恢复到原帧位置。

    Parameters
    ----------
    video_cap : cv2.VideoCapture
        OpenCV 视频捕获对象。

    frame_id : int
        目标帧的编号（从 1 开始计数）。

    recover : bool, optional
        是否在获取时间戳后跳回原始位置。如果为 True，则会恢复原跳转前的帧位置。

    Returns
    -------
    float
        指定帧对应的时间戳，单位为秒。

    Notes
    -----
    - 本函数调用 `video_jump()` 实现帧跳转，再用 `get_current_frame_time()` 读取时间。
    - 可用于定位关键帧、同步处理或调试输出。

    Workflow
    --------
    1. 记录当前帧位置，避免直接跳转后丢失原位置。
    2. 使用 `video_jump()` 跳转到指定帧（frame_id）。
    3. 调用 `get_current_frame_time()` 获取当前帧的时间戳（秒）。
    4. 若 `recover=True`，则跳回原始帧位置以保证视频状态不变。
    5. 返回该帧的时间戳结果。
    """
    current = get_current_frame_id(video_cap)
    video_jump(video_cap, frame_id)
    result = get_current_frame_time(video_cap)
    logger.debug(f"frame {frame_id} -> {result}")

    if recover:
        video_jump(video_cap, current + 1)
    return result


def get_frame_count(video_cap: "cv2.VideoCapture") -> int:
    """
    获取视频总帧数

    Parameters
    ----------
    video_cap : cv2.VideoCapture
        视频捕获对象，表示当前打开的视频。

    Returns
    -------
    int
        视频的总帧数。

    Notes
    -----
    使用 OpenCV 的 `CAP_PROP_FRAME_COUNT` 属性提取帧数，结果为整数。
    """
    return int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))


def get_frame_size(video_cap: "cv2.VideoCapture") -> tuple[int, int]:
    """
    获取帧的尺寸

    Parameters
    ----------
    video_cap : cv2.VideoCapture
        视频捕获对象，表示当前打开的视频。

    Returns
    -------
    tuple[int, int]
        视频帧的宽度和高度，格式为 (width, height)。

    Notes
    -----
    使用 OpenCV 的 `CAP_PROP_FRAME_WIDTH` 和 `CAP_PROP_FRAME_HEIGHT` 属性获取帧尺寸。
    """
    h = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    return int(w), int(h)


def get_frame(
    video_cap: "cv2.VideoCapture", frame_id: int, recover: typing.Optional[bool] = None
) -> "np.ndarray":
    """
    获取指定帧的图像数据

    Parameters
    ----------
    video_cap : cv2.VideoCapture
        视频读取对象。

    frame_id : int
        目标帧编号（从 1 开始）。

    recover : Optional[bool], default=None
        是否在读取完成后跳回当前帧位置。

    Returns
    -------
    np.ndarray
        指定帧的图像数据，格式为 BGR 图像。

    Raises
    ------
    AssertionError
        如果读取帧失败，抛出断言异常。

    Workflow
    --------
    1. 获取当前帧位置并记录（current）。
    2. 调用 `video_jump` 跳转至目标帧。
    3. 使用 `read()` 读取图像帧，并确保读取成功。
    4. 如果设置了 `recover=True`，则跳转回读取前的帧位置。
    """
    current = get_current_frame_id(video_cap)
    video_jump(video_cap, frame_id)
    ret, frame = video_cap.read()
    assert ret, f"read frame failed, frame id: {frame_id}"

    if recover:
        video_jump(video_cap, current + 1)
    return frame


def turn_grey(old: "np.ndarray") -> "np.ndarray":
    """
    将图像转换为灰度图

    Parameters
    ----------
    old : np.ndarray
        输入图像数组，通常为 RGB 或 BGR 格式。

    Returns
    -------
    np.ndarray
        转换后的灰度图像。

    Notes
    -----
    如果转换过程中发生异常（如图像格式不支持），则返回原始图像。
    """
    try:
        return cv2.cvtColor(old, cv2.COLOR_RGB2GRAY)
    except cv2.error:
        return old


def turn_binary(old: "np.ndarray") -> "np.ndarray":
    """
    将图像转为二值图

    Parameters
    ----------
    old : np.ndarray
        输入图像数组，通常为 RGB 或灰度格式。

    Returns
    -------
    np.ndarray
        转换后的二值图像，像素值为 0 或 255。

    Workflow
    --------
    1. 使用 `turn_grey()` 将图像转换为灰度图。
    2. 使用 OpenCV 的自适应阈值方法进行二值化处理。
    """
    grey = turn_grey(old).astype("uint8")
    return cv2.adaptiveThreshold(
        grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )


def turn_hog_desc(old: "np.ndarray") -> "np.ndarray":
    """
    计算图像的 HOG（Histogram of Oriented Gradients）描述子

    Parameters
    ----------
    old : np.ndarray
        输入图像数组，通常为灰度图。

    Returns
    -------
    np.ndarray
        表示图像纹理方向特征的 HOG 描述子向量。

    Workflow
    --------
    1. 对输入图像使用 skimage.hog 提取特征向量。
    2. 参数包括方向数、单元格大小、归一化方式等。
    """
    fd, _ = hog(
        old,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        block_norm="L2-Hys",
        visualize=True,
    )

    return fd


def turn_lbp_desc(old: "np.ndarray", radius: typing.Optional[int] = None) -> "np.ndarray":
    """
    计算图像的 LBP（Local Binary Pattern）描述子

    Parameters
    ----------
    old : np.ndarray
        输入图像数组，通常为 RGB 或灰度图。

    radius : int, optional
        LBP 半径，默认为 3，决定每个像素点周围采样点的数量。

    Returns
    -------
    np.ndarray
        表示图像局部纹理特征的 LBP 图像矩阵。

    Workflow
    --------
    1. 使用 `turn_grey()` 将图像转换为灰度图。
    2. 设置半径 `radius` 并计算采样点数 `n_points = 8 * radius`。
    3. 使用 `skimage.feature.local_binary_pattern()` 生成 LBP 描述图。
    """
    radius = radius if radius else 3
    n_points = 8 * radius

    grey = turn_grey(old)
    lbp = local_binary_pattern(grey, n_points, radius, method="default")

    return lbp


def turn_blur(old: "np.ndarray") -> "np.ndarray":
    """
    使用高斯模糊处理图像

    Parameters
    ----------
    old : np.ndarray
        输入图像数组。

    Returns
    -------
    np.ndarray
        模糊处理后的图像结果。

    Notes
    -----
    使用 7x7 的高斯核对图像进行平滑，常用于降噪或特征抑制处理。
    """
    return cv2.GaussianBlur(old, (7, 7), 0)


def sharpen_frame(old: "np.ndarray") -> "np.ndarray":
    """
    锐化图像边缘，突出图像轮廓

    Parameters
    ----------
    old : np.ndarray
        原始图像，支持彩色或灰度图。

    Returns
    -------
    np.ndarray
        锐化处理后的图像，通常为边缘二值图。

    Workflow
    --------
    1. 使用 `turn_blur()` 对图像进行高斯模糊，平滑细节。
    2. 使用 `cv2.addWeighted()` 减去部分原图信息以增强对比。
    3. 使用 `cv2.Canny()` 提取图像边缘并返回。
    """
    blur = turn_blur(old)
    smooth = cv2.addWeighted(blur, 1.5, old, -0.5, 0)
    canny = cv2.Canny(smooth, 50, 150)

    return canny


def calc_mse(pic1: "np.ndarray", pic2: "np.ndarray") -> float:
    """
    计算两张图像的均方误差（Mean Squared Error）

    Parameters
    ----------
    pic1 : np.ndarray
        第一张图像。

    pic2 : np.ndarray
        第二张图像。

    Returns
    -------
    float
        两张图像之间的均方误差，值越小表示越相似。

    Notes
    -----
    实际调用了 `compare_nrmse()` 函数实现计算，等价于 MSE 形式。
    """
    return compare_nrmse(pic1, pic2)


def calc_psnr(pic1: "np.ndarray", pic2: "np.ndarray") -> float:
    """
    计算两图像的峰值信噪比（PSNR）

    Parameters
    ----------
    pic1 : np.ndarray
        第一张图像。

    pic2 : np.ndarray
        第二张图像。

    Returns
    -------
    float
        标准化后的 PSNR 值，范围为 0 ~ 1，值越大表示图像越相似。

    Notes
    -----
    - 实际使用 `compare_psnr()` 计算原始 PSNR 值。
    - 若图像完全相同，PSNR 为无穷大，此时自动转为 100。
    - 最终返回值会除以 100 进行归一化处理。
    """
    psnr = compare_psnr(pic1, pic2)
    if math.isinf(psnr):
        psnr = 100.0

    return psnr / 100


def compress_frame(
    frame: "np.ndarray",
    compress_rate: typing.Optional[typing.Union[int, float]] = None,
    target_size: typing.Optional[tuple] = None,
    not_grey: typing.Optional[bool] = None,
    interpolation: typing.Optional[int] = None,
    *_,
    **__,
) -> "np.ndarray":
    """
    压缩图像帧（支持缩放或重设尺寸）

    Parameters
    ----------
    frame : np.ndarray
        原始图像帧。

    compress_rate : int or float, optional
        缩放比例（如 0.5 表示缩小为原尺寸的一半）。

    target_size : tuple, optional
        目标尺寸，格式为 (width, height)，优先于 compress_rate。

    not_grey : bool, optional
        若为 False，则先将图像转为灰度图后再处理。

    interpolation : int, optional
        插值方式，默认使用 `cv2.INTER_AREA`。

    Returns
    -------
    np.ndarray
        压缩后的图像帧。

    Workflow
    --------
    1. 根据 `not_grey` 判断是否先转为灰度图。
    2. 若指定 `target_size`，则直接重设尺寸。
    3. 否则使用 `compress_rate` 进行等比例缩放。
    4. 最终使用 `cv2.resize()` 返回结果。
    """
    target = frame if not_grey else turn_grey(frame)

    interpolation = interpolation or cv2.INTER_AREA

    if target_size:
        return cv2.resize(target, target_size, interpolation=interpolation)

    if not compress_rate:
        return target

    return cv2.resize(target, (0, 0), fx=compress_rate, fy=compress_rate, interpolation=interpolation)


def get_timestamp_str() -> str:
    """
    获取当前时间戳字符串

    Returns
    -------
    str
        当前时间戳字符串，格式为 "年月日时分秒 + 两位随机数"，如 "2025041015324592"。

    Notes
    -----
    - 使用 `time.strftime()` 获取当前时间。
    - 添加两位随机数作为后缀，确保字符串唯一性。
    """
    time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    salt = random.randint(10, 99)
    return f"{time_str}{salt}"


def np2b64str(frame: "np.ndarray") -> str:
    """
    将Numpy数组（图像）转为Base64字符串

    Parameters
    ----------
    frame : np.ndarray
        输入图像数组，通常为 OpenCV 格式。

    Returns
    -------
    str
        图像的 Base64 编码字符串。

    Notes
    -----
    - 使用 `cv2.imencode()` 将图像编码为 PNG 格式的二进制数据。
    - 再通过 `base64.b64encode()` 转为字符串以便存储或网络传输。
    """
    buffer = cv2.imencode(".png", frame)[1]
    return b64encode(buffer).decode()


def fps_convert(
    target_fps: int, source_path: str, target_path: str, ffmpeg_exe: typing.Optional[str] = None
) -> int:
    """
    转换视频的帧率

    Parameters
    ----------
    target_fps : int
        目标帧率（例如 30、60 等）。

    source_path : str
        输入视频的路径。

    target_path : str
        输出视频的保存路径。

    ffmpeg_exe : str, optional
        指定的 ffmpeg 可执行文件路径，默认为系统环境中的 ffmpeg。

    Returns
    -------
    int
        执行状态码，0 表示成功，非 0 表示失败。

    Notes
    -----
    - 使用 `subprocess.check_call` 调用 ffmpeg 命令行工具。
    - 构建的命令格式为：`ffmpeg -i 输入路径 -r 帧率 输出路径`。
    - 若未指定 ffmpeg 路径，则默认使用系统环境中的 `ffmpeg`。
    """
    ffmpeg_exe = ffmpeg_exe if ffmpeg_exe else r"ffmpeg"

    command: list[str] = [
        ffmpeg_exe, "-i", source_path, "-r", str(target_fps), target_path,
    ]
    logger.debug(f"convert video: {command}")

    return subprocess.check_call(command)


def match_template_with_object(
    template: "np.ndarray",
    target: "np.ndarray",
    engine_template_cv_method_name: str = None,
    **kwargs,
) -> dict[str, typing.Any]:
    """
    使用模板匹配找出目标图像中与模板相似的部分

    Parameters
    ----------
    template : np.ndarray
        用作匹配参考的模板图像。
    target : np.ndarray
        待匹配的目标图像。
    engine_template_cv_method_name : str, optional
        使用的 OpenCV 模板匹配方法名（如 "cv2.TM_CCOEFF_NORMED"），默认为此值。
    **kwargs :
        传递给 FindIt 引擎的其他参数，如匹配阈值、搜索范围等。

    Returns
    -------
    dict[str, typing.Any]
        包含匹配结果的字典，结构为 `TemplateEngine` 输出格式，通常包含位置、分数等信息。

    Notes
    -----
    - 使用 FindIt 引擎进行图像模板匹配，支持多种方法。
    - 若未指定匹配方法，默认使用 `cv2.TM_CCOEFF_NORMED`。
    - 匹配过程通过调用 `load_template` 和 `find` 方法完成，并从结果中提取目标模板对应的匹配数据。
    - 日志记录匹配结果以便调试。

    Workflow
    --------
    1. 初始化 FindIt 匹配器并设置匹配方法。
    2. 加载模板图像到匹配器中，命名为 "default"。
    3. 调用匹配器的 `find` 方法，对目标图像进行模板匹配。
    4. 从返回的结构中提取模板匹配的核心数据。
    5. 返回该模板在目标图中的匹配信息。
    """
    if not engine_template_cv_method_name:
        engine_template_cv_method_name = "cv2.TM_CCOEFF_NORMED"

    fi = FindIt(
        engine=["html"],
        engine_template_cv_method_name=engine_template_cv_method_name,
        **kwargs,
    )

    fi_template_name = "default"
    fi.load_template(fi_template_name, pic_object=template)

    result = fi.find(target_pic_name="", target_pic_object=target, **kwargs)
    logger.debug(f"findit result: {result}")

    return result["data"][fi_template_name]["TemplateEngine"]


def match_template_with_path(
    template: str, target: "np.ndarray", **kwargs
) -> dict[str, typing.Any]:
    """
    与上一个函数类似，但模板是从文件路径读取的

    Parameters
    ----------
    template : str
        模板图像的文件路径。
    target : np.ndarray
        目标图像，作为模板匹配的搜索对象。
    **kwargs :
        传递给匹配引擎的其他关键字参数。

    Returns
    -------
    dict[str, typing.Any]
        模板匹配结果字典，包含匹配位置、相似度得分等信息。

    Notes
    -----
    - 此函数是对 `match_template_with_object` 的封装，区别在于模板图像通过路径读取并转换为灰度图。
    - 模板图像必须存在于给定路径，若不存在将触发断言错误。

    Workflow
    --------
    1. 验证模板路径是否有效，确保文件存在。
    2. 使用 `imread()` 读取模板图像，并通过 `turn_grey()` 转换为灰度图。
    3. 将处理后的模板图像传递给 `match_template_with_object` 进行实际的模板匹配。
    4. 返回匹配结果字典。
    """
    assert os.path.isfile(template), f"image {template} not existed"
    template_object = turn_grey(imread(template))

    return match_template_with_object(template_object, target, **kwargs)


def show_progress(
        items: typing.Optional[typing.Union[typing.Sized, typing.Iterable]] = None,
        total: typing.Optional[int] = None,
        color: typing.Optional[int] = 245
) -> "tqdm":
    """
    用进度条包装可迭代对象或用总数初始化进度条。

    Parameters
    ----------
    items : typing.Optional[typing.Union[typing.Sized, typing.Iterable]]
        可选参数，如果提供，则将其包装为 tqdm 进度条。
    total : typing.Optional[int]
        可选参数，若未传入 items，则必须提供总进度值。
    color : typing.Optional[int]
        指定进度条的前景色（tqdm 标签与条形的 ANSI 颜色值），默认为 245。

    Returns
    -------
    tqdm
        tqdm 进度条对象，可用于包裹迭代对象或独立使用。

    Notes
    -----
    - 若传入 items（如 list、tuple、generator），则使用其长度作为进度条总数。
    - 若未提供 items，则必须提供 total 参数来设置进度条的总进度。
    - 控制台终端宽度由 `shutil.get_terminal_size()` 获取，若失败则使用默认宽度。
    - 支持通过 ANSI 256 色设置彩色输出，颜色编号参考: https://www.ditig.com/256-colors-cheat-sheet

    Workflow
    --------
    1. 构造彩色进度条格式 bar_format。
    2. 尝试获取终端列数，计算进度条宽度。
    3. 根据是否提供 items 或 total 创建 tqdm 对象。
    4. 若两者均未提供，则抛出 ValueError。
    """

    # 设置进度条的配色方案
    desc, color_begin, color_final = f"{const.DESC}", f"\033[1m\033[38;5;{color}m", "\033[0m"
    bar_format = f"{color_begin}{{l_bar}}%{{bar}}%|{{n_fmt:5}}/{{total_fmt:5}}{color_final}"

    # 确定进度条宽度的终端尺寸
    try:
        columns = shutil.get_terminal_size().columns
    except OSError:
        columns = 100  # 无法获取终端大小时的后备值

    progress_bar_length = int(columns * 0.6)

    # 根据输入参数配置 tqdm
    if items:
        tqdm_total = len(items) if hasattr(items, '__len__') else None
        return tqdm(
            items,
            total=total if total else tqdm_total,
            ncols=progress_bar_length,
            leave=True,
            bar_format=bar_format,
            desc=desc
        )
    elif total:
        return tqdm(
            total=total,
            ncols=progress_bar_length,
            leave=True,
            bar_format=bar_format,
            desc=desc
        )
    else:
        raise ValueError("Either 'items' or 'total' must be provided to show_progress.")


def draw_line(
        image_path: str,
        save_path: typing.Optional[str] = None
) -> None:
    """
    在图像上绘制参考线并保存。

    Parameters
    ----------
    image_path : str
        输入图像的路径。
    save_path : typing.Optional[str], optional
        图像绘制完成后的保存路径，若未提供，则覆盖原图。

    Returns
    -------
    None

    Notes
    -----
    - 在图像上绘制垂直与水平参考线以辅助定位。
    - 垂直线绘制于 20%、40%、60%、80% 的图像宽度位置。
    - 水平线绘制于每 5% 的高度处，并在左上角标注百分比标签。
    - 标签使用系统默认字体居中对齐绘制在对应横线的起始位置。

    Workflow
    --------
    1. 加载并转换图像为 RGB 模式。
    2. 获取图像尺寸，创建绘图对象。
    3. 循环绘制垂直线和水平线，并为每条水平线添加百分比文本。
    4. 若提供保存路径，则保存到指定位置；否则覆盖原始图像文件。
    """
    image = Image.open(image_path)
    image = image.convert("RGB")

    width, height = image.size

    draw = ImageDraw.Draw(image)

    font = ImageFont.load_default()  # 使用默认字体

    # 垂直线
    for i in range(1, 5):  # 20%, 40%, 60%, 80%
        x_line = int(width * (i * 0.2))
        draw.line([(x_line, 0), (x_line, height)], fill=(34, 220, 239), width=1)

    # 水平线
    for i in range(1, 20):  # 5%, 10%, ..., 95%
        y_line = int(height * (i * 0.05))
        text = f"{i * 5:02}%"
        bbox = draw.textbbox((0, 0), text, font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x_text_start = 3
        draw.line([(text_width + 5 + x_text_start, y_line), (width, y_line)], fill=(255, 182, 193), width=1)
        draw.text((x_text_start, y_line - text_height // 2), text, fill=(255, 182, 193), font=font)

    # 保存图像
    image.save(save_path) if save_path else image.save(image_path)


def generate_gradient_colors(start_hex: str, close_hex: str, steps: int) -> list[str]:
    """
    生成从起始颜色到结束颜色的线性渐变色列表。

    Parameters
    ----------
    start_hex : str
        起始颜色的十六进制表示，例如 "#FF0000"。

    close_hex : str
        结束颜色的十六进制表示，例如 "#00FF00"。

    steps : int
        生成的渐变步数，决定中间插值的精细程度。

    Returns
    -------
    list[str]
        按顺序排列的十六进制渐变色列表。

    Notes
    -----
    - 起始和结束颜色都包含在返回列表中。
    - 当 steps = 1 时，仅返回起始颜色。
    - 内部通过 RGB 空间线性插值得到渐变色，再转换回十六进制表示。
    """

    def hex_to_rgb(hex_color: str) -> tuple:
        """
        将十六进制颜色字符串转换为 RGB 三元组。
        """
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def rgb_to_hex(rgb: tuple) -> str:
        """
        将 RGB 三元组转换为十六进制颜色字符串。
        """
        return "#{:02X}{:02X}{:02X}".format(*rgb)

    start_rgb = hex_to_rgb(start_hex)
    close_rgb = hex_to_rgb(close_hex)

    gradient = []
    for step in range(steps):
        ratio = step / (steps - 1) if steps > 1 else 0  # 防止除0
        r = int(start_rgb[0] + (close_rgb[0] - start_rgb[0]) * ratio)
        g = int(start_rgb[1] + (close_rgb[1] - start_rgb[1]) * ratio)
        b = int(start_rgb[2] + (close_rgb[2] - start_rgb[2]) * ratio)
        gradient.append(rgb_to_hex((r, g, b)))

    return gradient


if __name__ == '__main__':
    pass
