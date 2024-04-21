import os
import cv2
import sys
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
from PIL import Image, ImageDraw, ImageFont
from skimage.feature import hog, local_binary_pattern
from skimage.metrics import normalized_root_mse as compare_nrmse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as origin_compare_ssim

from nexaflow import const


@contextlib.contextmanager
def video_capture(video_path: str):
    video_cap = cv2.VideoCapture(video_path)
    try:
        yield video_cap
    finally:
        video_cap.release()


def video_jump(video_cap: cv2.VideoCapture, frame_id: int):
    """
    跳转到指定的帧
    """
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1 - 1)
    video_cap.read()


def compare_ssim(pic1: np.ndarray, pic2: np.ndarray) -> float:
    """
    计算两张图片的结构相似度（SSIM）
    """
    pic1, pic2 = [turn_grey(i) for i in [pic1, pic2]]
    return origin_compare_ssim(pic1, pic2)


def multi_compare_ssim(
    pic1_list: typing.List, pic2_list: typing.List, hooks: typing.List = None
) -> typing.List[float]:
    """
    对多个图像进行SSIM比较
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


def get_current_frame_id(video_cap: cv2.VideoCapture) -> int:
    """
    获取当前帧的ID
    """
    return int(video_cap.get(cv2.CAP_PROP_POS_FRAMES))


def get_current_frame_time(video_cap: cv2.VideoCapture) -> float:
    """
    获取当前帧的时间戳
    """
    return video_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000


def imread(img_path: str, *_, **__) -> np.ndarray:
    """
    包装了cv2.imread函数，用于读取图像
    """
    assert os.path.isfile(img_path), f"file {img_path} is not existed"
    return cv2.imread(img_path, *_, **__)


def get_frame_time(
    video_cap: cv2.VideoCapture, frame_id: int, recover: bool = None
) -> float:
    """
    获取指定帧的时间戳
    """
    cur = get_current_frame_id(video_cap)
    video_jump(video_cap, frame_id)
    result = get_current_frame_time(video_cap)
    logger.debug(f"frame {frame_id} -> {result}")

    if recover:
        video_jump(video_cap, cur + 1)
    return result


def get_frame_count(video_cap: cv2.VideoCapture) -> int:
    """
    获取视频总帧数
    """
    return int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))


def get_frame_size(video_cap: cv2.VideoCapture) -> typing.Tuple[int, int]:
    """
    获取帧的尺寸
    """
    h = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    return int(w), int(h)


def get_frame(
    video_cap: cv2.VideoCapture, frame_id: int, recover: bool = None
) -> np.ndarray:
    """
    获取指定帧的图像数据
    """
    cur = get_current_frame_id(video_cap)
    video_jump(video_cap, frame_id)
    ret, frame = video_cap.read()
    assert ret, f"read frame failed, frame id: {frame_id}"

    if recover:
        video_jump(video_cap, cur + 1)
    return frame


def turn_grey(old: np.ndarray) -> np.ndarray:
    """
    将图像转换为灰度图
    """
    try:
        return cv2.cvtColor(old, cv2.COLOR_RGB2GRAY)
    except cv2.error:
        return old


def turn_binary(old: np.ndarray) -> np.ndarray:
    """
    将图像转为二值图
    """
    grey = turn_grey(old).astype("uint8")
    return cv2.adaptiveThreshold(
        grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )


def turn_hog_desc(old: np.ndarray) -> np.ndarray:
    """
    计算图像的HOG（Histogram of Oriented Gradients）描述子
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


def turn_lbp_desc(old: np.ndarray, radius: int = None) -> np.ndarray:
    """
    计算图像的LBP（Local Binary Pattern）描述子
    """
    if not radius:
        radius = 3
    n_points = 8 * radius

    grey = turn_grey(old)
    lbp = local_binary_pattern(grey, n_points, radius, method="default")
    return lbp


def turn_blur(old: np.ndarray) -> np.ndarray:
    """
    使用高斯模糊处理图像
    """
    return cv2.GaussianBlur(old, (7, 7), 0)


def sharpen_frame(old: np.ndarray) -> np.ndarray:
    """
    锐化图像
    """
    blur = turn_blur(old)
    smooth = cv2.addWeighted(blur, 1.5, old, -0.5, 0)
    canny = cv2.Canny(smooth, 50, 150)
    return canny


def calc_mse(pic1: np.ndarray, pic2: np.ndarray) -> float:
    """
    计算两图像的均方误差（MSE）
    """
    return compare_nrmse(pic1, pic2)


def calc_psnr(pic1: np.ndarray, pic2: np.ndarray) -> float:
    """
    计算两图像的峰值信噪比（PSNR）
    """
    psnr = compare_psnr(pic1, pic2)
    if math.isinf(psnr):
        psnr = 100.0

    return psnr / 100


def compress_frame(
    old: np.ndarray,
    compress_rate: float = None,
    target_size: typing.Tuple[int, int] = None,
    not_grey: bool = None,
    interpolation: int = None,
    *_,
    **__,
) -> np.ndarray:
    """
    压缩帧
    """

    target = turn_grey(old) if not not_grey else old

    if not interpolation:
        interpolation = cv2.INTER_AREA

    if target_size:
        return cv2.resize(target, target_size, interpolation=interpolation)

    if not compress_rate:
        return target
    return cv2.resize(
        target, (0, 0), fx=compress_rate, fy=compress_rate, interpolation=interpolation
    )


def get_timestamp_str() -> str:
    """
    获取当前时间戳字符串
    """
    time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    salt = random.randint(10, 99)
    return f"{time_str}{salt}"


def np2b64str(frame: np.ndarray) -> str:
    """
    将Numpy数组（图像）转为Base64字符串
    """
    buffer = cv2.imencode(".png", frame)[1]
    return b64encode(buffer).decode()


def fps_convert(
    target_fps: int, source_path: str, target_path: str, ffmpeg_exe: str = None
) -> int:
    """
    转换视频的帧率
    """

    if not ffmpeg_exe:
        ffmpeg_exe = r"ffmpeg"
    command: typing.List[str] = [
        ffmpeg_exe, "-i", source_path,
        "-r", str(target_fps), target_path,
    ]
    logger.debug(f"convert video: {command}")
    return subprocess.check_call(command)


def match_template_with_object(
    template: np.ndarray,
    target: np.ndarray,
    engine_template_cv_method_name: str = None,
    **kwargs,
) -> typing.Dict[str, typing.Any]:
    """
    使用模板匹配找出目标图像中与模板相似的部分
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
    template: str, target: np.ndarray, **kwargs
) -> typing.Dict[str, typing.Any]:
    """
    与上一个函数类似，但模板是从文件路径读取的
    """
    assert os.path.isfile(template), f"image {template} not existed"
    template_object = turn_grey(imread(template))
    return match_template_with_object(template_object, target, **kwargs)


def show_progress(total: int, color: int) -> tqdm:
    """https://www.ditig.com/256-colors-cheat-sheet"""
    colors = {"start": f"\033[1m\033[38;5;{color}m", "end": "\033[0m"}
    bar_format = "{l_bar}%{bar}%|{n_fmt:5}/{total_fmt:5}"
    colored_bar_format = f"{colors['start']}{bar_format}{colors['end']}"
    if sys.stdout.isatty():
        try:
            columns, _ = shutil.get_terminal_size()
        except OSError:
            columns = 150
    else:
        columns = 150
    progress_bar_length = int(columns * 0.8)
    progress_bar = tqdm(
        total=total,
        position=0, ncols=progress_bar_length, leave=True, bar_format=colored_bar_format,
        desc=f"{const.DESC} : Analyzer "
    )
    return progress_bar


def draw_line(image_path: str, save_path: str = None) -> None:
    # 打开图像
    image = Image.open(image_path)
    image = image.convert("RGB")
    # 获取图像尺寸
    width, height = image.size
    # 创建一个ImageDraw对象
    draw = ImageDraw.Draw(image)
    # 使用默认字体
    font = ImageFont.load_default()

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


if __name__ == '__main__':
    pass
