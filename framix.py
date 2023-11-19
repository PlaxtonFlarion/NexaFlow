import os
import sys
import cv2
import time
import asyncio
from loguru import logger
from argparse import ArgumentParser
from nexaflow.hook import CropHook
from nexaflow.video import VideoObject
from nexaflow.skills.report import Report
from nexaflow.skills.switch import Switch
from nexaflow.cutter.cutter import VideoCutter
from nexaflow.classifier.keras_classifier import KerasClassifier

FORMAT: str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>"
logger.remove(0)
logger.add(sys.stderr, format=FORMAT, level="INFO")


async def help_document():
    print(
        f"""Command line framix [Option] [Parameter]\nV1.0.0 Released:[Nov 18, 2023] 

        [Option]    :  [Parameter]
        {'-' * 50}
        -i --input  :  [必选]视频文件路径
        -b --boost  :  [可选]自动跳帧模式
        -f --focus  :  [可选]自动调整视频
        {'-' * 50}
        """
    )
    for i in range(10):
        print(f"{10 - i:02} 秒后退出 {'++' * (10 - i)}")
        await asyncio.sleep(1)
    print(f"00 秒后退出")


async def parse_cmd():
    parser = ArgumentParser(description="Command Line Arguments Framix")

    parser.add_argument('-i', '--input', type=str, help='视频文件路径')
    parser.add_argument('-b', '--boost', action='store_true', help='跳帧模式')
    parser.add_argument('-f', '--focus', action='store_true', help='调整视频')
    parser.add_argument('-c', '--crop', type=str, help='(x, y, x_size, y_size)')
    parser.add_argument('-o', '--omit', type=str, help='(x, y, x_size, y_size)')

    return parser.parse_args()


async def initialization():
    if len(sys.argv) == 1:
        await help_document()
        sys.exit(1)

    cmd_lines = await parse_cmd()
    vision_path = cmd_lines.input

    if not vision_path:
        await help_document()
        sys.exit(1)
    if not os.path.exists(vision_path):
        os.makedirs(vision_path, exist_ok=True)

    boost = cmd_lines.boost
    focus = cmd_lines.focus
    crop = cmd_lines.crop
    omit = cmd_lines.omit

    return vision_path, boost, focus


async def analyzer(vision_path: str, boost: bool, focus: bool):
    screen_tag = os.path.basename(vision_path)
    screen_cap = cv2.VideoCapture(vision_path)
    if not screen_cap:
        logger.error(f"{screen_tag} 不是一个标准的mp4视频文件，或视频文件已损坏 ...")
        screen_cap.release()
        return
    screen_cap.release()

    step = 1
    block = 6
    threshold = 0.97
    offset = 3
    compress_rate = 0.5
    window_size = 1
    window_coefficient = 2
    target_size = (350, 700)

    job_path = os.path.dirname(os.path.abspath(__file__))
    if getattr(sys, 'frozen', False):
        model_path = os.path.join(getattr(sys, '_MEIPASS', job_path), "model", "model.h5")
        proto_path = os.path.join(os.path.dirname(sys.executable))
    else:
        model_path = os.path.join(job_path, "model", "model.h5")
        proto_path = os.path.join(job_path, "report")

    logger.info(f"{screen_tag} 可正常播放，准备加载视频 ...")

    if focus:
        change_record = vision_path.split('.')[0] + "_" + time.strftime("%Y%m%d%H%M%S") + ".mp4"
        logger.info(f"转换视频开启 ...")
        await Switch().video_change(vision_path, change_record)
        logger.info(f"视频转换完成: {change_record}")
        os.remove(vision_path)
        logger.info(f"移除旧的视频: {vision_path}")
    else:
        logger.info(f"转换视频关闭 ...")
        change_record = vision_path

    video = VideoObject(change_record)
    video.load_frames()

    cutter = VideoCutter(
        step=step,
        compress_rate=compress_rate,
        target_size=target_size
    )
    crop_hook = CropHook((0.9, 1), (0.1, 0))
    cutter.add_hook(crop_hook)

    res = cutter.cut(
        video=video,
        block=block,
        window_size=window_size,
        window_coefficient=window_coefficient
    )

    stable, unstable = res.get_range(
        threshold=threshold,
        offset=offset
    )

    cl = KerasClassifier(target_size=target_size)
    cl.load_model(model_path)
    classify = cl.classify(video=video, valid_range=stable, keep_data=True)

    start_frame = classify.get_not_stable_stage_range()[0][1]
    end_frame = classify.get_not_stable_stage_range()[-1][-1]

    time_cost = end_frame.timestamp - start_frame.timestamp
    before, after, final = f"{start_frame.timestamp:.5f}", f"{end_frame.timestamp:.5f}", f"{time_cost:.5f}"
    logger.info(f"图像分类结果: [开始帧: {before}] [结束帧: {after}] [总耗时: {final}]")

    logger.info("跳帧模式开启 ...") if boost else logger.info("跳帧模式关闭 ...")
    Report.draw(
        classifier_result=classify,
        proto_path=proto_path,
        target_size=target_size,
        boost_mode=boost
    )


async def main():
    vision_path, boost, focus = await initialization()
    await analyzer(vision_path, boost, focus)


if __name__ == '__main__':
    asyncio.run(main())
