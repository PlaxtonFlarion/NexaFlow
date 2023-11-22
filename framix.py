import os
import re
import sys
import cv2
import time
import signal
import random
import shutil
import asyncio
import tempfile
import warnings
from loguru import logger
from multiprocessing import Pool
from argparse import ArgumentParser
from PIL import Image, ImageDraw, ImageFont
from nexaflow import toolbox
from nexaflow.classifier.base import SingleClassifierResult
from nexaflow.constants import Constants
from nexaflow.hook import OmitHook
from nexaflow.skills.alynex import Alynex
from nexaflow.terminal import Terminal
from nexaflow.video import VideoObject
from nexaflow.skills.report import Report
from nexaflow.skills.switch import Switch
from nexaflow.cutter.cutter import VideoCutter
from nexaflow.classifier.keras_classifier import KerasClassifier

warnings.filterwarnings("ignore")


async def help_document():
    print(
        f"""Command line framix [Option] [Parameter]\nV1.0.0 Released:[Nov 18, 2023] 

        [Major]     :  [Parameter]         
        -f --flick  :  录制分析模式        
        -a --alone  :  单独录制视频
        -p --paint  :  绘制分割线条
        -i --input  :  视频文件路径
        [Minor]
        -o --omits  :  忽略分析区域 [组合使用]
        """
    )
    await asyncio.sleep(1)
    for i in range(10):
        print(f"{10 - i:02} 秒后退出 {'++' * (10 - i)}")
        await asyncio.sleep(1)
    print(f"00 秒后退出")


async def parse_cmd():
    parser = ArgumentParser(description="Command Line Arguments Framix")

    parser.add_argument('-f', '--flick', action='store_true', help='录制分析模式')
    parser.add_argument('-a', '--alone', action='store_true', help='单独录制视频')
    parser.add_argument('-p', '--paint', action='store_true', help='绘制分割线条')
    parser.add_argument('-i', '--input', type=str, help='视频文件路径')
    parser.add_argument('-o', '--omits', action='append', help='忽略分析区域 (x, y, x_size, y_size)')

    return parser.parse_args()


async def check_device():

    class Phone(object):

        def __init__(self, *args):
            self.serial, self.brand, self.version, *_ = args

        def __str__(self):
            return f"<Phone Brand=[{self.brand}] Version=[{self.version}] Serial=[{self.serial}]>"

        __repr__ = __str__

    async def check(serial):
        basic = f"adb -s {serial} wait-for-usb-device shell getprop"
        brand, version = await asyncio.gather(
            Terminal.cmd_line_shell(f"{basic} ro.product.brand"),
            Terminal.cmd_line_shell(f"{basic} ro.build.version.release")
        )
        return Phone(serial, brand, version)

    while True:
        devices = await Terminal.cmd_line_shell(f"adb devices")
        if len(device_list := [i.split()[0] for i in devices.split("\n")[1:]]) == 1:
            return await check(device_list[0])
        elif len(device_list) > 1:
            device_dict = {}
            tasks = [check(serial) for serial in device_list]
            result = await asyncio.gather(*tasks)
            for idx, cur in enumerate(result):
                device_dict.update({str(idx + 1): cur})
                print(f"{idx + 1} : {cur}")
            while True:
                try:
                    return device_dict[input("请输入编号选择一台设备: ")]
                except KeyError:
                    logger.error(f"没有该序号,请重新选择 ...")
                    await asyncio.sleep(0.1)
        else:
            logger.warning(f"设备未连接,等待设备连接 ...")
            await asyncio.sleep(3)

    # while True:
    #     device = await Terminal.cmd_line_shell(f"adb devices")
    #     if len(device_list := [i.split()[0] for i in device.split("\n")[1:]]) == 1:
    #         return device_list[0]
    #     elif len(device_list) > 1:
    #         device_dict = {}
    #         logger.warning(f"连接了多台设备 {device_list}")
    #         await asyncio.sleep(0.1)
    #         for index, device in enumerate(device_list):
    #             device_dict.update({str(index + 1): device})
    #             print(f"{index + 1} : <Device Serial=[{device}]>")
    #         while True:
    #             try:
    #                 return device_dict[input("请输入编号选择一台设备: ")]
    #             except KeyError:
    #                 logger.error(f"没有该序号,请重新选择 ...")
    #                 await asyncio.sleep(0.1)
    #     else:
    #         logger.warning(f"设备未连接,等待设备连接 ...")
    #         await asyncio.sleep(3)


async def analysis(r, alone, omits, model_path, template_path, major_path, total_path):

    cellphone = None
    done_event = asyncio.Event()
    stop_event = asyncio.Event()

    async def timepiece(amount):
        await asyncio.sleep(1)
        for i in range(amount):
            if stop_event.is_set() and i + 1 != amount:
                logger.warning("主动停止 ...")
                break
            logger.warning(f"剩余时间 -> {amount - i:02} 秒 {'----' * (amount - i)}")
            await asyncio.sleep(1)
        logger.warning(f"剩余时间 -> 00 秒")

    async def input_stream(transports):
        async for line in transports.stdout:
            logger.info(stream := line.decode(encoding="UTF-8", errors="ignore").strip())
            if "Killing the server" in stream or "Recording complete" in stream:
                stop_event.set()
                done_event.set()
                break

    async def error_stream(transports):
        async for line in transports.stderr:
            logger.error(stream := line.decode(encoding="UTF-8", errors="ignore").strip())
            if "Could not find any ADB device" in stream or "Server connection failed" in stream:
                stop_event.set()
                break

    async def start():
        # 只录制视频
        if alone:
            cmd = [
                "scrcpy", "-s", cellphone.serial, "--no-audio",
                "--video-bit-rate", "8M", "--max-fps", "60", "--record",
                temp_video := f"{os.path.join(r.query_path, 'screen')}_"
                              f"{time.strftime('%Y%m%d%H%M%S')}_"
                              f"{random.randint(100, 999)}.mkv"
            ]
            transports = await Terminal.cmd_link(*cmd)
            asyncio.create_task(input_stream(transports))
            asyncio.create_task(error_stream(transports))
            await asyncio.sleep(1)
            await timepiece(timer_mode)
            await Terminal.cmd_line("taskkill", "/im", "scrcpy.exe")
            if done_event.is_set():
                logger.success(f"视频录制成功: {temp_video}")
            else:
                os.remove(temp_video)
                logger.error("录制视频失败,请重新录制视频 ...")
        # 录制视频后立刻分析视频帧
        else:
            r.set_query(time.strftime('%Y%m%d%H%M%S'))
            cmd = [
                "scrcpy", "-s", cellphone.serial, "--no-audio",
                "--video-bit-rate", "8M", "--max-fps", "60", "--record",
                temp_video := f"{os.path.join(r.video_path, 'screen')}_"
                              f"{time.strftime('%Y%m%d%H%M%S')}_"
                              f"{random.randint(100, 999)}.mkv"
            ]
            transports = await Terminal.cmd_link(*cmd)
            asyncio.create_task(input_stream(transports))
            asyncio.create_task(error_stream(transports))
            await asyncio.sleep(1)
            await timepiece(timer_mode)
            await Terminal.cmd_line("taskkill", "/im", "scrcpy.exe")
            if done_event.is_set():
                await analyzer(
                    r, temp_video, omits=omits,
                    model_path=model_path, template_path=template_path
                )
            else:
                os.remove(temp_video)
                logger.error("录制视频失败,请重新录制视频 ...")

    cellphone = await check_device()
    r.set_title(f"Framix_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}")

    while True:
        try:
            await asyncio.sleep(0.1)
            if action := input(f"{f'{cellphone}' if cellphone else ''} *-* 按 Enter 开始 *-*  "):
                if "header" in action:
                    if match := re.search(r"(?<=header).*", action):
                        src_title = f"Framix_{time.strftime('%Y%m%d_%H%M%S')}"
                        if title := match.group().strip():
                            new_title = f"{src_title}_{title}_{random.randint(100, 999)}"
                        else:
                            new_title = f"{src_title}_{os.getpid()}_{random.randint(100, 999)}"
                        r.set_title(new_title)
                        logger.success(f"生成新标题: {new_title}")
                        await asyncio.to_thread(r.create_report, major_path)
                    continue
                if "serial" in action:
                    cellphone = await check_device()
                    continue
                timer_mode = 5 if int(action) < 5 else int(action)
            else:
                timer_mode = 5
        except ValueError:
            logger.warning(
                """可选输入项
                
                1. header your_report_title [生成一个新标题文件夹]
                2. serial                   [重新选择已连接的设备]
                3. \\d+                     [任意数字代表录制时长]
                """
            )
        else:
            await start()
            if not done_event.is_set():
                cellphone = await check_device()
        finally:
            done_event.clear()
            stop_event.clear()


async def painting():
    image_folder = "/sdcard/Pictures/Shots"
    image = f"{time.strftime('%Y%m%d%H%M%S')}_{random.randint(100, 999)}_" + "Shot.png"
    await Terminal.cmd_line_shell(f"adb wait-for-usb-device shell mkdir -p {image_folder}")
    await Terminal.cmd_line_shell(f"adb wait-for-usb-device shell screencap -p {image_folder}/{image}")

    with tempfile.TemporaryDirectory() as temp_dir:
        image_save_path = os.path.join(temp_dir, image)
        await Terminal.cmd_line_shell(f"adb wait-for-usb-device pull {image_folder}/{image} {image_save_path}")

        image = Image.open(image_save_path)
        image = image.convert("RGB")

        resized = image.resize((new_w := 350, new_h := 700))

        draw = ImageDraw.Draw(resized)
        font = ImageFont.load_default()

        for i in range(1, 5):
            x_line = int(new_w * (i * 0.2))
            draw.line([(x_line, 0), (x_line, new_h)], fill=(0, 255, 255), width=1)

        for i in range(1, 20):
            y_line = int(new_h * (i * 0.05))
            text = f"{i * 5:02}%"
            bbox = draw.textbbox((0, 0), text, font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x_text_start = 3
            draw.line([(text_width + 5 + x_text_start, y_line), (new_w, y_line)], fill=(255, 182, 193), width=1)
            draw.text((x_text_start, y_line - text_height // 2), text, fill=(255, 182, 193), font=font)

        resized.show()
    await Terminal.cmd_line_shell(f"adb wait-for-usb-device shell rm {image_folder}/{image}")


async def analyzer(r, vision_path: str, **kwargs):
    model_path = kwargs["model_path"]
    template_path = kwargs["template_path"]

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

    logger.info(f"{screen_tag} 可正常播放，准备加载视频 ...")
    change_record = vision_path.split('.')[0] + "_" + time.strftime("%Y%m%d%H%M%S") + ".mp4"
    await Switch().video_change(vision_path, change_record)
    logger.info(f"视频转换完成: {change_record}")
    os.remove(vision_path)
    logger.info(f"移除旧的视频: {vision_path}")

    video = VideoObject(change_record)
    video.load_frames()

    cutter = VideoCutter(
        step=step,
        compress_rate=compress_rate,
        target_size=target_size
    )

    if len(omits := kwargs["omits"]) > 0:
        for hook in omits:
            x, y, x_size, y_size = hook
            omit_hook = OmitHook((y_size, x_size), (y, x))
            cutter.add_hook(omit_hook)

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

    files = os.listdir(r.extra_path)
    files.sort(key=lambda n: int(n.split("(")[0]))
    total_images = len(files)
    interval = total_images // 11 if total_images > 12 else 1
    for index, file in enumerate(files):
        if index % interval != 0:
            os.remove(
                os.path.join(r.extra_path, file)
            )

    draws = os.listdir(r.extra_path)
    for draw in draws:
        toolbox.draw_line(
            os.path.join(r.extra_path, draw)
        )

    cl = KerasClassifier(target_size=target_size)
    cl.load_model(model_path)
    classify = cl.classify(video=video, valid_range=stable, keep_data=True)

    important_frames: list["SingleClassifierResult"] = classify.get_important_frame_list()

    pbar = toolbox.show_progress(classify.get_length(), 50, "Faster   ")
    frames_list = [previous := important_frames[0]]
    pbar.update(1)
    for current in important_frames[1:]:
        frames_list.append(current)
        pbar.update(1)
        frames_diff = current.frame_id - previous.frame_id
        if not previous.is_stable() and not current.is_stable() and frames_diff > 1:
            for specially in classify.data[previous.frame_id: current.frame_id - 1]:
                frames_list.append(specially)
                pbar.update(1)
        previous = current
    pbar.close()

    try:
        start_frame = classify.get_not_stable_stage_range()[0][1]
        end_frame = classify.get_not_stable_stage_range()[-1][-1]
    except AssertionError:
        start_frame = classify.get_important_frame_list()[0]
        end_frame = classify.get_important_frame_list()[-1]

    time_cost = end_frame.timestamp - start_frame.timestamp
    before, after, final = f"{start_frame.timestamp:.5f}", f"{end_frame.timestamp:.5f}", f"{time_cost:.5f}"
    logger.info(f"图像分类结果: [开始帧: {before}] [结束帧: {after}] [总耗时: {final}]")

    with open(template_path, encoding="utf-8") as t:
        template_file = t.read()
    original_inform = r.draw(
        classifier_result=classify,
        proto_path=r.proto_path,
        target_size=target_size,
        framix_template=template_file
    )
    result = {
        "query": r.query,
        "stage": {
            "start": start_frame.frame_id,
            "end": end_frame.frame_id,
            "cost": f"{time_cost:.5f}"
        },
        "frame": r.frame_path,
        "extra": r.extra_path,
        "proto": original_inform
    }
    r.load(result)


def quick_task(folder):
    alynex = Alynex()
    for video in alynex.only_video(folder):
        alynex.report.set_title(video.title)
        for path in video.sheet:
            alynex.report.set_query(os.path.basename(path).split(".")[0])
            shutil.copy(path, alynex.report.video_path)
            alynex.analyzer()
        alynex.report.create_report()
    alynex.report.create_total_report()
    return alynex.report.total_path


def quick_mode(data):
    start_time = time.time()

    with Pool(len(data)) as pool:
        results = pool.map(quick_task, data)

    Report.merge_report(results)
    print(f"Total Time Cost: {(time.time() - start_time):.2f} 秒")


async def main():
    if len(sys.argv) == 1:
        await help_document()
        sys.exit(1)

    if ".exe" in os.path.basename(sys.argv[0]):
        r = Report()
        job_path = os.path.dirname(sys.argv[0])
        model_path = os.path.join(job_path, "framix.source", "mold", "model.h5")
        total_path = os.path.join(job_path, "framix.source", "page", "overall.html")
        major_path = os.path.join(job_path, "framix.source", "page", "template.html")
        template_path = os.path.join(job_path, "framix.source", "page", "extra.html")
    else:
        r = Report()
        job_path = os.path.dirname(__file__)
        model_path = os.path.join(job_path, "model", "model.h5")
        total_path = os.path.join(Constants.NEXA, "template", "overall.html")
        major_path = os.path.join(Constants.NEXA, "template", "template.html")
        template_path = os.path.join(Constants.NEXA, "template", "extra.html")

    cmd_lines = await parse_cmd()
    omits = []
    if cmd_lines.omits and len(cmd_lines.omits) > 0:
        for hook in cmd_lines.omits:
            if len(match_list := re.findall(r"-?\d*\.?\d+", hook)) > 0:
                omits.append(
                    tuple(
                        [
                            float(num) if "." in num else int(num) for num in match_list
                        ]
                    )
                )

    if cmd_lines.flick:
        await analysis(
            r, cmd_lines.alone, omits, model_path, template_path, major_path, total_path
        )
        sys.exit(1)
    elif cmd_lines.alone:
        await analysis(
            r, cmd_lines.alone, omits, model_path, template_path, major_path, total_path
        )
        sys.exit(1)
    elif cmd_lines.paint:
        await painting()
        sys.exit(1)

    vision_path = cmd_lines.input
    if not vision_path:
        await help_document()
        sys.exit(1)

    r.set_title(f"Framix_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}")
    r.set_query(f"{time.strftime('%Y%m%d%H%M%S')}")
    await analyzer(
        r, vision_path, omits=omits,
        model_path=model_path, template_path=template_path
    )
    await asyncio.to_thread(r.create_report, major_path)
    await asyncio.to_thread(r.create_total_report, total_path)
    sys.exit(1)


if __name__ == '__main__':
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        os.kill(os.getpid(), signal.CTRL_BREAK_EVENT)
