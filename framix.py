import os
import re
import sys
import cv2
import time
import shutil
import random
import asyncio
import tempfile
import aiofiles
from typing import Union
from loguru import logger
from rich.table import Table
from multiprocessing import Pool
from rich.console import Console
from argparse import ArgumentParser
from PIL import Image, ImageDraw, ImageFont
from nexaflow import toolbox
from nexaflow.terminal import Terminal
from nexaflow.video import VideoObject
from nexaflow.skills.report import Report
from nexaflow.skills.switch import Switch
from nexaflow.constants import Constants
from nexaflow.cutter.cutter import VideoCutter
from nexaflow.hook import OmitHook, FrameSaveHook
from nexaflow.classifier.keras_classifier import KerasClassifier

target_size: tuple = (350, 700)
step: int = 1
block: int = 6
threshold: Union[int | float] = 0.97
offset: int = 3
compress_rate: float = 0.5
window_size: int = 1
window_coefficient: int = 2

console: Console = Console()


def help_document():
    table_major = Table(
        title="[bold #FF851B]NexaFlow Framix Main Command Line[/bold #FF851B]",
        header_style="bold #FF851B", title_justify="center",
        show_header=True, show_lines=True
    )
    table_major.add_column("主要命令", justify="center", width=12)
    table_major.add_column("参数类型", justify="center", width=12)
    table_major.add_column("传递次数", justify="center", width=8)
    table_major.add_column("附加命令", justify="center", width=8)
    table_major.add_column("功能说明", justify="center", width=22)

    table_major.add_row(
        "[bold #FFDC00]--flick[/bold #FFDC00]  [bold]-f[/bold]", "[bold #7FDBFF]布尔[/bold #7FDBFF]", "[bold #8A8A8A]一次[/bold #8A8A8A]", "[bold #D7FF00]支持[/bold #D7FF00]", "[bold #39CCCC]录制分析视频帧[/bold #39CCCC]"
    )
    table_major.add_row(
        "[bold #FFDC00]--alone[/bold #FFDC00]  [bold]-a[/bold]", "[bold #7FDBFF]布尔[/bold #7FDBFF]", "[bold #8A8A8A]一次[/bold #8A8A8A]", "", "[bold #39CCCC]录制视频[/bold #39CCCC]"
    )
    table_major.add_row(
        "[bold #FFDC00]--paint[/bold #FFDC00]  [bold]-p[/bold]", "[bold #7FDBFF]布尔[/bold #7FDBFF]", "[bold #8A8A8A]一次[/bold #8A8A8A]", "", "[bold #39CCCC]绘制分割线条[/bold #39CCCC]"
    )
    table_major.add_row(
        "[bold #FFDC00]--input[/bold #FFDC00]  [bold]-i[/bold]", "[bold #7FDBFF]视频文件[/bold #7FDBFF]", "[bold #FFAFAF]多次[/bold #FFAFAF]", "[bold #D7FF00]支持[/bold #D7FF00]", "[bold #39CCCC]分析单个视频[/bold #39CCCC]"
    )
    table_major.add_row(
        "[bold #FFDC00]--whole[/bold #FFDC00]  [bold]-w[/bold]", "[bold #7FDBFF]视频集合[/bold #7FDBFF]", "[bold #FFAFAF]多次[/bold #FFAFAF]", "[bold #D7FF00]支持[/bold #D7FF00]", "[bold #39CCCC]分析全部视频[/bold #39CCCC]"
    )
    table_major.add_row(
        "[bold #FFDC00]--merge[/bold #FFDC00]  [bold]-m[/bold]", "[bold #7FDBFF]报告集合[/bold #7FDBFF]", "[bold #FFAFAF]多次[/bold #FFAFAF]", "", "[bold #39CCCC]聚合报告[/bold #39CCCC]"
    )
    table_major.add_row(
        "[bold #FFDC00]--datum[/bold #FFDC00]  [bold]-d[/bold]", "[bold #7FDBFF]视频文件[/bold #7FDBFF]", "[bold #FFAFAF]多次[/bold #FFAFAF]", "", "[bold #39CCCC]归类图片文件[/bold #39CCCC]"
    )
    table_major.add_row(
        "[bold #FFDC00]--train[/bold #FFDC00]  [bold]-t[/bold]", "[bold #7FDBFF]图片文件[/bold #7FDBFF]", "[bold #FFAFAF]多次[/bold #FFAFAF]", "", "[bold #39CCCC]训练模型文件[/bold #39CCCC]"
    )

    table_minor = Table(
        title="[bold #FF851B]NexaFlow Framix Extra Command Line[/bold #FF851B]",
        header_style="bold #FF851B", title_justify="center",
        show_header=True, show_lines=True
    )
    table_minor.add_column("附加命令", justify="center", width=12)
    table_minor.add_column("参数类型", justify="center", width=12)
    table_minor.add_column("传递次数", justify="center", width=8)
    table_minor.add_column("默认状态", justify="center", width=8)
    table_minor.add_column("功能说明", justify="center", width=22)

    table_minor.add_row(
        "[bold #FFDC00]--boost[/bold #FFDC00]  [bold]-b[/bold]", "[bold #7FDBFF]布尔[/bold #7FDBFF]", "[bold #8A8A8A]一次[/bold #8A8A8A]", "[bold #AFAFD7]关闭[/bold #AFAFD7]", "[bold #39CCCC]快速模式[/bold #39CCCC]"
    )
    table_minor.add_row(
        "[bold #FFDC00]--color[/bold #FFDC00]  [bold]-c[/bold]", "[bold #7FDBFF]布尔[/bold #7FDBFF]", "[bold #8A8A8A]一次[/bold #8A8A8A]", "[bold #AFAFD7]关闭[/bold #AFAFD7]", "[bold #39CCCC]彩色模式[/bold #39CCCC]"
    )
    table_minor.add_row(
        "[bold #FFDC00]--omits[/bold #FFDC00]  [bold]-o[/bold]", "[bold #7FDBFF]坐标[/bold #7FDBFF]", "[bold #FFAFAF]多次[/bold #FFAFAF]", "", "[bold #39CCCC]忽略区域[/bold #39CCCC]"
    )

    nexaflow_logo = """[bold #D0D0D0]
    ███╗   ██╗███████╗██╗  ██╗ █████╗   ███████╗██╗      ██████╗ ██╗    ██╗
    ██╔██╗ ██║██╔════╝╚██╗██╔╝██╔══██╗  ██╔════╝██║     ██╔═══██╗██║    ██║
    ██║╚██╗██║█████╗   ╚███╔╝ ███████║  █████╗  ██║     ██║   ██║██║ █╗ ██║
    ██║ ╚████║██╔══╝   ██╔██╗ ██╔══██║  ██╔══╝  ██║     ██║   ██║██║███╗██║
    ██║  ╚███║███████╗██╔╝ ██╗██║  ██║  ██║     ███████╗╚██████╔╝╚███╔███╔╝
    ╚═╝   ╚══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝
    [/bold #D0D0D0]"""
    console.print(nexaflow_logo)
    console.print(table_major)
    console.print(table_minor)
    for _ in range(10):
        time.sleep(1)


def help_option():
    table = Table(show_header=True, header_style="bold #D7FF00")
    table.add_column("[bold]选项[/bold]", justify="center", width=22)
    table.add_column("[bold]参数[/bold]", justify="center", width=22)
    table.add_column("[bold]说明[/bold]", justify="center", width=22)
    table.add_row("[bold #FFAFAF]header[/bold #FFAFAF]", "[bold #AFD7FF]标题名[/bold #AFD7FF]", "[bold #DADADA]生成一个新标题文件夹[/bold #DADADA]")
    table.add_row("[bold #FFAFAF]serial[/bold #FFAFAF]", "[bold #AFD7FF]无参数[/bold #AFD7FF]", "[bold #DADADA]重新选择已连接的设备[/bold #DADADA]")
    table.add_row("[bold #FFAFAF]******[/bold #FFAFAF]", "[bold #AFD7FF]无参数[/bold #AFD7FF]", "[bold #DADADA]任意数字代表录制时长[/bold #DADADA]")
    console.print(table)


def parse_cmd():
    parser = ArgumentParser(description="Command Line Arguments Framix")

    parser.add_argument('-f', '--flick', action='store_true', help='录制分析视频帧')
    parser.add_argument('-a', '--alone', action='store_true', help='录制视频')
    parser.add_argument('-p', '--paint', action='store_true', help='绘制分割线条')
    parser.add_argument('-i', '--input', action='append', help='分析单个视频')
    parser.add_argument('-w', '--whole', action='append', help='分析全部视频')
    parser.add_argument('-m', '--merge', action='append', help='聚合报告')
    parser.add_argument('-d', '--datum', action='append', help='归类图片文件')
    parser.add_argument('-t', '--train', action='append', help='训练模型文件')

    parser.add_argument('-b', '--boost', action='store_true', help='快速模式')
    parser.add_argument('-c', '--color', action='store_true', help='彩色模式')
    parser.add_argument('-o', '--omits', action='append', help='忽略区域')

    return parser.parse_args()


def only_video(folder: str):

    class Entry(object):

        def __init__(self, title: str, place: str, sheet: list):
            self.title = title
            self.place = place
            self.sheet = sheet

    folder = folder if os.path.basename(folder) == "Nexa_Collection" else os.path.join(folder, "Nexa_Collection")

    return [
        Entry(
            os.path.basename(root), root,
            [os.path.join(root, f) for f in sorted(file)]
        )
        for root, _, file in os.walk(folder) if file
    ]


async def check_device():

    class Phone(object):

        def __init__(self, *args):
            self.serial, self.brand, self.version, *_ = args

        def __str__(self):
            return f"<Phone [{self.brand}] [OS{self.version}] [{self.serial}]>"

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
            logger.warning(f"已连接多台设备 {device_list}")
            device_dict = {}
            tasks = [check(serial) for serial in device_list]
            result = await asyncio.gather(*tasks)
            for idx, cur in enumerate(result):
                device_dict.update({str(idx + 1): cur})
                logger.warning(f"{idx + 1} -> {cur}")
            while True:
                try:
                    return device_dict[input("请输入编号选择一台设备: ")]
                except KeyError:
                    logger.error(f"没有该序号,请重新选择 ...")
                    await asyncio.sleep(0.1)
        else:
            logger.warning(f"设备未连接,等待设备连接 ...")
            await asyncio.sleep(3)


async def analysis(reporter, alone):

    cellphone = None
    head_event = asyncio.Event()
    done_event = asyncio.Event()
    stop_event = asyncio.Event()
    fail_event = asyncio.Event()

    async def timepiece(amount):
        while True:
            if head_event.is_set():
                for i in range(amount):
                    if stop_event.is_set() and i + 1 != amount:
                        logger.warning("主动停止 ...")
                        break
                    elif fail_event.is_set():
                        logger.warning("意外停止 ...")
                        break
                    if amount - i <= 10:
                        logger.warning(f"剩余时间 -> {amount - i:02} 秒 {'----' * (amount - i)}")
                    else:
                        logger.warning(f"剩余时间 -> {amount - i:02} 秒 {'----' * 10} ...")
                    await asyncio.sleep(1)
                logger.warning(f"剩余时间 -> 00 秒")
            elif fail_event.is_set():
                logger.warning("意外停止 ...")
                break
            break

    async def input_stream(transports):
        async for line in transports.stdout:
            logger.info(stream := line.decode(encoding="UTF-8", errors="ignore").strip())
            if "Recording started" in stream:
                head_event.set()
            elif "Recording complete" in stream:
                stop_event.set()
                done_event.set()
                break

    async def error_stream(transports):
        async for line in transports.stderr:
            logger.info(stream := line.decode(encoding="UTF-8", errors="ignore").strip())
            if "Could not find" in stream or "connection failed" in stream or "Recorder error" in stream:
                fail_event.set()
                break

    async def start():
        if alone:
            if not os.path.exists(reporter.query_path):
                os.makedirs(reporter.query_path)
            cmd = [
                "scrcpy", "-s", cellphone.serial, "--no-audio",
                "--video-bit-rate", "8M", "--max-fps", "60", "--record",
                temp_video := f"{os.path.join(reporter.query_path, 'screen')}_"
                              f"{time.strftime('%Y%m%d%H%M%S')}_"
                              f"{random.randint(100, 999)}.mkv"
            ]
            transports = await Terminal.cmd_link(*cmd)
            asyncio.create_task(input_stream(transports))
            asyncio.create_task(error_stream(transports))
            await asyncio.sleep(1)
            await timepiece(timer_mode)
            await Terminal.cmd_line("taskkill", "/im", "scrcpy.exe")
            for _ in range(10):
                if done_event.is_set():
                    logger.success(f"视频录制成功: {temp_video}")
                    return
                elif fail_event.is_set():
                    break
                await asyncio.sleep(0.2)
            logger.error("录制视频失败,请重新录制视频 ...")

        else:
            reporter.query = f"{random.randint(10, 99)}"
            cmd = [
                "scrcpy", "-s", cellphone.serial, "--no-audio",
                "--video-bit-rate", "8M", "--max-fps", "60", "--record",
                temp_video := f"{os.path.join(reporter.video_path, 'screen')}_"
                              f"{time.strftime('%Y%m%d%H%M%S')}_"
                              f"{random.randint(100, 999)}.mkv"
            ]
            transports = await Terminal.cmd_link(*cmd)
            asyncio.create_task(input_stream(transports))
            asyncio.create_task(error_stream(transports))
            await asyncio.sleep(1)
            await timepiece(timer_mode)
            await Terminal.cmd_line("taskkill", "/im", "scrcpy.exe")
            for _ in range(10):
                if done_event.is_set():
                    await analyzer(
                        reporter, temp_video, boost=_boost, color=_color, omits=_omits, model_path=_model_path, proto_path=_proto_path
                    )
                    return
                elif fail_event.is_set():
                    break
                await asyncio.sleep(0.2)
            shutil.rmtree(reporter.query_path)
            logger.error("录制视频失败,请重新录制视频 ...")

    cellphone = await check_device()
    if alone:
        reporter.title = f"Record_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    else:
        reporter.title = f"Framix_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

    while True:
        try:
            await asyncio.sleep(0.1)
            if action := input(f"{f'{cellphone}' if cellphone else ''}  *-* 按 Enter 开始 *-*  "):
                if "header" in action.strip():
                    if match := re.search(r"(?<=header\s).*", action):
                        if match.group().strip():
                            src_title = f"Record_{time.strftime('%Y%m%d_%H%M%S')}" if alone else f"Framix_{time.strftime('%Y%m%d_%H%M%S')}"
                            if title := match.group().strip():
                                new_title = f"{src_title}_{title}"
                            else:
                                new_title = f"{src_title}_{random.randint(10000, 99999)}"
                            logger.success("新标题设置成功 ...")
                            reporter.title = new_title
                        else:
                            logger.warning("未设置新标题 ...")
                            help_option()
                    else:
                        logger.warning("未设置新标题 ...")
                        help_option()
                    continue
                elif action.strip() == "serial" and len(action.strip()) == 6:
                    cellphone = await check_device()
                    continue
                timer_mode = 5 if int(action) < 5 else int(action)
            else:
                timer_mode = 5
        except ValueError:
            help_option()
        else:
            await start()
            if not done_event.is_set():
                cellphone = await check_device()
        finally:
            head_event.clear()
            done_event.clear()
            stop_event.clear()
            fail_event.clear()


async def analyzer(reporter: Report, vision_path: str, **kwargs):
    boost = kwargs.get("boost", True)
    color = kwargs.get("color", True)
    omits = kwargs.get("omits", [])
    model_path = kwargs["model_path"]
    proto_path = kwargs["proto_path"]

    if not os.path.isfile(vision_path):
        return

    screen_tag = os.path.basename(vision_path)
    screen_cap = cv2.VideoCapture(vision_path)
    if not screen_cap:
        logger.error(f"{screen_tag} 不是一个标准的mp4视频文件，或视频文件已损坏 ...")
        screen_cap.release()
        return
    screen_cap.release()

    logger.info(f"{screen_tag} 可正常播放，准备加载视频 ...")
    change_record = vision_path.split('.')[0] + ".mp4"
    if change_record == vision_path:
        change_record = os.path.join(os.path.dirname(vision_path), f"screen_{random.randint(100, 999)}.mp4")
    await Switch().video_change(vision_path, change_record)
    logger.info(f"视频转换完成: {change_record}")
    os.remove(vision_path)
    logger.info(f"移除旧的视频: {vision_path}")

    video = VideoObject(change_record)
    task, hued = video.load_frames(True)

    cutter = VideoCutter(
        step=step,
        compress_rate=compress_rate,
        target_size=target_size
    )

    if len(omits) > 0:
        for omit in omits:
            x, y, x_size, y_size = omit
            omit_hook = OmitHook((y_size, x_size), (y, x))
            cutter.add_hook(omit_hook)
    save_hook = FrameSaveHook(reporter.extra_path)
    cutter.add_hook(save_hook)

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

    files = os.listdir(reporter.extra_path)
    files.sort(key=lambda n: int(n.split("(")[0]))
    total_images = len(files)
    interval = total_images // 11 if total_images > 12 else 1
    for index, file in enumerate(files):
        if index % interval != 0:
            os.remove(
                os.path.join(reporter.extra_path, file)
            )

    draws = os.listdir(reporter.extra_path)
    for draw in draws:
        toolbox.draw_line(
            os.path.join(reporter.extra_path, draw)
        )

    cl = KerasClassifier(target_size=target_size)
    cl.load_model(model_path)
    classify = cl.classify(video=video, valid_range=stable, keep_data=True)

    important_frames = classify.get_important_frame_list()

    pbar = toolbox.show_progress(classify.get_length(), 50, "Faster")
    frames_list = []
    if boost:
        frames_list.append(previous := important_frames[0])
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
    else:
        for current in classify.data:
            frames_list.append(current)
            pbar.update(1)
        pbar.close()

    if color:
        video.hued_data = tuple(hued.result())
        logger.info(f"彩色帧已加载: {video.frame_details(video.hued_data)}")
        task.shutdown()
        frames = [video.hued_data[frame.frame_id - 1] for frame in frames_list]
    else:
        frames = [frame for frame in frames_list]

    try:
        start_frame = classify.get_not_stable_stage_range()[0][1]
        end_frame = classify.get_not_stable_stage_range()[-1][-1]
    except AssertionError:
        start_frame = classify.get_important_frame_list()[0]
        end_frame = classify.get_important_frame_list()[-1]

    if start_frame == end_frame:
        start_frame = classify.data[0]
        end_frame = classify.data[-1]

    time_cost = end_frame.timestamp - start_frame.timestamp
    before, after, final = f"{start_frame.timestamp:.5f}", f"{end_frame.timestamp:.5f}", f"{time_cost:.5f}"
    logger.info(f"图像分类结果: [开始帧: {before}] [结束帧: {after}] [总耗时: {final}]")

    async def frame_forge(frame):
        short_timestamp = format(round(frame.timestamp, 5), ".5f")
        pic_name = f"{frame.frame_id}_{short_timestamp}.png"
        pic_path = os.path.join(reporter.frame_path, pic_name)
        _, codec = cv2.imencode(".png", frame.data)
        async with aiofiles.open(pic_path, "wb") as f:
            await f.write(codec.tobytes())

    await asyncio.gather(*(frame_forge(frame) for frame in frames))

    with open(proto_path, encoding="utf-8") as t:
        proto_file = t.read()
        original_inform = reporter.draw(
            classifier_result=classify,
            proto_path=reporter.proto_path,
            target_size=target_size,
            framix_template=proto_file
        )

    result = {
        "total_path": reporter.total_path,
        "title": reporter.title,
        "query_path": reporter.query_path,
        "query": reporter.query,
        "stage": {
            "start": start_frame.frame_id,
            "end": end_frame.frame_id,
            "cost": f"{time_cost:.5f}"
        },
        "frame": reporter.frame_path,
        "extra": reporter.extra_path,
        "proto": original_inform,
    }
    logger.debug(f"Restore: {result}")
    reporter.load(result)


async def painting():
    cellphone = await check_device()
    image_folder = "/sdcard/Pictures/Shots"
    image = f"{time.strftime('%Y%m%d%H%M%S')}_{random.randint(100, 999)}_" + "Shot.png"
    await Terminal.cmd_line_shell(f"adb -s {cellphone.serial} wait-for-usb-device shell mkdir -p {image_folder}")
    await Terminal.cmd_line_shell(f"adb -s {cellphone.serial} wait-for-usb-device shell screencap -p {image_folder}/{image}")

    with tempfile.TemporaryDirectory() as temp_dir:
        image_save_path = os.path.join(temp_dir, image)
        await Terminal.cmd_line_shell(f"adb -s {cellphone.serial} wait-for-usb-device pull {image_folder}/{image} {image_save_path}")

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
    await Terminal.cmd_line_shell(f"adb -s {cellphone.serial} wait-for-usb-device shell rm {image_folder}/{image}")


def single_video_task(input_video, *args):
    Constants.initial_logger()
    boost, color, omits, model_path, total_path, major_path, proto_path = args
    new_total_path = os.path.join(os.path.dirname(
        sys.argv[0]), f"Nexa_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}", "Nexa_Collection"
    ) if ".exe" in os.path.basename(sys.argv[0]) else None
    reporter = Report(new_total_path)
    reporter.title = f"Framix_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    reporter.query = f"{random.randint(10, 99)}"
    new_video_path = os.path.join(reporter.video_path, os.path.basename(input_video))
    shutil.move(input_video, new_video_path)
    looper = asyncio.get_event_loop()
    looper.run_until_complete(
        analyzer(
            reporter, new_video_path, boost=boost, color=color, omits=omits, model_path=model_path, proto_path=proto_path
        )
    )
    looper.run_until_complete(
        reporter.ask_create_total_report(
            os.path.dirname(reporter.total_path), total_path, major_path
        )
    )


def multiple_folder_task(folder, *args):
    Constants.initial_logger()
    boost, color, omits, model_path, total_path, major_path, proto_path = args
    new_total_path = os.path.join(os.path.dirname(
        sys.argv[0]), f"Nexa_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}", "Nexa_Collection"
    ) if ".exe" in os.path.basename(sys.argv[0]) else None
    reporter = Report(new_total_path)
    looper = asyncio.get_event_loop()
    for video in only_video(folder):
        reporter.title = video.title
        for path in video.sheet:
            reporter.query = os.path.basename(path).split(".")[0]
            shutil.copy(path, reporter.video_path)
            looper.run_until_complete(
                analyzer(
                    reporter, path, boost=boost, color=color, omits=omits, model_path=model_path, proto_path=proto_path
                )
            )
    looper.run_until_complete(
        reporter.ask_create_total_report(
            os.path.dirname(reporter.total_path), total_path, major_path
        )
    )
    return reporter.total_path


def train_model(video_file):
    Constants.initial_logger("DEBUG")
    reporter = Report(write_log=False)
    reporter.title = f"Model_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}"
    if not os.path.exists(reporter.query_path):
        os.makedirs(reporter.query_path)

    video = VideoObject(
        video_file,
        fps=60
    )
    video.load_frames()
    cutter = VideoCutter(
        step=step,
        compress_rate=compress_rate,
        target_size=target_size
    )
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
    res.pick_and_save(
        range_list=stable,
        frame_count=20,
        to_dir=reporter.query_path,
        meaningful_name=True
    )


def build_model(src):
    Constants.initial_logger("DEBUG")
    if not re.search(r"Model_\d+_\d+", basic_path := os.path.basename(src)):
        if build_path := re.search(r"(?<=Nexa_)\d+_\d+", basic_path):
            src = os.path.join(src, "Nexa_Collection", f"Model_{build_path.group()}")
        else:
            return

    new_model_path = os.path.join(src, f"Create_Model_{time.strftime('%Y%m%d%H%M%S')}")
    new_model_name = f"Keras_Model_{random.randint(10000, 99999)}.h5"
    final_model = os.path.join(new_model_path, new_model_name)
    if not os.path.exists(new_model_path):
        os.makedirs(new_model_path)
    cl = KerasClassifier(target_size=target_size)
    cl.train(src, final_model)
    cl.save_model(final_model, overwrite=True)


async def main():
    Constants.initial_logger()
    if cmd_lines.flick or cmd_lines.alone:
        if cmd_lines.alone:
            await analysis(Report(initial_total_path, write_log=False), cmd_lines.alone)
        else:
            await analysis(Report(initial_total_path), cmd_lines.alone)
    elif cmd_lines.paint:
        await painting()
    elif cmd_lines.merge and len(cmd_lines.merge) > 0:
        tasks = [Report.ask_create_total_report(merge, _total_path, _major_path) for merge in cmd_lines.merge]
        await asyncio.gather(*tasks)
    else:
        help_document()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        help_document()
        sys.exit(1)

    if ".exe" in os.path.basename(sys.argv[0]):
        job_path = os.path.dirname(sys.argv[0])
        _model_path = os.path.join(job_path, "framix.source", "mold", "model.h5")
        _total_path = os.path.join(job_path, "framix.source", "page")
        _major_path = os.path.join(job_path, "framix.source", "page")
        _proto_path = os.path.join(job_path, "framix.source", "page", "extra.html")
        initial_total_path = os.path.join(
            os.path.dirname(sys.argv[0]), f"Nexa_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}", "Collection"
        )
    else:
        job_path = os.path.dirname(__file__)
        _model_path = os.path.join(job_path, "model", "model.h5")
        _total_path = os.path.join(Constants.NEXA, "template")
        _major_path = os.path.join(Constants.NEXA, "template")
        _proto_path = os.path.join(Constants.NEXA, "template", "extra.html")
        initial_total_path = None

    cmd_lines = parse_cmd()
    _omits = []
    if cmd_lines.omits and len(cmd_lines.omits) > 0:
        for hook in cmd_lines.omits:
            if len(match_list := re.findall(r"-?\d*\.?\d+", hook)) > 0:
                _omits.append(
                    tuple(
                        [
                            float(num) if "." in num else int(num) for num in match_list
                        ]
                    )
                )

    _boost, _color = cmd_lines.boost, cmd_lines.color

    if cmd_lines.whole and len(cmd_lines.whole) > 0:
        members = len(cmd_lines.whole)
        if members == 1:
            multiple_folder_task(
                cmd_lines.whole[0], _boost, _color, _omits, _model_path, _total_path, _major_path, _proto_path
            )
        else:
            Constants.initial_logger()
            with Pool(members if members <= 6 else 6) as pool:
                results = pool.starmap(
                    multiple_folder_task,
                    [(i, _boost, _color, _omits, _model_path, _total_path, _major_path, _proto_path) for i in cmd_lines.whole]
                )
            Report.merge_report(results)
        sys.exit(1)
    elif cmd_lines.input and len(cmd_lines.input) > 0:
        members = len(cmd_lines.input)
        if members == 1:
            single_video_task(
                cmd_lines.input[0], _boost, _color, _omits, _model_path, _total_path, _major_path, _proto_path
            )
        else:
            with Pool(members if members <= 6 else 6) as pool:
                pool.starmap(
                    single_video_task,
                    [(i, _boost, _color, _omits, _model_path, _total_path, _major_path, _proto_path) for i in cmd_lines.input]
                )
        sys.exit(1)
    elif cmd_lines.datum and len(cmd_lines.datum) > 0:
        members = len(cmd_lines.datum)
        if members == 1:
            train_model(cmd_lines.datum[0])
        else:
            with Pool(members if members <= 6 else 6) as pool:
                pool.map(train_model, cmd_lines.datum)
        sys.exit(1)
    elif cmd_lines.train and len(cmd_lines.train) > 0:
        members = len(cmd_lines.train)
        if members == 1:
            build_model(cmd_lines.train[0])
        else:
            with Pool(members if members <= 6 else 6) as pool:
                pool.map(build_model, cmd_lines.train)
        sys.exit(1)
    else:
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
            sys.exit(1)
        except KeyboardInterrupt:
            sys.exit(1)
