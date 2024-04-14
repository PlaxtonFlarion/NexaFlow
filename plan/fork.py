import os
import sys
import cv2
import time
import signal
import typing
import random
import pygame
import asyncio
import aiofiles
from pathlib import Path
from loguru import logger
from engine.terminal import Terminal
from engine.activate import Review
from engine.switch import Switch
from frameflow.skills.parser import Parser
from frameflow.skills.show import Show
from nexaflow import toolbox, const
from nexaflow.cutter.cutter import VideoCutter
from nexaflow.video import VideoObject
from nexaflow.classifier.keras_classifier import KerasClassifier
from nexaflow.hook import CompressHook, FrameSaveHook
from nexaflow.hook import PaintCropHook, PaintOmitHook
from nexaflow.report import Report
from concurrent.futures import ThreadPoolExecutor
from frameflow.skills.device import Device
from frameflow.skills.manage import Manage

_platform = sys.platform


class Med(object):

    def __init__(self):
        self.__volume = 1.0
        self.__scrcpy = "scrcpy"
        self.__record = asyncio.Event()

    async def audio_player(self, audio_file: str):
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.set_volume(self.__volume)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    def start_record(self, serial, dst, events):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_complete = loop.run_until_complete(
            self._start_record(serial, dst, events)
        )
        loop.close()
        return loop_complete

    async def _start_record(self, serial, dst, events):

        async def input_stream():
            async for line in transports.stdout:
                logger.info(stream := line.decode(encoding="UTF-8", errors="ignore").strip())
                if "Recording started" in stream:
                    events["head_event"].set()
                elif "Recording complete" in stream:
                    events["stop_event"].set()
                    events["done_event"].set()
                    break

        async def error_stream():
            async for line in transports.stderr:
                logger.info(stream := line.decode(encoding="UTF-8", errors="ignore").strip())
                if "Could not find" in stream or "connection failed" in stream or "Recorder error" in stream:
                    events["fail_event"].set()
                    break

        flag_video = f"{time.strftime('%Y%m%d%H%M%S')}_{random.randint(100, 999)}.mkv"
        cmd = [self.__scrcpy, "-s", serial, "--no-audio", "--video-bit-rate=8M", "--max-fps=60"]
        cmd += ["--no-video", video_temp := f"{os.path.join(dst, 'screen')}_{flag_video}"]

        transports = await Terminal.cmd_link(*cmd)
        asyncio.create_task(input_stream())
        asyncio.create_task(error_stream())
        await asyncio.sleep(1)

        return video_temp, transports

    @staticmethod
    async def close_record(transports, event):
        transports.send_signal(signal.CTRL_C_EVENT)
        for _, v in event.items():
            if isinstance(v, asyncio.Event):
                v.clear()

        try:
            await Terminal.cmd_line("taskkill", "/im", "scrcpy.exe")
        except KeyboardInterrupt:
            logger.info("Stop with Ctrl_C_Event ...")


class Aly(object):

    def __init__(self, model_place, model_shape, model_aisle):
        self.kc = KerasClassifier(data_size=model_shape, aisle=model_aisle)
        self.kc.load_model(model_place)

    async def analyzer(self, vision, deploy, *args, **kwargs) -> typing.Optional["Review"]:

        frame_path, extra_path, fmp, fpb, *_ = args

        if deploy:
            boost = deploy.boost
            color = deploy.color

            shape = deploy.shape
            scale = deploy.scale
            start = deploy.start
            close = deploy.close
            limit = deploy.limit

            begin = deploy.begin
            final = deploy.final

            crops = deploy.crops
            omits = deploy.omits

            frate = deploy.frate
            thres = deploy.thres
            shift = deploy.shift
            block = deploy.block

        else:
            boost = kwargs.get("boost", const.BOOST)
            color = kwargs.get("color", const.COLOR)

            shape = kwargs.get("shape", const.SHAPE)
            scale = kwargs.get("scale", const.SCALE)
            start = kwargs.get("start", const.START)
            close = kwargs.get("close", const.CLOSE)
            limit = kwargs.get("limit", const.LIMIT)

            begin = kwargs.get("begin", const.BEGIN)
            final = kwargs.get("final", const.FINAL)

            crops = kwargs.get("crops", const.CROPS)
            omits = kwargs.get("omits", const.OMITS)

            frate = kwargs.get("frate", const.FRATE)
            thres = kwargs.get("thres", const.THRES)
            shift = kwargs.get("shift", const.SHIFT)
            block = kwargs.get("block", const.BLOCK)

        # boost = deploy.boost if deploy else kwargs.get("boost", const.BOOST)
        # color = deploy.color if deploy else kwargs.get("color", const.COLOR)
        #
        # shape = deploy.shape if deploy else kwargs.get("shape", const.SHAPE)
        # scale = deploy.scale if deploy else kwargs.get("scale", const.SCALE)
        # start = deploy.start if deploy else kwargs.get("start", const.START)
        # close = deploy.close if deploy else kwargs.get("close", const.CLOSE)
        # limit = deploy.limit if deploy else kwargs.get("limit", const.LIMIT)
        #
        # begin = deploy.begin if deploy else kwargs.get("begin", const.BEGIN)
        # final = deploy.final if deploy else kwargs.get("final", const.FINAL)
        #
        # crops = deploy.crops if deploy else kwargs.get("crops", const.CROPS)
        # omits = deploy.omits if deploy else kwargs.get("omits", const.OMITS)
        #
        # frate = deploy.frate if deploy else kwargs.get("frate", const.FRATE)
        # thres = deploy.thres if deploy else kwargs.get("thres", const.THRES)
        # shift = deploy.shift if deploy else kwargs.get("shift", const.SHIFT)
        # block = deploy.block if deploy else kwargs.get("block", const.BLOCK)

        # Copy
        async def check():
            screen_cap = None
            if os.path.isfile(vision):
                screen = cv2.VideoCapture(vision)
                if screen.isOpened():
                    screen_cap = Path(vision)
                screen.release()
            elif os.path.isdir(vision):
                file_list = [
                    file for file in os.listdir(vision) if os.path.isfile(os.path.join(vision, file))
                ]
                if len(file_list) >= 1:
                    screen = cv2.VideoCapture(open_file := os.path.join(vision, file_list[0]))
                    if screen.isOpened():
                        screen_cap = Path(open_file)
                    screen.release()
            return screen_cap

        # Copy
        async def frame_flip():
            change_record = os.path.join(
                os.path.dirname(vision),
                f"screen_fps{frate}_{random.randint(100, 999)}.mp4"
            )

            duration = await Switch.ask_video_length(fpb, vision)
            vision_start, vision_close, vision_limit = await Switch.ask_magic_point(
                Parser.parse_mills(start),
                Parser.parse_mills(close),
                Parser.parse_mills(limit),
                duration
            )
            vision_start = Parser.parse_times(vision_start)
            vision_close = Parser.parse_times(vision_close)
            vision_limit = Parser.parse_times(vision_limit)
            logger.info(f"视频时长: [{duration}] [{Parser.parse_times(duration)}]")
            logger.info(f"start=[{vision_start}] - close=[{vision_close}] - limit=[{vision_limit}]")

            await Switch.ask_video_change(
                fmp, frate, vision, change_record,
                start=vision_start, close=vision_close, limit=vision_limit
            )
            logger.info(f"视频转换完成: {Path(change_record).name}")
            os.remove(vision)
            logger.info(f"移除旧的视频: {Path(vision).name}")

            if shape:
                original_shape = await Switch.ask_video_larger(fpb, change_record)
                w, h, ratio = await Switch.ask_magic_frame(original_shape, shape)
                target_shape = w, h
                target_scale = scale
                logger.info(f"调整宽高比: {w} x {h}")
            elif scale:
                target_shape = shape
                target_scale = max(0.1, min(1.0, scale))
            else:
                target_shape = shape
                target_scale = 0.4

            video = VideoObject(change_record)
            task, hued = video.load_frames(
                silently_load_hued=color,
                not_transform_gray=False,
                shape=target_shape,
                scale=target_scale
            )
            return video, task, hued

        # Copy
        async def frame_flow():
            video, task, hued = await frame_flip()
            cutter = VideoCutter()

            compress_hook = CompressHook(1, None, False)
            cutter.add_hook(compress_hook)

            if len(crop_list := crops) > 0 and sum([j for i in crop_list for j in i.values()]) > 0:
                for crop in crop_list:
                    x, y, x_size, y_size = crop.values()
                    crop_hook = PaintCropHook((y_size, x_size), (y, x))
                    cutter.add_hook(crop_hook)
                    logger.debug(f"{crop_hook.__class__.__name__}: {x, y, x_size, y_size}")

            if len(omit_list := omits) > 0 and sum([j for i in omit_list for j in i.values()]) > 0:
                for omit in omit_list:
                    x, y, x_size, y_size = omit.values()
                    omit_hook = PaintOmitHook((y_size, x_size), (y, x))
                    cutter.add_hook(omit_hook)
                    logger.debug(f"{omit_hook.__class__.__name__}: {x, y, x_size, y_size}")

            save_hook = FrameSaveHook(extra_path)
            cutter.add_hook(save_hook)

            res = cutter.cut(
                video=video, block=block
            )

            stable, unstable = res.get_range(
                threshold=thres, offset=shift
            )

            file_list = os.listdir(extra_path)
            file_list.sort(key=lambda n: int(n.split("(")[0]))
            total_images, desired_count = len(file_list), 12

            if total_images <= desired_count:
                retain_indices = range(total_images)
            else:
                retain_indices = [int(i * (total_images / desired_count)) for i in range(desired_count)]
                if len(retain_indices) < desired_count:
                    retain_indices.append(total_images - 1)
                elif len(retain_indices) > desired_count:
                    retain_indices = retain_indices[:desired_count]

            for index, file in enumerate(file_list):
                if index not in retain_indices:
                    os.remove(os.path.join(extra_path, file))

            for draw in os.listdir(extra_path):
                toolbox.draw_line(os.path.join(extra_path, draw))

            classify = self.kc.classify(
                video=video, valid_range=stable, keep_data=True
            )

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

            return classify, frames

        # Copy
        async def frame_flick(classify):
            logger.info(f"阶段划分: {classify.get_ordered_stage_set()}")
            begin_stage, begin_frame = begin
            final_stage, final_frame = final
            try:
                start_frame = classify.get_not_stable_stage_range()[begin_stage][begin_frame]
                end_frame = classify.get_not_stable_stage_range()[final_stage][final_frame]
            except AssertionError as e:
                logger.error(f"{e}")
                start_frame = classify.get_important_frame_list()[0]
                end_frame = classify.get_important_frame_list()[-1]
                logger.warning(f"{const.DESC} Analyzer recalculate ...")
            except IndexError as e:
                logger.error(f"{e}")
                for i, unstable_stage in enumerate(classify.get_specific_stage_range("-3")):
                    Show.console.print(f"[bold]第 {i:02} 个非稳定阶段")
                    Show.console.print(f"[bold]{'=' * 30}")
                    for j, frame in enumerate(unstable_stage):
                        Show.console.print(f"[bold]第 {j:05} 帧: {frame}")
                    Show.console.print(f"[bold]{'=' * 30}\n")
                start_frame = classify.get_important_frame_list()[0]
                end_frame = classify.get_important_frame_list()[-1]
                logger.warning(f"{const.DESC} Analyzer recalculate ...")

            if start_frame == end_frame:
                logger.warning(f"{start_frame} == {end_frame}")
                start_frame, end_frame = classify.data[0], classify.data[-1]
                logger.warning(f"{const.DESC} Analyzer recalculate ...")

            time_cost = end_frame.timestamp - start_frame.timestamp
            logger.info(
                f"图像分类结果: [开始帧: {start_frame.timestamp:.5f}] [结束帧: {end_frame.timestamp:.5f}] [总耗时: {time_cost:.5f}]"
            )
            return start_frame.frame_id, end_frame.frame_id, time_cost

        # Copy
        async def frame_forge(frame):
            try:
                (_, codec), pic_path = cv2.imencode(".png", frame.data), os.path.join(
                    frame_path, f"{frame.frame_id}_{format(round(frame.timestamp, 5), '.5f')}.png"
                )
                async with aiofiles.open(pic_path, "wb") as f:
                    await f.write(codec.tobytes())
            except Exception as e:
                return e

        async def analytics_basic():
            video, task, hued = await frame_flip()

            if color:
                video.hued_data = tuple(hued.result())
                logger.info(f"彩色帧已加载: {video.frame_details(video.hued_data)}")
                task.shutdown()
                frames = [i for i in video.hued_data]
            else:
                frames = [i for i in video.grey_data]

            logger.debug(f"运行环境: {_platform}")
            if _platform == "win32":
                forge_result = await asyncio.gather(
                    *(frame_forge(frame) for frame in frames), return_exceptions=True
                )
            else:
                tasks = [
                    [frame_forge(frame) for frame in chunk] for chunk in
                    [frames[i:i + 100] for i in range(0, len(frames), 100)]
                ]
                forge_list = []
                for task in tasks:
                    task_result = await asyncio.gather(*task, return_exceptions=True)
                    forge_list.extend(task_result)
                forge_result = tuple(forge_list)

            for result in forge_result:
                if isinstance(result, Exception):
                    logger.error(f"Error: {result}")

            start_frame, end_frame = frames[0], frames[-1]

            time_cost = end_frame.timestamp - start_frame.timestamp
            return (start_frame.frame_id, end_frame.frame_id, time_cost), None

        async def analytics_keras():
            classify, frames = await frame_flow()

            logger.debug(f"运行环境: {_platform}")
            if _platform == "win32":
                flick_result, *forge_result = await asyncio.gather(
                    frame_flick(classify), *(frame_forge(frame) for frame in frames),
                    return_exceptions=True
                )
            else:
                tasks = [
                    [frame_forge(frame) for frame in chunk] for chunk in
                    [frames[i:i + 100] for i in range(0, len(frames), 100)]
                ]
                flick_task = asyncio.create_task(frame_flick(classify))
                forge_list = []
                for task in tasks:
                    task_result = await asyncio.gather(*task, return_exceptions=True)
                    forge_list.extend(task_result)
                forge_result = tuple(forge_list)
                flick_result = await flick_task

            for result in forge_result:
                if isinstance(result, Exception):
                    logger.error(f"Error: {result}")

            return flick_result, classify

        # Analyzer first ===============================================================================================
        if (screen_record := await check()) is None:
            return logger.error(f"{vision} 不是一个标准的视频文件或视频文件已损坏 ...")
        logger.info(f"{screen_record.name} 可正常播放，准备加载视频 ...")
        # Analyzer first ===============================================================================================

        # Analyzer last ================================================================================================
        (start, end, cost), classifier = await analytics_keras() if self.kc else await analytics_basic()
        return Review(start, end, cost, classifier)
        # Analyzer last ================================================================================================


class Tmp(object):

    def __init__(self, device: "Device"):
        self.device = device
        self.report = Report(const.CREDO)
        self.medias = Med()
        self.alynex = Aly(const.MODEL, const.MODEL_SHAPE, const.MODEL_AISLE)

    async def test_01(self):
        audio = os.path.join(const.AUDIO, query := "关闭蓝牙.mp3")
        self.report.title = query

        device_events = {}
        for _ in range(2):
            self.report.query = query.split(".")[0]

            device_events[self.device.serial] = {
                "head_event": asyncio.Event(), "done_event": asyncio.Event(),
                "stop_event": asyncio.Event(), "fail_event": asyncio.Event()
            }
            exe = ThreadPoolExecutor()
            video_temp, transports = await _loop.run_in_executor(
                exe, self.medias.start_record,
                self.device.serial, self.report.video_path, device_events[self.device.serial]
            )

            await self.device.ask_key_event(231)
            await self.device.ask_sleep(1)
            await self.medias.audio_player(audio)
            await self.device.ask_sleep(3)

            await self.medias.close_record(transports, device_events[self.device.serial])
            await self.alynex.analyzer(
                video_temp, None,
                self.report.frame_path, self.report.frame_path, "ffmpeg", "ffprobe"
            )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


async def main():
    manage = Manage("adb")
    device_list = await manage.operate_device()
    async with Tmp(device_list[0]) as test:
        await test.test_01()


if __name__ == '__main__':
    try:
        _loop = asyncio.get_event_loop()
        _loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("__main__ Ctrl C Event ...")
