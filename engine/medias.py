import os
import time
import signal
import random
import asyncio

try:
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
    import pygame
except ImportError as e:
    raise ImportError("AudioPlayer requires pygame. install it first.")

from loguru import logger
from engine.terminal import Terminal


class Record(object):

    device_events: dict = {}
    rhythm_events: asyncio.Event = asyncio.Event()

    def __init__(self, scrcpy: str, system: str,  *_, **kwargs):
        self.scrcpy, self.system = scrcpy, system
        self.alone = kwargs.get("alone", False)
        self.whist = kwargs.get("whist", False)
        if self.alone and self.whist:
            self.whist = False

    async def timing_less(self, device, amount):
        serial = device.serial
        bridle = self.device_events[serial]["stop_event"] if self.alone else self.rhythm_events
        events = self.device_events[serial]

        while True:
            if events["head_event"].is_set():
                for i in range(amount):
                    row = amount - i if amount - i <= 10 else 10
                    logger.info(f"{serial} 剩余时间 -> {amount - i:02} 秒 {'----' * row} ...")
                    if bridle.is_set() and i != amount:
                        return logger.info(f"{serial} 主动停止 ...")
                    elif events["fail_event"].is_set():
                        return logger.info(f"{serial} 意外停止 ...")
                    await asyncio.sleep(1)
                return logger.info(f"{serial} 剩余时间 -> 00 秒")
            elif events["fail_event"].is_set():
                return logger.info(f"{serial} 意外停止 ...")
            await asyncio.sleep(0.2)

    async def timing_many(self, device):
        serial = device.serial
        bridle = self.device_events[serial]["stop_event"] if self.alone else self.rhythm_events
        events = self.device_events[serial]

        while True:
            if events["head_event"].is_set():
                while True:
                    if bridle.is_set():
                        return logger.info(f"{serial} 主动停止 ...")
                    elif events["fail_event"].is_set():
                        return logger.info(f"{serial} 意外停止 ...")
                    await asyncio.sleep(1)
            elif events["fail_event"].is_set():
                return logger.info(f"{serial} 意外停止 ...")
            await asyncio.sleep(0.2)

    async def start_record(self, device, dst):

        async def input_stream():
            async for line in transports.stdout:
                logger.info(stream := line.decode(encoding="UTF-8", errors="ignore").strip())
                if "Recording started" in stream:
                    events["head_event"].set()
                elif "Recording complete" in stream:
                    bridle.set()
                    events["done_event"].set()
                    break

        async def error_stream():
            async for line in transports.stderr:
                logger.info(stream := line.decode(encoding="UTF-8", errors="ignore").strip())
                if "Could not find" in stream or "connection failed" in stream or "Recorder error" in stream:
                    events["fail_event"].set()
                    break

        self.device_events[device.serial] = {
            "head_event": asyncio.Event(), "done_event": asyncio.Event(),
            "stop_event": asyncio.Event(), "fail_event": asyncio.Event(),
        }

        serial = device.serial
        bridle = self.device_events[serial]["stop_event"] if self.alone else self.rhythm_events
        events = self.device_events[serial]

        video_flag = f"{time.strftime('%Y%m%d%H%M%S')}_{random.randint(100, 999)}.mkv"

        cmd = [self.scrcpy, "-s", serial]
        cmd += ["--no-audio", "--video-bit-rate", "8M", "--max-fps", "60"]
        cmd += ["-Nr" if self.whist else "--record", video_temp := f"{os.path.join(dst, 'screen')}_{video_flag}"]

        transports = await Terminal.cmd_link(*cmd)

        asyncio.create_task(input_stream())
        asyncio.create_task(error_stream())
        await asyncio.sleep(1)

        return video_temp, transports

    async def close_record(self, video_temp, transports, device):

        async def close():
            for _ in range(10):
                if events["done_event"].is_set():
                    return f"{device.serial} 视频录制成功", banner
                elif events["fail_event"].is_set():
                    return f"{device.serial} 视频录制失败", banner
                await asyncio.sleep(0.2)
            return f"{device.serial} 视频录制失败", banner

        events = self.device_events[device.serial]
        banner = os.path.basename(video_temp)

        # TODO Feasibility to be tested -> transports.send_signal(signal.SIGINT) ?
        if self.system != "win32":
            transports.terminate()
            return await close()

        if self.whist:
            logger.info(f"PID: {transports.pid}")
            transports.send_signal(signal.CTRL_C_EVENT)

        try:
            await Terminal.cmd_line("taskkill", "/im", "scrcpy.exe")
        except KeyboardInterrupt:
            logger.warning(f"Stop With Ctrl_C_Event ...")
        return await close()

    async def event_check(self):
        return any(
            event["fail_event"].is_set() for event in self.device_events.values()
        )

    async def clean_check(self):
        self.rhythm_events.clear()
        for event_dict in self.device_events.values():
            for event in event_dict.values():
                if isinstance(event, asyncio.Event):
                    event.clear()
        self.device_events.clear()


class Player(object):

    player_events: dict = {}
    melody_events: asyncio.Event = asyncio.Event()

    @staticmethod
    async def audio_player(audio_file: str):
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.set_volume(1.0)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    async def event_check(self):
        return all(
            event.is_set() for event in self.player_events.values()
        )

    async def clean_check(self):
        self.melody_events.clear()
        for event_dict in self.player_events.values():
            for event in event_dict.values():
                if isinstance(event, asyncio.Event):
                    event.clear()
        self.player_events.clear()


if __name__ == '__main__':
    pass
