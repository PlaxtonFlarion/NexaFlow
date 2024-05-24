import os
import sys
import time
import signal
import random
import asyncio

try:
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
    import pygame
except ImportError as e:
    raise ImportError("AudioPlayer requires pygame. install it first.")

from frameflow.skills.show import Show
from engine.terminal import Terminal


class Record(object):

    record_events: dict[dict[str, asyncio.Event]] = {}
    melody_events: asyncio.Event = asyncio.Event()

    def __init__(self, **kwargs):
        self.platform = sys.platform

        self.alone = kwargs.get("alone", False)
        self.whist = kwargs.get("whist", False)
        self.frate = kwargs.get("frate", 60)

    async def ask_start_record(self, device, dst, **kwargs):

        async def input_stream():
            async for line in transports.stdout:
                Show.annal(stream := line.decode(encoding="UTF-8", errors="ignore").strip())
                if "Recording started" in stream:
                    events["head"].set()
                elif "Recording complete" in stream:
                    bridle.set()
                    events["done"].set()
                    break

        async def error_stream():
            async for line in transports.stderr:
                Show.annal(stream := line.decode(encoding="UTF-8", errors="ignore").strip())
                if "Could not find" in stream or "connection failed" in stream or "Recorder error" in stream:
                    events["fail"].set()
                    break

        self.record_events[device.sn]: dict[str, asyncio.Event] = {
            "head": asyncio.Event(), "done": asyncio.Event(),
            "stop": asyncio.Event(), "fail": asyncio.Event(),
        }

        bridle: asyncio.Event = self.record_events[device.sn]["stop"] if self.alone else self.melody_events
        events: dict[str, asyncio.Event] = self.record_events[device.sn]

        video_flag = f"{time.strftime('%Y%m%d%H%M%S')}_{random.randint(100, 999)}.mkv"

        loc_name = ["--window-x", "--window-y", "--window-width", "--window-height"]
        location = [f"{k}={v}" for k, v in zip(loc_name, loc)] if (loc := kwargs.get("location", ())) else []

        cmd = ["scrcpy", "-s", device.sn]
        cmd += [f"--display-id={device.id}"] if device.id != 0 else []
        cmd += location if location else []
        cmd += ["--no-audio"]
        cmd += ["--video-bit-rate", "8M", "--max-fps", f"{self.frate}"]
        cmd += ["-Nr" if self.whist else "--record", video_temp := f"{os.path.join(dst, 'screen')}_{video_flag}"]

        transports = await Terminal.cmd_link(*cmd)

        asyncio.create_task(input_stream())
        asyncio.create_task(error_stream())
        await asyncio.sleep(1)

        return video_temp, transports

    async def ask_close_record(self, device, video_temp, transports):

        # async def complete():
        #     for _ in range(10):
        #         if events["done"].is_set():
        #             return f"{desc} 视频录制成功", banner
        #         elif events["fail"].is_set():
        #             return f"{desc} 视频录制失败", banner
        #         await asyncio.sleep(0.2)
        #     return f"{desc} 视频录制失败", banner

        desc = f"{device.tag} {device.sn} PID={transports.pid}"

        events: dict[str, asyncio.Event] = self.record_events[device.sn]
        banner: str = os.path.basename(video_temp)

        if self.whist:
            cancel = signal.CTRL_C_EVENT if self.platform == "win32" else signal.SIGINT
            try:
                transports.send_signal(cancel)
            except ProcessLookupError:
                Show.notes(f"{desc} Stop With Signal")

            try:
                shell = True if self.platform == "win32" else False
                Terminal.cmd_oneshot(["echo", "Cancel"], shell=shell)
            except KeyboardInterrupt:
                Show.notes(f"{desc} Stop With {cancel.name} Code={cancel.value}")

        else:
            if self.platform == "win32":
                kill_process = ["taskkill", "/im", "scrcpy.exe"]
                await Terminal.cmd_link(*kill_process)
            else:
                kill_process = ["xargs", "kill", "-SIGINT"]
                if pid_list := await Terminal.cmd_line("pgrep", "scrcpy"):
                    await Terminal.cmd_line(
                        *kill_process, input="\n".join(pid_list).encode()
                    )
            Show.notes(f"{desc} Stop With {' '.join(kill_process)}")

        try:
            await asyncio.wait_for(events["done"].wait(), 2)
        except asyncio.TimeoutError:
            return f"{desc} 视频录制失败", banner
        else:
            return f"{desc} 视频录制成功", banner

    async def check_timer(self, device, amount):
        bridle = self.record_events[device.sn]["stop"] if self.alone else self.melody_events
        events = self.record_events[device.sn]

        desc = f"{device.tag} {device.sn}"

        while True:
            if events["head"].is_set():
                for i in range(amount):
                    row = amount - i if amount - i <= 10 else 10
                    Show.notes(f"{desc} 剩余时间 -> {amount - i:02} 秒 {'----' * row} ...")
                    if bridle.is_set() and i != amount:
                        return Show.notes(f"{desc} 主动停止 ...")
                    elif events["fail"].is_set():
                        return Show.notes(f"{desc} 意外停止 ...")
                    await asyncio.sleep(1)
                return Show.notes(f"{desc} 剩余时间 -> 00 秒")
            elif events["fail"].is_set():
                return Show.notes(f"{desc} 意外停止 ...")
            await asyncio.sleep(0.2)

    async def check_event(self, device, exec_tasks):
        if self.alone and (events := self.record_events.get(device.sn, None)):
            await asyncio.wait(
                [asyncio.create_task(events[ev].wait()) for ev in ["stop", "done", "fail"]],
                return_when=asyncio.FIRST_COMPLETED
            )
        else:
            await self.melody_events.wait()

        if task := exec_tasks.get(device.sn, []):
            task.cancel()
        return Show.notes(f"[bold #CD853F]{device.sn} Cancel task[/]")

    # async def check_event(self, device, exec_tasks):
    #     if self.alone and (events := self.record_events.get(device.sn, None)):
    #         bridle = events["stop"], events["done"], events["fail"]
    #         while True:
    #             if any(event.is_set() for event in bridle):
    #                 break
    #             await asyncio.sleep(1)
    #     else:
    #         await self.melody_events.wait()
    #
    #     if task := exec_tasks.get(device.sn, []):
    #         task.cancel()
    #     return Show.notes(f"[bold #CD853F]{device.sn} Cancel task[/]")

    async def flunk_event(self):
        return any(
            events["fail"].is_set() for events in self.record_events.values()
        )

    async def clean_event(self):
        self.melody_events.clear()
        for event_dict in self.record_events.values():
            for events in event_dict.values():
                events.clear()
        self.record_events.clear()


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


if __name__ == '__main__':
    pass
