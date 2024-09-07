#
#   __  __          _ _
#  |  \/  | ___  __| (_) __ _ ___
#  | |\/| |/ _ \/ _` | |/ _` / __|
#  | |  | |  __/ (_| | | (_| \__ \
#  |_|  |_|\___|\__,_|_|\__,_|___/
#

import os
import re
import sys
import time
import random
import asyncio

try:
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
    import pygame
except ImportError as e:
    raise ImportError("AudioPlayer requires pygame. install it first.")

from engine.terminal import Terminal
from frameflow.skills.show import Show
from nexaflow import const


class Record(object):

    record_events: dict[dict[str, asyncio.Event]] = {}
    melody_events: asyncio.Event = asyncio.Event()

    def __init__(self, version: str, **kwargs):
        self.version, self.station = version, sys.platform

        self.alone = kwargs.get("alone", False)
        self.whist = kwargs.get("whist", False)
        self.frate = kwargs.get("frate", const.FRATE)

    async def ask_start_record(self, device, dst, **kwargs):
        """
        异步启动视频录制。

        该方法启动视频录制进程，并监控其标准输出和标准错误输出，记录录制状态。

        参数:
            device: 设备对象，包含设备的序列号 (sn) 和显示 ID (id)。
            dst (str): 视频文件的保存路径。
            **kwargs: 其他可选参数，包括窗口位置和大小。

        返回:
            tuple: 包含视频文件路径和进程对象。

        内部函数:
            input_stream(): 异步读取进程的标准输出，监控录制状态。
            error_stream(): 异步读取进程的标准错误输出，监控可能的错误。

        注意:
            - 初始化录制事件字典，包含 `head`、`done`、`stop` 和 `fail` 事件。
            - 根据设备信息和可选参数构建录制命令。
            - 启动录制进程并创建异步任务监控输出流。
            - 通过 asyncio.sleep(1) 等待录制进程启动稳定。
        """

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

        cmd = ["scrcpy", "-s", device.sn, "--no-audio", "--video-bit-rate=8M", f"--max-fps={self.frate}"]

        if device.id != 0:
            cmd += [f"--display-id={device.id}"]

        loc_name = ["--window-x", "--window-y", "--window-width", "--window-height"]
        cmd += [f"{k}={v}" for k, v in zip(loc_name, loc)] if (loc := kwargs.get("location", ())) else []

        if self.whist:
            try:
                vs = float(re.search(r"(?<=scrcpy\s)\d.*(?=\.\d\s)", self.version).group())
            except (AttributeError, TypeError):
                vs = 2.5
            cmd += ["--no-display"] if vs <= 2.4 else ["--no-window"]

        cmd += ["--record", video_temp := f"{os.path.join(dst, 'screen')}_{video_flag}"]

        transports = await Terminal.cmd_link(*cmd)

        _ = asyncio.create_task(input_stream())
        _ = asyncio.create_task(error_stream())
        await asyncio.sleep(1)

        return video_temp, transports

    async def ask_close_record(self, device, video_temp, transports):
        """
        异步停止视频录制。

        该方法通过终止录制进程及其子进程来停止视频录制，并监控录制完成状态。

        参数:
            device: 设备对象，包含设备的标签 (tag) 和序列号 (sn)。
            video_temp (str): 视频文件的临时路径。
            transports: 录制进程对象。

        返回:
            tuple: 包含描述信息和视频文件名称。

        内部函数:
            find_child(pid): 异步查找指定进程 ID 的子进程。
                - 如果平台是 Windows，使用 PowerShell 命令获取子进程 ID。
                - 否则，使用 pgrep 命令获取子进程 ID。
            stop_child(pid): 异步终止指定进程 ID 的子进程。
                - 如果平台是 Windows，使用 PowerShell 命令终止进程。
                - 否则，使用 xargs 和 kill 命令发送 SIGINT 信号终止进程。

        注意:
            - 初始化录制事件字典，包含 `done` 事件。
            - 查找并终止录制进程的所有子进程。
            - 等待录制完成事件触发，超时时返回失败信息。
            - 如果录制成功完成，返回成功信息。
        """

        async def find_child(pid):
            if self.station == "win32":
                child_pids = await Terminal.cmd_line(
                    "powershell", "-Command", "Get-CimInstance", "Win32_Process", "|", "Where-Object",
                    f"{{ $_.ParentProcessId -eq {pid} }}", "|", "Select-Object", "-ExpandProperty", "ProcessId"
                )
            else:
                child_pids = await Terminal.cmd_line(
                    "pgrep", "-P", pid
                )
            Show.notes(f"{desc} PID={child_pids}")

            return [line.strip() for line in child_pids.splitlines() if line.strip().isdigit()] if child_pids else []

        async def stop_child(pid):
            if self.station == "win32":
                off = await Terminal.cmd_line(
                    "powershell", "-Command", "Stop-Process", "-Id", pid, "-Force"
                )
            else:
                off = await Terminal.cmd_line(
                    "xargs", "kill", "-SIGINT", transmit=pid.encode()
                )
            Show.notes(f"{desc} OFF={off}")

        desc = f"{device.tag} {device.sn} PPID={(record_pid := transports.pid)}"

        events: dict[str, asyncio.Event] = self.record_events[device.sn]
        banner: str = os.path.basename(video_temp)

        if child_process_list := await find_child(record_pid):
            await asyncio.gather(
                *(stop_child(pid) for pid in child_process_list if pid)
            )

        try:
            await asyncio.wait_for(events["done"].wait(), 3)
        except asyncio.TimeoutError:
            return f"{desc} 视频录制失败", banner
        else:
            return f"{desc} 视频录制成功", banner

    async def check_timer(self, device, amount):
        """
        异步检查计时器状态。

        该方法监控设备录制状态，并在指定时间内输出剩余时间。

        参数:
            device: 设备对象，包含设备的标签 (tag) 和序列号 (sn)。
            amount (int): 计时器的总秒数。

        内部变量:
            bridle: 停止录制事件，根据 self.alone 决定是否使用全局或局部事件。
            events: 录制事件字典，包含 `head`、`stop`、`fail` 等事件。
            desc (str): 设备描述信息，包括标签和序列号。

        处理流程:
            - 检查 `head` 事件是否已触发，如果是，则开始倒计时。
            - 在倒计时过程中，每秒检查 `stop` 或 `fail` 事件是否触发：
                - 如果 `stop` 事件触发且尚未结束，输出主动停止信息。
                - 如果 `fail` 事件触发，输出意外停止信息。
                - 否则，继续倒计时并每秒更新剩余时间。
            - 如果 `fail` 事件在 `head` 事件触发前触发，输出意外停止信息。
            - 每 0.2 秒检查一次事件状态。

        返回:
            None
        """

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
        """
        异步检查事件状态。

        该方法监控设备录制过程中的事件状态，并根据事件状态执行相应操作。

        参数:
            device: 设备对象，包含设备的序列号 (sn)。
            exec_tasks (dict): 存储任务的字典，其中键是设备的序列号，值是对应的异步任务。

        内部变量:
            events (dict): 录制事件字典，包含 `stop`、`done`、`fail` 等事件。
            bridle (tuple): 包含需要监控的事件 (`stop`、`done`、`fail`)。
            task (coroutine): 存储在 exec_tasks 字典中的异步任务。

        处理流程:
            - 如果 self.alone 为 True，则获取设备的录制事件字典，并监控 `stop`、`done` 和 `fail` 事件。
                - 进入循环，检查是否有任何一个事件被触发，如果是，则跳出循环。
                - 每秒钟检查一次事件状态。
            - 如果 self.alone 为 False，则等待全局事件 `melody_events` 被触发。
            - 检查 exec_tasks 字典中是否有与设备序列号对应的任务，如果有，则取消该任务。
            - 输出任务取消信息。

        返回:
            None
        """

        if self.alone and (events := self.record_events.get(device.sn, None)):
            bridle = events["stop"], events["done"], events["fail"]
            while True:
                if any(event.is_set() for event in bridle):
                    break
                await asyncio.sleep(1)
        else:
            await self.melody_events.wait()

        if task := exec_tasks.get(device.sn, []):
            task.cancel()
        return Show.notes(f"[bold #CD853F]{device.sn} Cancel task[/]")

    async def flunk_event(self):
        """
        检查是否有任何设备触发了失败事件。

        该方法遍历所有记录的设备事件，检查是否存在任何一个设备的 `fail` 事件被触发。

        返回:
            bool: 如果任何一个设备的 `fail` 事件被触发，返回 True；否则返回 False。

        内部变量:
            events (dict): 包含录制事件的字典，其中 `fail` 事件表示录制失败。

        处理流程:
            - 遍历 self.record_events 字典的所有值。
            - 对每个设备的事件字典，检查 `fail` 事件是否被触发。
            - 如果找到任何一个 `fail` 事件被触发，返回 True。
            - 如果所有设备的 `fail` 事件均未触发，返回 False。
        """

        return any(
            events["fail"].is_set() for events in self.record_events.values()
        )

    async def clean_event(self):
        """
        清理所有录制事件并重置状态。

        该方法用于清理所有录制事件，重置状态，以便为新地录制任务做好准备。

        操作步骤:
            1. 清除全局事件 `self.melody_events`。
            2. 遍历 `self.record_events` 字典的所有值，并清除其中的每个事件。
            3. 清空 `self.record_events` 字典。

        处理流程:
            - `self.melody_events.clear()`: 清除全局事件，确保其处于未触发状态。
            - 遍历 `self.record_events` 中的每个设备事件字典：
                - 对每个设备事件字典中的事件调用 `clear()` 方法，使其处于未触发状态。
            - 清空 `self.record_events` 字典，移除所有设备的事件记录。
        """

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
        """
        异步播放音频文件。

        该方法使用 pygame 库异步播放指定的音频文件，播放时音量设置为最大。

        参数:
            audio_file (str): 音频文件的路径。

        注意:
            - 初始化 pygame.mixer 模块并加载音频文件。
            - 设置音量为 1.0（最大音量）。
            - 播放音频文件，并在音频播放期间通过检查 pygame.mixer.music.get_busy() 保持循环，直到播放结束。
            - 在循环中通过 pygame.time.Clock().tick(10) 控制帧率，防止占用过多 CPU 资源。
        """

        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.set_volume(1.0)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)


if __name__ == '__main__':
    pass
