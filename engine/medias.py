#
#   __  __          _ _
#  |  \/  | ___  __| (_) __ _ ___
#  | |\/| |/ _ \/ _` | |/ _` / __|
#  | |  | |  __/ (_| | | (_| \__ \
#  |_|  |_|\___|\__,_|_|\__,_|___/
#

"""
版权所有 (c) 2024  Framix(画帧秀)
此文件受 Framix(画帧秀) 许可证的保护。您可以在 LICENSE.md 文件中查看详细的许可条款。

Copyright (c) 2024  Framix(画帧秀)
This file is licensed under the Framix(画帧秀) License. See the LICENSE.md file for more details.

# Copyright (c) 2024  Framix(画帧秀)
# このファイルは Framix(画帧秀) ライセンスの下でライセンスされています。詳細は LICENSE.md ファイルを参照してください。
"""

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

from engine.device import Device
from engine.terminal import Terminal
from nexacore.design import Design
from nexaflow import const


class Record(object):
    """
    视频录制管理类。

    该类用于控制 Android 设备的视频录制行为，支持启动、停止录制，监控录制状态，并处理录制任务相关事件。

    Class Attributes
    ----------------
    record_events : dict[dict[str, asyncio.Event]]
        每台设备的录制事件集合，包含 "head"（录制开始）、"done"（录制完成）、
        "stop"（主动停止）和 "fail"（录制失败）等事件标识。

    melody_events : asyncio.Event
        全局录制事件控制器，适用于非独立模式下的统一停止控制。

    Instance Attributes
    -------------------
    version : str
        scrcpy 的版本号信息，用于判断是否支持无窗口录制选项。

    station : str
        当前操作系统平台，例如 'win32' 或 'linux'。

    alone : bool
        是否为独立控制模式，如果为 True，则每台设备分别控制录制流程。

    whist : bool
        是否启用静默录制（无窗口模式），用于降低图形资源消耗。

    frate : int
        录制时的最大帧率，默认为 `const.FRATE`。

    Notes
    -----
    - 支持异步控制多设备录制任务。
    - 提供全局与局部事件机制，适用于不同的录制调度需求。
    - 所有录制状态通过事件进行管理与清理。
    """

    record_events: dict[dict[str, asyncio.Event]] = {}
    melody_events: asyncio.Event = asyncio.Event()

    def __init__(self, version: str, **kwargs):
        self.version, self.station = version, sys.platform

        self.alone = kwargs.get("alone", False)
        self.whist = kwargs.get("whist", False)
        self.frate = kwargs.get("frate", const.FRATE)

    async def ask_start_record(
            self, device: "Device", dst: str, **kwargs
    ) -> tuple[str, "asyncio.subprocess.Process"]:
        """
        异步启动视频录制。

        该方法启动视频录制进程，并监控其标准输出和标准错误输出，记录录制状态。

        Parameters
        ----------
        device : object
            设备对象，包含设备的序列号 (sn) 和显示 ID (id)。

        dst : str
            视频文件的保存路径。

        **kwargs : dict
            其他可选参数，包括窗口位置和大小。

        Returns
        -------
        tuple
            包含视频文件路径和进程对象。

        Notes
        -----
        - 初始化录制事件字典，包含 `head`、`done`、`stop` 和 `fail` 事件。
        - 根据设备信息和可选参数构建录制命令。
        - 启动录制进程并创建异步任务监控输出流。
        - 通过 `asyncio.sleep(1)` 等待录制进程启动稳定。

        Workflow
        --------
        1. 构建录制命令（使用 scrcpy），包括设备序列号、窗口位置、帧率等参数。
        2. 启动子进程，并并发创建标准输出和错误输出的监听任务。
        3. 通过事件字典监控录制状态。
        4. 返回录制文件路径和进程对象，供后续停止录制使用。
        """

        async def input_stream():
            async for line in transports.stdout:
                Design.annal(stream := line.decode(encoding=const.CHARSET, errors="ignore").strip())
                if "Recording started" in stream:
                    events["head"].set()
                elif "Recording complete" in stream:
                    bridle.set()
                    events["done"].set()
                    break

        async def error_stream():
            async for line in transports.stderr:
                Design.annal(stream := line.decode(encoding=const.CHARSET, errors="ignore").strip())
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

        transports = await Terminal.cmd_link(cmd)

        asyncio.create_task(input_stream())
        asyncio.create_task(error_stream())

        await asyncio.sleep(1)

        return video_temp, transports

    async def ask_close_record(
            self, device: "Device", video_temp: str, transports: "asyncio.subprocess.Process"
    ) -> tuple:
        """
        异步停止视频录制。

        该方法通过终止录制进程及其子进程来停止视频录制，并监控录制完成状态。

        Parameters
        ----------
        device : Device
            设备对象，包含设备的标签 (tag) 和序列号 (sn)。

        video_temp : str
            视频文件的临时路径。

        transports : asyncio.subprocess.Process
            录制进程对象。

        Returns
        -------
        tuple
            包含描述信息和视频文件名称。

        Notes
        -----
        - 初始化录制事件字典，包含 `done` 事件。
        - 查找并终止录制进程的所有子进程。
        - 等待录制完成事件触发，超时时返回失败信息。
        - 如果录制成功完成，返回成功信息。

        Workflow
        --------
        1. 使用 `find_child(pid)` 获取录制进程的所有子进程 PID。
        2. 使用 `stop_child(pid)` 逐个终止子进程：
            - Windows 使用 PowerShell 的 `Stop-Process`。
            - 类 Unix 系统使用 `xargs kill -SIGINT`。
        3. 等待 `done` 事件的触发，超时则视为失败。
        4. 根据事件状态返回成功或失败的描述信息及视频文件名。
        """

        async def find_child(pid: str) -> list:
            if self.station == "win32":
                child_pids = await Terminal.cmd_line(
                    [
                        "powershell", "-Command", "Get-CimInstance", "Win32_Process", "|", "Where-Object",
                        f"{{ $_.ParentProcessId -eq {pid} }}", "|", "Select-Object", "-ExpandProperty", "ProcessId"
                    ]
                )
            else:
                child_pids = await Terminal.cmd_line(
                    ["pgrep", "-P", pid]
                )
            Design.notes(f"{desc} PID={child_pids}")

            return [
                line.strip() for line in child_pids.splitlines() if line.strip().isdigit()
            ] if child_pids else []

        async def stop_child(pid: str) -> None:
            if self.station == "win32":
                off = await Terminal.cmd_line(
                    ["powershell", "-Command", "Stop-Process", "-Id", pid, "-Force"]
                )
            else:
                off = await Terminal.cmd_line(
                    ["xargs", "kill", "-SIGINT"], transmit=pid.encode()
                )
            Design.notes(f"{desc} OFF={off}")

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

    async def check_timer(self, device: "Device", amount: int) -> None:
        """
        监控录制计时状态。

        该方法在视频录制过程中，实时倒计时，并在指定时间内监控事件状态。

        Parameters
        ----------
        device : object
            设备对象，包含设备的标签 (tag) 和序列号 (sn)。

        amount : int
            倒计时的总秒数。

        Returns
        -------
        None

        Notes
        -----
        - 使用 `head` 事件判断录制是否已启动。
        - 若 `stop` 在倒计时中被触发，视为主动停止。
        - 若 `fail` 在任意阶段被触发，视为录制失败。
        - 每秒更新剩余时间，展示提示信息。

        Workflow
        --------
        1. 等待 `head` 事件触发后进入倒计时循环。
        2. 在倒计时过程中：
            - 每秒输出剩余时间提示；
            - 若 `stop` 事件被设定，则提前终止；
            - 若 `fail` 事件被设定，则输出意外终止信息。
        3. 若在 `head` 前就收到 `fail`，立即终止。
        4. 倒计时正常结束后输出完成提示。
        """
        bridle = self.record_events[device.sn]["stop"] if self.alone else self.melody_events
        events = self.record_events[device.sn]

        desc = f"{device.tag} {device.sn}"

        while True:
            if events["head"].is_set():
                for i in range(amount):
                    row = amount - i if amount - i <= 10 else 10
                    Design.notes(f"{desc} 剩余时间 -> {amount - i:02} 秒 {'----' * row} ...")
                    if bridle.is_set() and i != amount:
                        return Design.notes(f"{desc} 主动停止 ...")
                    elif events["fail"].is_set():
                        return Design.notes(f"{desc} 意外停止 ...")
                    await asyncio.sleep(1)
                return Design.notes(f"{desc} 剩余时间 -> 00 秒")
            elif events["fail"].is_set():
                return Design.notes(f"{desc} 意外停止 ...")
            await asyncio.sleep(0.2)

    async def check_event(self, device: "Device", exec_tasks: dict[str, "asyncio.Task"]) -> None:
        """
        监听录制过程中的任务中断事件。

        该方法用于在任务执行期间，实时监听 `stop`、`done` 或 `fail` 事件是否触发，并安全中断任务。

        Parameters
        ----------
        device : object
            设备对象，包含设备的序列号 (sn)。

        exec_tasks : dict
            异步任务字典，key 是设备序列号，value 是任务实例。

        Returns
        -------
        None

        Notes
        -----
        - 独立控制模式下监听设备专属事件；
        - 非独立模式下监听全局 `melody_events`；
        - 一旦事件触发，取消当前设备的任务。

        Workflow
        --------
        1. 判断当前控制模式是否为独立 (`self.alone`)。
        2. 独立模式下循环等待 `stop`、`done`、`fail` 中任一事件触发。
        3. 非独立模式下等待全局 `melody_events`。
        4. 若任务存在于 `exec_tasks` 中，调用 `cancel()` 中断执行。
        5. 输出取消任务提示信息。
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

        return Design.notes(f"[bold #CD853F]{device.sn} Cancel task[/]")

    async def flunk_event(self) -> bool:
        """
        检查是否存在录制失败事件。

        该方法遍历所有设备的事件状态，判断是否有任何设备在录制过程中触发了 `fail` 事件。

        Returns
        -------
        bool
            若有任一设备的 `fail` 事件被触发，则返回 True；否则返回 False。

        Notes
        -----
        - 每个设备在录制启动时会生成一组事件；
        - `fail` 表示录制异常中断，可能由 scrcpy 异常、连接错误等引起；
        - 本方法适用于判断录制是否需要中断或重试。

        Workflow
        --------
        1. 遍历 `self.record_events` 中所有设备的事件字典；
        2. 检查每个事件字典中的 `fail` 是否处于已触发状态；
        3. 若任一事件触发，则返回 True；
        4. 全部未触发时，返回 False。
        """
        return any(events["fail"].is_set() for events in self.record_events.values())

    async def clean_event(self) -> None:
        """
        清理所有录制事件状态。

        该方法在一次录制任务结束后调用，用于重置所有事件对象，防止状态残留影响后续任务。

        Returns
        -------
        None

        Notes
        -----
        - 包含清空全局事件 `melody_events`；
        - 遍历所有设备事件并逐个 `clear()`；
        - 最终清空 `record_events` 字典。

        Workflow
        --------
        1. 调用 `self.melody_events.clear()` 重置全局事件；
        2. 遍历 `record_events`，对每个设备的事件集合执行清除操作；
        3. 清空 `record_events`，为下一轮任务做好准备。
        """
        self.melody_events.clear()
        for event_dict in self.record_events.values():
            for events in event_dict.values():
                events.clear()
        self.record_events.clear()


class Player(object):
    """
    音频播放控制类。

    该类用于管理音频播放行为，支持异步播放本地音频文件。适用于任务流程中需要音效提示或语音播放的场景。
    内部维护事件标志以支持多线程/异步控制。
    """

    player_events: dict = {}
    melody_events: asyncio.Event = asyncio.Event()

    @staticmethod
    async def audio_player(audio_file: str) -> None:
        """
        异步播放音频文件。

        使用 pygame.mixer 异步播放指定的本地音频文件，适用于提示音或语音播报。

        Parameters
        ----------
        audio_file : str
            音频文件的本地路径。

        Notes
        -----
        - 初始化 pygame.mixer 模块并加载音频资源。
        - 设置音量为最大值（1.0）。
        - 进入播放循环，判断播放是否结束。
        - 使用 `pygame.time.Clock().tick(10)` 控制轮询频率，避免 CPU 占用过高。
        """
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.set_volume(1.0)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)


if __name__ == '__main__':
    pass
