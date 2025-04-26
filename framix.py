###########################################################
#                                                         #
#                                                         #
# ███████╗ ██████╗   █████╗      ███╗   ███╗ ██╗ ██╗  ██╗ #
# ██╔════╝ ██╔══██╗ ██╔══██╗     ████╗ ████║ ██║ ╚██╗██╔╝ #
# █████╗   ██████╔╝ ███████║     ██╔████╔██║ ██║  ╚███╔╝  #
# ██╔══╝   ██╔══██╗ ██╔══██║     ██║╚██╔╝██║ ██║  ██╔██╗  #
# ██║      ██║  ██║ ██║  ██║     ██║ ╚═╝ ██║ ██║ ██╔╝ ██╗ #
# ╚═╝      ╚═╝  ╚═╝ ╚═╝  ╚═╝     ╚═╝     ╚═╝ ╚═╝ ╚═╝  ╚═╝ #
#                                                         #
#              Welcome to the Framix Engine!              #
#                                                         #
###########################################################

# ==== Notes: 版权申明 ====
# 版权所有 (c) 2024  Framix(画帧秀)
# 此文件受 Framix(画帧秀) 许可证的保护。您可以在 LICENSE.md 文件中查看详细的许可条款。

# ==== Notes: License ====
# Copyright (c) 2024  Framix(画帧秀)
# This file is licensed under the Framix(画帧秀) License. See the LICENSE.md file for more details.

# ==== Notes: ライセンス ====
# Copyright (c) 2024  Framix(画帧秀)
# このファイルは Framix(画帧秀) ライセンスの下でライセンスされています。詳細は LICENSE.md ファイルを参照してください。

"""
版权所有 (c) 2024  Framix(画帧秀)
此文件受 Framix(画帧秀) 许可证的保护。您可以在 LICENSE.md 文件中查看详细的许可条款。

Copyright (c) 2024  Framix(画帧秀)
This file is licensed under the Framix(画帧秀) License. See the LICENSE.md file for more details.

# Copyright (c) 2024  Framix(画帧秀)
# このファイルは Framix(画帧秀) ライセンスの下でライセンスされています。詳細は LICENSE.md ファイルを参照してください。
"""

__all__ = ["Clipix", "Alynex"]  # 接口

# ====[ 内置模块 ]====
import os
import re
import sys
import json
import time
import uuid
import stat
import signal
import shutil
import random
import typing
import inspect
import asyncio
import datetime
import tempfile

# ====[ 第三方库 ]====
import cv2
import numpy
import aiofiles

# ====[ from: 内置模块 ]====
from pathlib import Path
from functools import partial
from multiprocessing import freeze_support
from concurrent.futures import ProcessPoolExecutor

# ====[ from: 第三方库 ]====
from loguru import logger
from rich.prompt import Prompt
from PIL import (
    Image, ImageDraw, ImageFont
)

# ====[ from: 本地模块 ]====
from engine.device import Device
from engine.switch import Switch
from engine.manage import (
    ScreenMonitor, SourceMonitor,
    AsyncAnimationManager, Manage
)
from engine.medias import (
    Record, Player
)
from engine.terminal import Terminal
from engine.tinker import (
    Craft, Search, Active, Review, FramixError
)
from nexacore.cubicle import DB
from nexacore.argument import Wind
from nexacore.design import Design
from nexacore.parser import Parser
from nexacore.profile import (
    Deploy, Option
)
from nexaflow import const
from nexaflow import toolbox
from nexaflow.video import (
    VideoObject, VideoFrame
)
from nexaflow.report import Report
from nexaflow.cutter.cutter import VideoCutter
from nexaflow.hook import (
    FrameSizeHook, FrameSaveHook, PaintCropHook, PaintOmitHook
)
from nexaflow.classifier.base import ClassifierResult
from nexaflow.classifier.keras_classifier import KerasStruct

_T = typing.TypeVar("_T")  # 定义类型变量


# 信号处理器
def signal_processor(*_, **__) -> None:
    """
    程序信号处理函数，用于响应中断（如 Ctrl+C）时的清理与退出操作。

    该函数通常注册为信号处理器，用于在用户主动中止程序执行时，
    优雅地清理状态、打印退出提示，并安全终止进程。

    Parameters
    ----------
    *_, **__ : Any
        占位参数，用于兼容 signal.signal 的回调签名要求。
    """
    Design.exit()
    sys.exit(Design.closure())


class Missions(object):
    """
    Missions Engine

    该类作为 Framix 的核心，负责协调模型路径、运行参数、脚本配置、进程分配等多项输入，构建统一的任务执行上下文。
    内部集成 Design 对象作为标签与结构控制中枢，并管理多个临时路径与资源入口，支持大规模异步视频分析与处理。
    """

    __design: typing.Optional["Design"] = None

    # """Initialization"""
    def __init__(self, wires: list, level: str, power: int, *args, **kwargs):
        self.wires = wires  # 命令参数
        self.level = level  # 日志级别
        self.power = power  # 最大进程

        self.design = Design(self.level)
        self.animation = AsyncAnimationManager()

        self.flick, self.carry, self.fully, self.speed, self.basic, self.keras, *_ = args
        *_, self.alone, self.whist, self.alike, self.shine, self.group = args

        self.atom_total_temp: str = kwargs["atom_total_temp"]
        self.main_share_temp: str = kwargs["main_share_temp"]
        self.main_total_temp: str = kwargs["main_total_temp"]
        self.view_share_temp: str = kwargs["view_share_temp"]
        self.view_total_temp: str = kwargs["view_total_temp"]

        self.initial_option: str = kwargs["initial_option"]
        self.initial_deploy: str = kwargs["initial_deploy"]
        self.initial_script: str = kwargs["initial_script"]

        self.adb: str = kwargs["adb"]
        self.fmp: str = kwargs["fmp"]
        self.fpb: str = kwargs["fpb"]

    @property
    def design(self) -> typing.Optional["Design"]:
        """
        获取当前的标注设计对象。

        该属性提供对 Missions 内部 Design 配置的访问，通常用于读取标签样式、
        颜色映射、分类结构等标注语义配置。

        Returns
        -------
        Optional[Design]
            当前使用的 Design 实例，若未设置则为 None。
        """
        return self.__design

    @design.setter
    def design(self, value: typing.Optional["Design"]) -> None:
        """
        设置标注设计对象。

        用于注入或更新 Missions 所使用的 Design 配置，控制标签结构与展示样式等。

        Parameters
        ----------
        value : Optional[Design]
            要设置的 Design 实例，可为 None 表示清除或重置标注配置。
        """
        self.__design = value

    # """Child Process"""
    def amazing(self, option: "Option", deploy: "Deploy", vision: str, *args) -> "_T":
        """
        异步分析视频的子进程方法。

        在异步任务调度器中运行，负责加载 Keras 模型并调用分析器执行视频分析任务。
        该方法应在子进程中运行，以避免阻塞主事件循环。

        Parameters
        ----------
        option : Option
            包含模型路径与执行选项的配置对象。

        deploy : Deploy
            视频处理相关配置对象，封装各类路径与处理参数。

        vision : str
            待分析的视频文件路径。

        *args : Any
            传递给分析器的附加参数，需为基本类型或可序列化对象。

        Returns
        -------
        Any
            视频分析任务的执行结果，返回值取决于分析器的实现。

        Notes
        -----
        - 推荐传入参数为字符串、数字、列表等基本数据类型，避免复杂对象。
        - 该方法应在 `asyncio.subprocess` 或 `ProcessPoolExecutor` 中调用，用于并行分析任务。
        - 内部使用 Alynex 实例加载 Keras 模型，并异步调用其 `ask_analyzer` 方法处理视频数据。
        """
        loop = asyncio.new_event_loop()  # Note: 子进程内必须创建新的事件循环

        matrix = option.model_place if self.keras else None

        alynex = Alynex(matrix, option, deploy, Design(self.level))

        try:
            loop.run_until_complete(alynex.ask_model_load())
        except FramixError:
            pass

        loop_complete = loop.run_until_complete(
            alynex.ask_analyzer(vision, *args)
        )
        return loop_complete

    # """Child Process"""
    def bizarre(self, option: "Option", deploy: "Deploy", vision: str, *args) -> "_T":
        """
        异步执行视频分析的子进程方法。

        该方法由异步进程执行器调用，使用 Alynex 工具对指定视频进行分析处理。
        适用于批量视频分析、模型推理任务的异步分发场景。

        Parameters
        ----------
        option : Option
            含有模型路径、任务模式等运行时配置的选项对象。

        deploy : Deploy
            视频处理相关的配置对象，包含路径、分辨率等参数。

        vision : str
            需要分析的视频文件路径。

        *args : Any
            传递给 Alynex 分析器的其他附加参数，须确保可序列化。

        Returns
        -------
        Any
            视频处理后的分析结果，类型由 Alynex 工具返回值决定。

        Notes
        -----
        - 请避免传入复杂或不可序列化的对象。
        - 方法内部基于配置初始化 Alynex 实例，并调用其 `ask_exercise()` 进行视频处理。
        - 适用于异步任务池、子进程调度环境。
        """
        loop = asyncio.new_event_loop()  # Note: 子进程内必须创建新的事件循环

        alynex = Alynex(None, option, deploy, Design(self.level))

        loop_complete = loop.run_until_complete(
            alynex.ask_exercise(vision, *args)
        )

        return loop_complete

    @staticmethod
    async def enforce(db: "DB", style: str, total: str, title: str, nest: str) -> None:
        """
        异步插入分析数据到数据库，并确保表结构已创建。

        此方法会根据提供的信息（分析方式、报告路径、标题、嵌套标识）将数据写入数据库，
        在写入前自动检查并创建所需的表结构。

        Parameters
        ----------
        db : DB
            数据库连接对象，必须实现 `create` 和 `insert` 方法。

        style : str
            分析方式的标识，用于标注数据的来源或处理流程类型。

        total : str
            报告存储的根目录路径，用于记录结果输出位置。

        title : str
            数据集标题或报告名，通常用于标识当前分析任务。

        nest : str
            嵌套标识，用于记录子任务、子路径或层级结构。
        """
        await db.create(column_list := ["style", "total", "title", "nest"])
        await db.insert(column_list, [style, total, title, nest])

    async def fst_track(
            self, deploy: "Deploy", clipix: "Clipix", task_list: list[list]
    ) -> tuple[list, list]:
        """
        异步执行视频处理与追踪，包括内容提取与长度平衡操作。

        根据部署参数从多个视频中提取必要信息，并在需要时统一其播放长度。
        处理结束后返回原始视频与处理状态信息。

        Parameters
        ----------
        deploy : Deploy
            部署配置对象，包含起始时间、结束时间、限制时长、帧率等视频处理参数。

        clipix : Clipix
            视频处理工具对象，提供内容提取与长度平衡等方法。

        task_list : list of list
            视频处理任务列表，每项包含视频路径及其处理上下文参数。

        Returns
        -------
        tuple of list
            返回两个列表：
            - 原始视频提取信息列表。
            - 指示处理状态或长度平衡信息的结果列表。

        Notes
        -----
        - 此函数为异步函数，需在异步事件循环中运行。
        - 内部使用 `asyncio.gather()` 并行处理多个视频。
        - 要求每个视频符合 `Deploy` 所定义的帧率与时间限制标准。
        - 临时文件将在处理后被自动清理。
        - 所有异常应在处理过程中被捕获，避免中断任务流程。

        Workflow
        --------
        1. 解析部署配置中的时间范围与帧率。
        2. 异步调用 `clipix.vision_content()` 提取视频信息。
        3. 可选调用 `clipix.vision_balance()` 对视频长度进行统一处理。
        4. 删除处理过程中产生的临时文件。
        5. 汇总并返回所有视频的处理结果。
        """
        looper = asyncio.get_running_loop()

        # Video information
        start_ms = Parser.parse_mills(deploy.start)
        close_ms = Parser.parse_mills(deploy.close)
        limit_ms = Parser.parse_mills(deploy.limit)

        content_list = await asyncio.gather(
            *(clipix.vision_content(video_temp, start_ms, close_ms, limit_ms)
              for video_temp, *_ in task_list)
        )

        for (rlt, avg, dur, org, pnt), (video_temp, *_) in zip(content_list, task_list):
            vd_start, vd_close, vd_limit = pnt["start"], pnt["close"], pnt["limit"]
            logger.debug(f"视频尺寸: {list(org)}")
            logger.debug(f"实际帧率: [{rlt}] 平均帧率: [{avg}] 转换帧率: [{deploy.frate}]")
            logger.debug(f"视频时长: [{dur:.6f}] [{Parser.parse_times(dur)}]")
            logger.debug(f"视频剪辑: start=[{vd_start}] close=[{vd_close}] limit=[{vd_limit}]")
            self.design.content_pose(
                rlt, avg, f"{dur:.6f}", org, vd_start, vd_close, vd_limit, video_temp, deploy.frate
            )

        *_, durations, originals, indicates = zip(*content_list)

        # Video Balance
        eliminate = []
        if self.alike and len(task_list) > 1:
            logger.debug(tip := f"平衡时间: [{(standard := min(durations)):.6f}] [{Parser.parse_times(standard)}]")
            self.design.show_panel(tip, Wind.STANDARD)

            video_dst_list = await asyncio.gather(
                *(clipix.vision_balance(duration, standard, video_src, deploy.frate)
                  for duration, (video_src, *_) in zip(durations, task_list))
            )

            panel_blc_list = []
            for (video_idx, (video_dst, video_blc)), (video_src, *_) in zip(enumerate(video_dst_list), task_list):
                logger.debug(tip := f"{video_blc}")
                panel_blc_list.append(tip)
                eliminate.append(looper.run_in_executor(None, os.remove, video_src))
                task_list[video_idx][0] = video_dst
            self.design.show_panel("\n".join(panel_blc_list), Wind.TAILOR)

        await asyncio.gather(*eliminate, return_exceptions=True)

        return originals, indicates

    async def fst_waves(
            self, deploy: "Deploy", clipix: "Clipix", task_list: list[list], originals: list
    ) -> list:
        """
        异步执行视频的过滤与改进操作。

        根据部署配置，对原始视频应用帧率、色彩、模糊、锐化等多个视频增强过滤器，
        以提升视频质量或适应后续分析要求。

        Parameters
        ----------
        deploy : Deploy
            视频处理配置对象，定义帧率、模糊、锐化等过滤器参数。

        clipix : Clipix
            视频处理工具类，封装底层 `ffmpeg` 命令，执行实际的视频增强操作。

        task_list : list of list
            视频任务参数列表，每项包含视频路径和相关任务配置。

        originals : list
            原始视频文件路径列表，用于提取与增强处理。

        Returns
        -------
        tuple
            包含一个处理后的视频列表，表示增强处理完成的视频路径或状态信息。

        Notes
        -----
        - 本函数为异步函数，应在异步事件循环中调用。
        - 使用 `asyncio.gather()` 并发执行多个视频过滤任务。
        - 所有视频必须符合 `Deploy` 中定义的处理标准。
        - 异常在内部被捕获和处理，确保流程稳定不中断。

        Workflow
        --------
        1. 根据 `deploy` 参数初始化过滤器列表（如帧率、模糊、色彩等）。
        2. 异步执行每个视频的 `clipix.vision_improve()` 操作。
        3. 记录每个过滤任务的处理日志。
        4. 返回处理后的视频结果列表。
        """
        filters = [f"fps={deploy.frate}"]
        if self.speed:
            filters += [] if deploy.color else [f"format=gray"]

        filters = filters + [f"gblur=sigma={gauss}"] if (gauss := deploy.gauss) else filters
        filters = filters + [f"unsharp=luma_amount={grind}"] if (grind := deploy.grind) else filters

        video_filter_list = await asyncio.gather(
            *(clipix.vision_improve(deploy, original, filters) for original in originals)
        ) if self.speed else [filters for _ in originals]

        panel_filter_list = []
        for flt, (video_temp, *_) in zip(video_filter_list, task_list):
            logger.debug(tip := f"视频过滤: {flt} {os.path.basename(video_temp)}")
            panel_filter_list.append(tip)
        self.design.show_panel("\n".join(panel_filter_list), Wind.FILTER)

        return video_filter_list

    async def als_speed(
            self, deploy: "Deploy", clipix: "Clipix", report: "Report", task_list: list[list]
    ) -> None:
        """
        异步执行视频的内容优化与速度适配处理。

        此函数根据部署配置对一批视频执行滤镜处理、尺寸调整、帧率转换等操作，
        并通过报告对象记录每一步的处理进度与统计信息。

        Parameters
        ----------
        deploy : Deploy
            视频处理参数配置，定义帧率、画面尺寸、滤镜强度等调整行为。

        clipix : Clipix
            视频引擎工具，提供底层的图像处理与帧生成能力。

        report : Report
            报告管理器，负责收集处理数据并生成最终可视化分析报告。

        task_list : list of list
            视频任务列表，每项包含视频路径及其对应的处理参数集。

        Notes
        -----
        - 本方法为异步方法，建议在 `asyncio` 事件循环中调用。
        - 处理过程自动并发执行多个视频任务，提升整体效率。
        - 支持自动检测视频的格式、维度，并根据配置标准进行统一化输出。
        - 所有异常将在处理阶段内部捕获，确保主流程稳定不中断。

        Workflow
        --------
        1. 初始化部署参数与处理路径。
        2. 调用 `als_track` 执行视频信息提取。
        3. 使用 `als_waves` 应用帧率、色彩与模糊等滤镜处理。
        4. 使用 `clipix.pixels()` 对视频执行帧提取与尺寸调整。
        5. 并发处理所有任务，收集处理结果。
        6. 调用 `report.render()` 渲染分析图表，输出到指定报告目录。
        """
        logger.debug(f"**<* 光速穿梭 *>**")
        self.design.show_panel(Wind.SPEED_TEXT, Wind.SPEED)
        await self.design.pulse_track()

        originals, indicates = await self.fst_track(deploy, clipix, task_list)

        video_filter_list = await self.fst_waves(deploy, clipix, task_list, originals)

        video_target_list = [
            (flt, frame_path) for flt, (*_, frame_path, _, _) in zip(video_filter_list, task_list)
        ]

        await self.animation.start(self.design.frame_grid_initializer)

        detach_result = await asyncio.gather(
            *(clipix.pixels(
                Switch.ask_video_detach, video_filter, video_temp, target, **points
            ) for (video_filter, target), (video_temp, *_), points in zip(video_target_list, task_list, indicates))
        )

        await self.animation.stop()

        for detach, (video_temp, *_) in zip(detach_result, task_list):
            logger.debug(detach)
            for message in reversed(detach.splitlines()):
                if matcher := re.search(r"frame.*fps.*speed.*", message):
                    discover: typing.Any = lambda x: re.findall(r"(\w+)=\s*([\w.\-:/x]+)", x)
                    fmt_msg = " ".join([f"{k}={v}" for k, v in discover(matcher.group())])
                    self.design.show_panel(fmt_msg, Wind.METRIC)
                    break
                elif re.search(r"Error", message, re.IGNORECASE):
                    self.design.show_panel(message, Wind.KEEPER)
                    break

        async def render_speed(
                todo_list: list[list[typing.Union[str, "asyncio.subprocess.Process", None]]]
        ) -> tuple:

            start, end, cost, scores, struct = 0, 0, 0, None, None
            *_, total_path, title, query_path, query, frame_path, extra_path, proto_path = todo_list

            result = {
                "total": os.path.basename(total_path),
                "title": title,
                "query": query,
                "stage": {"start": start, "end": end, "cost": cost},
                "frame": os.path.basename(frame_path),
                "style": "speed"
            }

            await report.load(result)
            logger.debug(f"Speeder: {(nest := json.dumps(result, ensure_ascii=False))}")

            return result, nest

        await self.animation.start(self.design.boot_html_renderer)

        render_result = await asyncio.gather(
            *(render_speed(todo_list) for todo_list in task_list)
        )

        await self.animation.stop()

        async with DB(Path(report.reset_path) / const.DB_FILES_NAME) as db:
            await asyncio.gather(
                *(self.enforce(db, rs["style"], rs["total"], rs["title"], ns)
                  for rs, ns in render_result)
            )
            logger.debug(f"DB: {render_result}")

    async def als_basic_or_keras(
            self, deploy: "Deploy", clipix: "Clipix", report: "Report", task_list: list[list], **kwargs
    ) -> None:
        """
        异步执行视频的 Keras 模型分析或基础模式分析任务。

        根据部署参数执行视频预处理与分析流程，可选择基于深度学习模型（Keras）或传统分析方式处理。
        处理过程包括帧率调整、滤镜应用、尺寸标准化、拆帧及报告生成。

        Parameters
        ----------
        deploy : Deploy
            视频处理配置对象，包含帧率、尺寸、色彩等调整参数。

        clipix : Clipix
            视频处理引擎，提供滤镜、拆帧与像素提取等核心功能。

        report : Report
            报告引擎对象，用于汇总分析数据并生成 HTML 报告。

        task_list : list of list
            待分析的视频任务列表，每项包含视频路径及对应配置。

        **kwargs : dict
            额外配置项，包括：
                option : Option
                    全局运行参数，控制分析方式与行为。
                alynex : Alynex
                    分析工具实例，包含 Keras 模型状态与分析方法。

        Notes
        -----
        - 本方法需在异步环境下运行。
        - 若 `alynex.ks.model` 存在，将优先使用深度模型执行视频分析。
        - 所有中间视频文件将在处理结束后自动清除。
        - 并发任务通过 `asyncio.gather` 实现，提升性能。

        Workflow
        --------
        1. 提取 `option` 和 `alynex` 实例。
        2. 执行 `als_track` 获取原始视频和指示信息。
        3. 应用 `als_waves` 对视频进行帧率与滤镜预处理。
        4. 使用 `clipix.pixels()` 拆解视频帧并标准化尺寸。
        5. 清除中间缓存文件。
        6. 根据模型状态执行 Keras 模型或基础分析流程。
        7. 渲染图表并生成完整分析报告。
        """
        looper = asyncio.get_running_loop()

        option: "Option" = kwargs["option"]
        alynex: "Alynex" = kwargs["alynex"]

        if alynex.ks.model:
            logger.debug(f"**<* 思维导航 *>**")
            self.design.show_panel(Wind.KERAS_TEXT, Wind.KERAS)
        else:
            logger.debug(f"**<* 基石阵地 *>**")
            self.design.show_panel(Wind.BASIC_TEXT, Wind.BASIC)

        await self.design.collapse_star_expanded()

        originals, indicates = await self.fst_track(deploy, clipix, task_list)

        video_filter_list = await self.fst_waves(deploy, clipix, task_list, originals)

        video_target_list = [
            (flt, os.path.join(
                os.path.dirname(video_temp), f"vision_fps{deploy.frate}_{random.randint(100, 999)}.mp4")
             ) for flt, (video_temp, *_) in zip(video_filter_list, task_list)
        ]

        await self.animation.start(self.design.frame_grid_initializer)

        change_result = await asyncio.gather(
            *(clipix.pixels(
                Switch.ask_video_change, video_filter, video_temp, target, **points
            ) for (video_filter, target), (video_temp, *_), points in zip(video_target_list, task_list, indicates))
        )

        await self.animation.stop()

        eliminate = []
        for change, (video_temp, *_) in zip(change_result, task_list):
            logger.debug(change)
            for message in reversed(change.splitlines()):
                if matcher := re.search(r"frame.*fps.*speed.*", message):
                    discover: typing.Any = lambda x: re.findall(r"(\w+)=\s*([\w.\-:/x]+)", x)
                    fmt_msg = " ".join([f"{k}={v}" for k, v in discover(matcher.group())])
                    self.design.show_panel(fmt_msg, Wind.METRIC)
                    break
                elif re.search(r"Error", message, re.IGNORECASE):
                    self.design.show_panel(message, Wind.KEEPER)
                    break
            eliminate.append(
                looper.run_in_executor(None, os.remove, video_temp)
            )
        await asyncio.gather(*eliminate, return_exceptions=True)

        if alynex.ks.model:
            deploy.view_deploy()
            await self.design.neural_sync_loading()

        monitor = SourceMonitor()
        monitor_task = asyncio.create_task(monitor.monitor_stable())
        await self.design.multi_load_ripple_vision(monitor)
        await monitor_task
        logger.debug(f"系统负载检测通过: {monitor.usages}")

        if len(task_list) == 1:
            task = [
                alynex.ask_analyzer(target, frame_path, extra_path, src_size)
                for (_, target), src_size, (*_, frame_path, extra_path, _)
                in zip(video_target_list, originals, task_list)
            ]
            futures = await asyncio.gather(*task)

        else:
            await random.choice(
                [self.design.boot_process_matrix, self.design.boot_process_sequence]
            )(min(5, max(2, len(task_list))))

            this_level = self.level
            self.level = "ERROR"
            func = partial(self.amazing, option, deploy)
            with ProcessPoolExecutor(self.power, None, Active.active, ("ERROR",)) as exe:
                task = [
                    looper.run_in_executor(exe, func, target, frame_path, extra_path, src_size)
                    for (_, target), src_size, (*_, frame_path, extra_path, _)
                    in zip(video_target_list, originals, task_list)
                ]
                futures = await asyncio.gather(*task)
            self.level = this_level

        atom_tmp = await Craft.achieve(self.atom_total_temp)

        async def render_keras(
                future: "Review",
                todo_list: list[list[typing.Union[str, "asyncio.subprocess.Process", None]]]
        ) -> tuple:

            start, end, cost, scores, struct = future.material
            *_, total_path, title, query_path, query, frame_path, extra_path, proto_path = todo_list

            result = {
                "total": os.path.basename(total_path),
                "title": title,
                "query": query,
                "stage": {"start": start, "end": end, "cost": f"{cost:.5f}"},
                "frame": os.path.basename(frame_path)
            }

            if struct:
                stages_inform = await report.ask_draw(
                    scores, struct, proto_path, atom_tmp, deploy.boost
                )
                result["extra"] = os.path.basename(extra_path)
                result["proto"] = os.path.basename(stages_inform)
                result["style"] = "keras"
            else:
                result["style"] = "basic"

            await report.load(result)
            logger.debug(f"Restore: {(nest := json.dumps(result, ensure_ascii=False))}")

            return result, nest

        await self.animation.start(self.design.boot_html_renderer)

        render_result = await asyncio.gather(
            *(render_keras(future, todo_list) for future, todo_list in zip(futures, task_list) if future)
        )

        await self.animation.stop()

        for resp, _ in render_result:
            if rp := resp.get("proto", None):
                logger.debug(tip := f"模版引擎渲染完成 {Path(rp).name}")
                self.design.show_panel(tip, Wind.REPORTER)

        async with DB(Path(report.reset_path) / const.DB_FILES_NAME) as db:
            await asyncio.gather(
                *(self.enforce(db, rs["style"], rs["total"], rs["title"], ns)
                  for rs, ns in render_result)
            )
            logger.debug(f"DB: {render_result}")

    async def combine(self, report: "Report") -> None:
        """
        异步生成组合报告。

        根据报告对象中提供的数据范围与配置参数，动态选择合适的合并方法生成综合分析报告。
        支持多场景对比与汇总视图，适用于复杂测试或多源视频任务。

        Parameters
        ----------
        report : Report
            报告引擎对象，包含报告路径、数据范围以及合并配置。
        """
        if report.range_list:
            function = getattr(self, "combine_view" if self.speed else "combine_main")
            return await function([os.path.dirname(report.total_path)])

        logger.debug(tip := f"没有可以生成的报告")
        return self.design.show_panel(tip, Wind.KEEPER)

    async def combine_crux(self, share_temp: str, total_temp: str, merge: list) -> None:
        """
        异步生成汇总报告。

        根据共享模板与总模板路径，整合多个子报告内容，最终生成完整的汇总分析文档。

        Parameters
        ----------
        share_temp : str
            共享内容模板路径，用于插入所有子报告中共通的页面或描述元素。

        total_temp : str
            汇总页面模板路径，用于构建最终的总报告布局与展示结构。

        merge : list of str
            子报告路径列表，所有待整合的报告资源将统一纳入汇总流程中。

        Notes
        -----
        - 此函数需在异步环境中执行。
        - 所有模板加载操作均通过异步文件读取完成。
        - 若模板文件缺失或格式错误，将抛出异常并记录日志。
        - 支持多源报告的批量合并，适用于跨任务、跨模型的统一视图整合。

        Workflow
        --------
        1. 异步读取共享模板和汇总模板内容。
        2. 加载并解析所有待合并的子报告内容。
        3. 拼接共享内容与各子报告片段，构造统一报告体。
        4. 渲染汇总模板并生成最终 HTML 报告。
        5. 记录汇总路径与操作日志。
        """
        template_list = await asyncio.gather(
            *(Craft.achieve(i) for i in (share_temp, total_temp))
        )

        share_form, total_form = template_list

        logger.debug(tip := f"正在生成汇总报告 ...")
        self.design.show_panel(tip, Wind.REPORTER)

        await self.animation.start(self.design.render_horizontal_pulse)

        resp_state = await asyncio.gather(
            *(Report.ask_create_total_report(
                m, self.group, share_form, total_form) for m in merge), return_exceptions=True
        )

        await self.animation.stop()

        for resp in resp_state:
            logger.debug(resp)
            if isinstance(resp, Exception):
                self.design.show_panel(resp, Wind.KEEPER)
            else:
                self.design.show_panel(f"成功生成汇总报告 {(state := Path(resp)).name}", Wind.REPORTER)
                self.design.show_file(state)

    # """时空纽带分析系统"""
    async def combine_view(self, merge: list) -> None:
        """
        合并视图数据。
        """
        await self.combine_crux(
            self.view_share_temp, self.view_total_temp, merge
        )

    # """时序融合分析系统"""
    async def combine_main(self, merge: list) -> None:
        """
        合并视图数据。
        """
        await self.combine_crux(
            self.main_share_temp, self.main_total_temp, merge
        )

    # """视频解析探索"""
    async def video_file_task(self, video_file_list: list, option: "Option", deploy: "Deploy") -> None:
        """
        异步处理视频文件任务，并根据配置选项执行相应的分析流程。

        通过分析视频列表，执行基本或深度学习分析操作，并最终生成分析报告。

        Parameters
        ----------
        video_file_list : list of str
            视频文件路径列表，包含所有待分析的原始视频资源。

        option : Option
            配置选项对象，定义任务运行的各类参数与模型选型策略。

        deploy : Deploy
            部署配置对象，提供具体的处理规则，如帧率、剪辑时长、输出格式等。

        Notes
        -----
        - 若视频列表为空，函数将直接中止并记录日志信息。
        - 分析模式由 `option.speed` 控制：
            - 若启用，执行快速分析路径（als_speed）。
            - 否则使用 Keras 模型进行深度分析（als_keras）。
        - 所有模型加载及处理操作均封装为异步任务。
        - 支持异常捕获与错误日志打印，确保运行稳定性。

        Workflow
        --------
        1. 过滤并验证视频文件有效性。
        2. 初始化 Clipix 和 Report 对象，用于处理与结果管理。
        3. 将视频复制到报告路径中，生成任务队列。
        4. 根据配置执行快速分析或 Keras 深度分析。
        5. 渲染最终分析报告并完成任务。
        """

        # Notes: Start from here
        if not (video_file_list := [
            video_file for video_file in video_file_list if os.path.isfile(video_file)]
        ):
            logger.debug(tip := f"没有有效任务")
            return self.design.show_panel(tip, Wind.KEEPER)

        clipix = Clipix(self.fmp, self.fpb)
        report = Report(option.total_place)
        report.title = f"{const.DESC}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

        # Notes: Profession
        task_list = []
        for video_file in video_file_list:
            report.query = f"{os.path.basename(video_file).split('.')[0]}_{time.strftime('%Y%m%d%H%M%S')}"
            new_video_path = os.path.join(report.video_path, os.path.basename(video_file))
            shutil.copy(video_file, new_video_path.format())
            task_list.append(
                [new_video_path, None, report.total_path, report.title, report.query_path,
                 report.query, report.frame_path, report.extra_path, report.proto_path]
            )

        # Notes: Analyzer
        if self.speed:
            await self.als_speed(
                deploy, clipix, report, task_list
            )
        else:
            matrix = option.model_place if self.keras else None
            alynex = Alynex(matrix, option, deploy, self.design)
            try:
                await alynex.ask_model_load()
            except FramixError as e:
                logger.debug(e)
                self.design.show_panel(e, Wind.KEEPER)

            await self.als_basic_or_keras(
                deploy, clipix, report, task_list, option=option, alynex=alynex
            )

        # Notes: Create Report
        await self.combine(report)

    # """影像堆叠导航"""
    async def video_data_task(self, video_data_list: list, option: "Option", deploy: "Deploy") -> None:
        """
        异步处理视频数据任务，并根据配置执行快速或深度分析流程。

        通过 Search 工具定位视频数据源，组织分析任务并输出处理报告。

        Parameters
        ----------
        video_data_list : list of str
            视频数据路径组成的列表，表示待处理的输入数据源。

        option : Option
            分析配置对象，包含模型选择、运行参数等选项控制信息。

        deploy : Deploy
            视频处理部署对象，定义处理参数，如时间区间、输出格式、帧率限制等。

        Notes
        -----
        - 利用 `Search` 类对输入路径加速检索，自动组织数据结构。
        - 每项任务对应一个报告输出目录，支持并行处理多个数据集。
        - 若 `option.speed=True`，使用快速路径分析 (`als_speed`)。
        - 若使用 Keras 模型，则通过 `als_keras` 启动深度学习分析流程。
        - 所有异常（如搜索失败或模型加载异常）都将被记录且不中断整体流程。

        Workflow
        --------
        1. 使用 `Search` 工具解析并验证数据源结构。
        2. 为每项视频数据创建对应的报告目录和任务清单。
        3. 判断当前配置，选择快速分析或 Keras 深度分析流程。
        4. 执行任务分析并生成 HTML 报告。
        """

        async def load_entries() -> "typing.AsyncGenerator":
            """
            加载视频数据条目。
            """
            for video_data in video_data_list:
                logger.debug(f"查找文件夹: {video_data}")
                if isinstance(search_result := search.accelerate(video_data), Exception):
                    logger.debug(search_result)
                    self.design.show_panel(search_result, Wind.KEEPER)
                    continue
                yield search_result[0]

        # Notes: Start from here
        search = Search()
        clipix = Clipix(self.fmp, self.fpb)

        # Notes: Profession
        async for entries in load_entries():
            if entries:
                report = Report(option.total_place)
                for entry in entries:
                    report.title = entry.title
                    task_list = []
                    for video in entry.sheet:
                        report.query = video["query"]
                        video_path = video["video"]
                        video_name = os.path.basename(video_path)
                        shutil.copy(video_path, report.video_path)
                        new_video_path = os.path.join(report.video_path, video_name)
                        task_list.append(
                            [new_video_path, None, report.total_path, report.title, report.query_path,
                             report.query, report.frame_path, report.extra_path, report.proto_path]
                        )

                    # Notes: Analyzer
                    if self.speed:
                        await self.als_speed(deploy, clipix, report, task_list)
                    else:
                        matrix = option.model_place if self.keras else None
                        alynex = Alynex(matrix, option, deploy, self.design)
                        try:
                            await alynex.ask_model_load()
                        except FramixError as e:
                            logger.debug(e)
                            self.design.show_panel(e, Wind.KEEPER)

                        await self.als_basic_or_keras(
                            deploy, clipix, report, task_list, option=option, alynex=alynex
                        )

                # Notes: Create Report
                await self.combine(report)

    # """模型训练大师"""
    async def train_model(self, video_file_list: list, option: "Option", deploy: "Deploy") -> None:
        """
        异步执行模型训练任务，对输入视频进行预处理后启动单任务或多任务训练流程。

        Parameters
        ----------
        video_file_list : list of str
            待训练的视频文件路径列表，需为有效的媒体文件。

        option : Option
            配置选项对象，包含模型路径、训练模式、日志等级等选项参数。

        deploy : Deploy
            部署配置对象，定义训练过程中涉及的视频处理参数，如帧率、剪辑时间、输出路径等。

        Notes
        -----
        - 所有无效或不存在的视频路径将被自动过滤。
        - 使用 `Clipix` 处理视频结构，生成任务清单后依赖 `als_track` 与 `als_waves` 进行预处理。
        - 若任务仅包含一个视频，则采用直接异步训练模式。
        - 若任务数量大于 1，则采用多进程训练方式，并设置日志等级为 ERROR，避免输出干扰。
        - 所有临时视频文件将在训练完成后清理，节省存储空间。

        Workflow
        --------
        1. 验证输入路径并构建处理任务列表。
        2. 初始化 `Clipix` 与 `Report`，并准备视频分析任务。
        3. 执行视频追踪和过滤操作，准备训练输入。
        4. 根据任务数量选择训练模式：
            - 单任务直接调用 `Alynex.ask_analyzer`
            - 多任务使用进程池并发执行训练任务
        5. 分析结束后清除中间产物，记录训练日志。
        """

        # Notes: Start from here
        if not (video_file_list := [
            video_file for video_file in video_file_list if os.path.isfile(video_file)]
        ):
            logger.debug(tip := f"没有有效任务")
            return self.design.show_panel(tip, Wind.KEEPER)

        looper = asyncio.get_running_loop()

        clipix = Clipix(self.fmp, self.fpb)
        report = Report(option.total_place)

        # Notes: Profession
        task_list = []
        for video_file in video_file_list:
            report.title = f"Model_{uuid.uuid4()}"
            new_video_path = os.path.join(report.video_path, os.path.basename(video_file))
            shutil.copy(video_file, new_video_path.format())
            task_list.append(
                [new_video_path, None, report.total_path, report.title, report.query_path,
                 report.query, report.frame_path, report.extra_path, report.proto_path]
            )

        originals, indicates = await self.fst_track(deploy, clipix, task_list)

        video_filter_list = await self.fst_waves(deploy, clipix, task_list, originals)

        video_target_list = [
            (flt, os.path.join(
                report.query_path, f"tmp_fps{deploy.frate}_{random.randint(10000, 99999)}.mp4")
             ) for flt, (video_temp, *_) in zip(video_filter_list, task_list)
        ]

        await self.animation.start(self.design.frame_grid_initializer)

        change_result = await asyncio.gather(
            *(clipix.pixels(
                Switch.ask_video_change, video_filter, video_temp, target, **points
            ) for (video_filter, target), (video_temp, *_), points in zip(video_target_list, task_list, indicates))
        )

        await self.animation.stop()

        eliminate = []
        for change, (video_temp, *_) in zip(change_result, task_list):
            logger.debug(change)
            for message in reversed(change.splitlines()):
                if matcher := re.search(r"frame.*fps.*speed.*", message):
                    discover: typing.Any = lambda x: re.findall(r"(\w+)=\s*([\w.\-:/x]+)", x)
                    fmt_msg = " ".join([f"{k}={v}" for k, v in discover(matcher.group())])
                    self.design.show_panel(fmt_msg, Wind.METRIC)
                    break
                elif re.search(r"Error", message, re.IGNORECASE):
                    self.design.show_panel(message, Wind.KEEPER)
                    break
            eliminate.append(
                looper.run_in_executor(None, os.remove, video_temp)
            )
        await asyncio.gather(*eliminate, return_exceptions=True)

        alynex = Alynex(None, option, deploy, self.design)

        monitor = SourceMonitor()
        monitor_task = asyncio.create_task(monitor.monitor_stable())
        await self.design.multi_load_ripple_vision(monitor)
        await monitor_task
        logger.debug(f"系统负载检测通过: {monitor.usages}")

        # Notes: Analyzer
        if len(task_list) == 1:
            task = [
                alynex.ask_exercise(target, query_path, src_size)
                for (_, target), src_size, (_, _, _, _, query_path, *_)
                in zip(video_target_list, originals, task_list)
            ]
            futures = await asyncio.gather(*task)

        else:
            await random.choice(
                [self.design.boot_process_matrix, self.design.boot_process_sequence]
            )(min(5, max(2, len(task_list))))

            this_level = self.level
            self.level = "ERROR"
            func = partial(self.bizarre, option, deploy)
            with ProcessPoolExecutor(self.power, None, Active.active, ("ERROR",)) as exe:
                task = [
                    looper.run_in_executor(exe, func, target, query_path, src_size)
                    for (_, target), src_size, (_, _, _, _, query_path, *_)
                    in zip(video_target_list, originals, task_list)
                ]
                futures = await asyncio.gather(*task)
            self.level = this_level

        await self.design.channel_animation()

        for future in futures:
            logger.debug(future)
            if isinstance(future, Exception):
                self.design.show_panel(future, Wind.KEEPER)
            else:
                self.design.show_panel(f"保存样本: {Path(future).name}", Wind.PROVIDER)

        await asyncio.gather(
            *(looper.run_in_executor(None, os.remove, target)
              for (_, target) in video_target_list)
        )

        Design.console.print()

    # """模型编译大师"""
    async def build_model(self, video_data_list: list, option: "Option", deploy: "Deploy") -> None:
        """
        异步执行模型构建任务，从视频数据目录中提取图像信息并构建模型结构。

        Parameters
        ----------
        video_data_list : list of str
            包含视频数据文件夹路径的列表，每个路径下需包含结构化的图像数据。

        option : Option
            模型构建选项，包含路径设置、日志等级、模型输出参数等配置。

        deploy : Deploy
            视频部署配置对象，用于提供图像格式、尺寸要求、路径约束等信息。

        Notes
        -----
        - 提供的数据目录需为有效的结构化图像路径，通常包含多个子序列。
        - 使用 `conduct` 函数检索图像路径，并按规则构建训练输入集。
        - 使用 `channel` 函数分析图片通道信息，推导图像维度和模型输入结构。
        - 支持自动选择构建模式：单任务模式或多进程并发构建。
        - 多进程模式下将日志等级提升至 ERROR，以避免并发输出干扰。

        Workflow
        --------
        1. 检查并过滤无效路径，生成合法任务列表。
        2. 检索每个路径下的图像结构，提取图像维度与通道信息。
        3. 构建模型构建参数，并调用 `Alynex.ks.build` 进行模型生成。
        4. 如果任务量大于 1，使用多进程池并发构建模型。
        5. 构建完成后输出成功信息或错误报告。
        """

        # Notes: Start from here
        if not (video_data_list := [
            video_data for video_data in video_data_list if os.path.isdir(video_data)]
        ):
            logger.debug(tip := f"没有有效任务")
            return self.design.show_panel(tip, Wind.KEEPER)

        await self.design.frame_stream_flux()

        async def conduct() -> list[str]:
            """
            遍历指定的视频数据目录，提取按数字命名的子目录路径列表。

            Returns
            -------
            list[str]
                按目录名称升序排列的不重复的数字子目录路径列表。

            Notes
            -----
            - 仅匹配目录名为纯数字的文件夹路径。
            - 使用 `dict.fromkeys()` 保证路径唯一性并保留顺序。
            - 用于构建后续图像学习数据源的预筛选入口。
            """
            search_file_list, search_dirs_list = [], []

            for root, dirs, files in os.walk(video_data, topdown=False):
                search_file_list.extend(
                    os.path.join(root, name) for name in files if name
                )
                search_dirs_list.extend(
                    os.path.join(root, name) for name in dirs if re.search(r"^\d+$", name)
                )

            if search_dirs_list:
                search_dirs_list.sort(key=lambda x: int(os.path.basename(x)))

            return list(dict.fromkeys(search_dirs_list))

        async def channel() -> tuple["numpy.ndarray", str, int]:
            """
            遍历所有图像目录，分析图像通道类型，并在设计面板中展示图像信息。

            Returns
            -------
            tuple[numpy.ndarray, str, int]
                - image_learn: 第一个读取成功的图像数据（用于建模）。
                - image_color: 图像颜色模式（"grayscale" 或 "rgb"）。
                - image_aisle: 图像通道数量（1 或 3）。

            Notes
            -----
            - 支持的图像格式包括：`.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.tiff`。
            - 对每张图像调用 `measure()` 进行通道识别，并将结果实时展示在设计面板。
            - 最终返回用于建模的首张图像及其颜色通道信息。
            """
            image_learn, image_color, image_aisle = None, "grayscale", 1

            for dirs in dirs_list:
                channel_list = []
                for name in os.listdir(dirs):
                    if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")):
                        image_color, image_aisle, image_debug = await measure(
                            image_learn := cv2.imread(os.path.join(dirs, name))
                        )
                        logger.debug(image_debug)
                        channel_list.append(image_debug)
                self.design.show_panel("\n".join(channel_list), Wind.DESIGNER)

            return image_learn, image_color, image_aisle

        async def measure(image: "numpy.ndarray") -> tuple[str, int, str]:
            """
            判断图像是灰度图还是彩色图，并返回通道类型及描述信息。

            Parameters
            ----------
            image : numpy.ndarray
                待分析的图像数据，通常为 OpenCV 加载图像。

            Returns
            -------
            tuple[str, int, str]
                - 图像颜色模式字符串（"grayscale" 或 "rgb"）
                - 通道数量（1 表示灰度图，3 表示 RGB 彩色图）
                - 可用于展示或日志的诊断描述信息

            Notes
            -----
            - 判断依据为图像的维度及 RGB 三通道内容是否相等。
            - 若三通道内容相同，则视为灰度图以 RGB 格式存储。
            """
            if image.ndim != 3:
                return "grayscale", 1, f"Image: {list(image.shape)} is grayscale image"

            if numpy.array_equal(image[:, :, 0], image[:, :, 1]) and numpy.array_equal(image[:, :, 1], image[:, :, 2]):
                return "grayscale", 1, f"Image: {list(image.shape)} is grayscale image, stored in RGB format"

            return "rgb", image.ndim, f"Image: {list(image.shape)} is color image"

        looper = asyncio.get_running_loop()

        alynex = Alynex(None, option, deploy, self.design)
        report = Report(option.total_place)

        # Notes: Profession
        task_list = []
        for video_data in video_data_list:
            logger.debug(tip := f"搜索文件夹: {os.path.basename(video_data)}")
            self.design.show_panel(tip, Wind.DESIGNER)
            if dirs_list := await conduct():
                logger.debug(tip := f"分类文件夹: {os.path.basename(cf_src := os.path.dirname(dirs_list[0]))}")
                self.design.show_panel(tip, Wind.DESIGNER)
                try:
                    ready_image, ready_color, ready_aisle = await channel()
                    image_shape = deploy.shape if deploy.shape else ready_image.shape
                except Exception as e:
                    logger.debug(e)
                    self.design.show_panel(e, Wind.KEEPER)
                    continue

                image_w, image_h = image_shape[:2]
                w, h = max(image_w, 10), max(image_h, 10)

                src_model_name = f"Gray" if ready_aisle == 1 else f"Hued"
                # new_model_name = f"Keras_{name}_W{w}_H{h}_{random.randint(10000, 99999)}.h5"
                new_model_name = f"Keras_{src_model_name}_W{w}_H{h}_{random.randint(10000, 99999)}"

                report.title = f"Create_Model_{time.strftime('%Y%m%d%H%M%S')}_{random.randint(100, 999)}"

                task_list.append(
                    [ready_color, image_shape, ready_aisle, cf_src, report.query_path, new_model_name]
                )

            else:
                logger.debug(tip := f"分类不正确: {os.path.basename(video_data)}")
                self.design.show_panel(tip, Wind.KEEPER)

        if len(task_list) == 0:
            logger.debug(tip := f"没有有效任务")
            return self.design.show_panel(tip, Wind.KEEPER)

        await self.design.boot_core_sequence()

        monitor = SourceMonitor()
        monitor_task = asyncio.create_task(monitor.monitor_stable())
        await self.design.multi_load_ripple_vision(monitor)
        await monitor_task
        logger.debug(f"系统负载检测通过: {monitor.usages}")

        # Notes: Analyzer
        if len(task_list) == 1:
            task = [
                looper.run_in_executor(None, alynex.ks.build, *compile_data)
                for compile_data in task_list
            ]
            futures = await asyncio.gather(*task)

        else:
            await random.choice(
                [self.design.boot_process_matrix, self.design.boot_process_sequence]
            )(min(5, max(2, len(task_list))))

            this_level = self.level
            self.level = "ERROR"
            func = partial(alynex.ks.build)
            with ProcessPoolExecutor(self.power, None, Active.active, ("ERROR",)) as exe:
                task = [
                    looper.run_in_executor(exe, func, *compile_data)
                    for compile_data in task_list
                ]
                futures = await asyncio.gather(*task)
            self.level = this_level

        await self.design.model_manifest()

        for future in futures:
            logger.debug(future)
            if isinstance(future, Exception):
                self.design.show_panel(future, Wind.KEEPER)
            else:
                self.design.show_panel(f"Model saved successfully: {Path(future).name}", Wind.DESIGNER)

        Design.console.print()

    # """线迹创造者"""
    async def painting(self, option: "Option", deploy: "Deploy") -> None:
        """
        使用设备截图进行绘制，并在图像上添加网格辅助信息。

        Parameters
        ----------
        option : Option
            选项配置对象，包含运行时选项设置，如模型路径、输出控制等。

        deploy : Deploy
            绘图部署配置对象，包含颜色模式、图像尺寸、裁剪方式以及保存选项等参数。

        Notes
        -----
        - 支持处理多设备截图，自动适配每台设备执行相同的绘制逻辑。
        - 可选择彩色或灰度模式，依据部署参数自动进行图像裁剪、缩放和格式调整。
        - 图像将以网格形式展示，适用于视觉验证或调试用途。
        - 最后通过控制台交互询问用户是否保存处理结果。

        Workflow
        --------
        1. 加载运行选项和部署参数。
        2. 枚举已连接设备，逐一进行截图抓取。
        3. 执行图像转换：灰度/彩色选择、裁剪、省略边缘、缩放等。
        4. 添加标准网格线以增强图像可读性。
        5. 展示图像预览，支持用户交互式选择是否保存。
        """

        async def paint_lines(device: "Device") -> typing.Coroutine | "Image":
            """
            从指定设备获取屏幕截图，并进行图像处理和网格绘制。

            该函数会从设备截取屏幕图像，对其进行可选的裁剪、省略、灰度转换和尺寸缩放，最后在图像上绘制辅助网格线并展示。

            Parameters
            ----------
            device : Device
                表示目标设备的实例，包含唯一标识（如 serial）用于操作 ADB。

            Returns
            -------
            Image.Image
                最终处理完成的 PIL 图像对象，包含裁剪、省略和网格线效果。

            Notes
            -----
            - 本方法在临时目录中执行所有图像处理，处理后图像自动展示。
            - 所有绘图比例和尺寸自动适配图像实际宽高。
            - 所需配置通过全局变量 `deploy` 和 `self.design` 获取。

            Workflow
            --------
            1. 截图采集：
                - 使用 ADB 命令将设备当前屏幕截图保存至设备端并拉取到本地临时目录。
            2. 图像加载与预处理：
                - 加载图像并根据 `deploy.color` 决定是否转为灰度。
                - 应用裁剪（crops）或省略（omits）区域调整图像。
            3. 图像缩放与尺寸调整：
                - 依据 `deploy.shape` 获取目标尺寸。
                - 按 `deploy.scale` 配置对图像进行缩放。
            4. 绘制网格线：
                - 根据图像宽高比自动决定网格行列数。
                - 绘制水平与垂直线，附带百分比文本标注。
            5. 图像展示与清理：
                - 显示最终图像。
                - 删除设备端的临时截图。
            """
            image_folder = "/sdcard/Pictures/Shots"
            image = f"{time.strftime('%Y%m%d%H%M%S')}_{random.randint(100, 999)}_Shot.png"

            await Terminal.cmd_line(
                [self.adb, "-s", device.sn, "wait-for-device"]
            )
            await Terminal.cmd_line(
                [self.adb, "-s", device.sn, "shell", "mkdir", "-p", image_folder]
            )
            await Terminal.cmd_line(
                [self.adb, "-s", device.sn, "shell", "screencap", "-p", f"{image_folder}/{image}"]
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                image_save_path = os.path.join(temp_dir, image)
                await Terminal.cmd_line(
                    [self.adb, "-s", device.sn, "pull", f"{image_folder}/{image}", image_save_path]
                )

                if deploy.color:
                    old_image = toolbox.imread(image_save_path)
                    new_image = VideoFrame(0, 0, old_image)
                else:
                    old_image = toolbox.imread(image_save_path)
                    old_image = toolbox.turn_grey(old_image)
                    new_image = VideoFrame(0, 0, old_image)

                if len(deploy.crops) > 0:
                    for crop in deploy.crops:
                        if sum(crop.values()) > 0:
                            x, y, x_size, y_size = crop["x"], crop["y"], crop["x_size"], crop["y_size"]
                            paint_crop_hook = PaintCropHook((y_size, x_size), (y, x))
                            paint_crop_hook.do(new_image)

                if len(deploy.omits) > 0:
                    for omit in deploy.omits:
                        if sum(omit.values()) > 0:
                            x, y, x_size, y_size = omit["x"], omit["y"], omit["x_size"], omit["y_size"]
                            paint_omit_hook = PaintOmitHook((y_size, x_size), (y, x))
                            paint_omit_hook.do(new_image)

                cv2.imencode(".png", new_image.data)[1].tofile(image_save_path)

                image_file = Image.open(image_save_path)
                image_file = image_file.convert("RGB")

                original_w, original_h = image_file.size
                if deploy.shape:
                    twist_w, twist_h, _ = await Switch.ask_magic_frame(image_file.size, deploy.shape)
                else:
                    twist_w, twist_h = original_w, original_h

                min_scale, max_scale = 0.1, 1.0
                if deploy.scale:
                    image_scale = max_scale if deploy.shape else max(min_scale, min(max_scale, deploy.scale))
                else:
                    image_scale = max_scale if deploy.shape else const.DEFAULT_SCALE

                new_w, new_h = int(twist_w * image_scale), int(twist_h * image_scale)
                logger.debug(
                    tip := f"原始尺寸: {(original_w, original_h)} 调整尺寸: {(new_w, new_h)} 缩放比例: {int(image_scale * 100)} %"
                )
                self.design.show_panel(tip, Wind.DRAWER)

                if new_w == new_h:
                    x_line_num, y_line_num = 10, 10
                elif new_w > new_h:
                    x_line_num, y_line_num = 10, 20
                else:
                    x_line_num, y_line_num = 20, 10

                resized = image_file.resize((new_w, new_h))

                draw = ImageDraw.Draw(resized)
                font = ImageFont.load_default()

                if y_line_num > 0:
                    for i in range(1, y_line_num):
                        x_line = int(new_w * (i * (1 / y_line_num)))
                        text = f"{i * int(100 / y_line_num):02}"
                        bbox = draw.textbbox((0, 0), text, font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        y_text_start = 3
                        draw.line(
                            [(x_line, text_width + 5 + y_text_start), (x_line, new_h)],
                            fill=(0, 255, 255), width=1
                        )
                        draw.text(
                            (x_line - text_height // 2, y_text_start),
                            text, fill=(0, 255, 255), font=font
                        )

                if x_line_num > 0:
                    for i in range(1, x_line_num):
                        y_line = int(new_h * (i * (1 / x_line_num)))
                        text = f"{i * int(100 / x_line_num):02}"
                        bbox = draw.textbbox((0, 0), text, font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        x_text_start = 3
                        draw.line(
                            [(text_width + 5 + x_text_start, y_line), (new_w, y_line)],
                            fill=(255, 182, 193), width=1
                        )
                        draw.text(
                            (x_text_start, y_line - text_height // 2),
                            text, fill=(255, 182, 193), font=font
                        )

                resized.show()

            await Terminal.cmd_line(
                [self.adb, "-s", device.sn, "shell", "rm", f"{image_folder}/{image}"]
            )
            return resized

        async def persistence(device: "Device", image: "Image") -> typing.Coroutine | None:
            pic_path = Path(
                report.query_path
            ) / f"hook_{device.sn}_{random.randint(10000, 99999)}.png"

            await asyncio.to_thread(image.save, pic_path)

            logger.debug(tip := f"保存图片: {pic_path.name}")
            self.design.show_panel(tip, Wind.DRAWER)

        # Notes: Start from here
        await self.design.pixel_bloom()

        manage = Manage(self.adb)
        device_list = await manage.operate_device()

        resized_result = await asyncio.gather(
            *(paint_lines(device) for device in device_list)
        )

        while True:
            action = Prompt.ask(
                f"[bold]保存图片([bold #5FD700]Y[/]/[bold #FF87AF]N[/])?[/]", console=Design.console, default="Y"
            )
            if action.strip().upper() == "Y":
                report = Report(option.total_place)
                report.title = f"Hooks_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

                return await asyncio.gather(
                    *(persistence(device, resize_img) for device, resize_img in zip(device_list, resized_result))
                )

            elif action.strip().upper() == "N":
                break

            else:
                self.design.show_panel(f"没有该选项,请重新输入\n", Wind.KEEPER)

        Design.console.print()

    # """循环节拍器 | 脚本驱动者 | 全域执行者"""
    async def analysis(self, option: "Option", deploy: "Deploy", tp_ver: typing.Any) -> None:
        """
        根据当前执行模式启动视频录制流程或自动化批处理流程，支持速度优先、基础分析、深度模型等多种模式。

        Parameters
        ----------
        option : Option
            系统配置项，包括模型路径与输出位置等选项封装。

        deploy : Deploy
            部署参数对象，包含所有 CLI 配置项，如滑动窗口大小、权重参数、剪辑规则等。

        tp_ver : typing.Any
            依赖的 scrcpy 三方应用版本。

        Notes
        -----
        - 若指定 `--flick` 参数，将启用交互式录制流程（flick_loop），由用户手动控制录制与时间。
        - 若指定 `--carry` 或 `--fully` 参数，则进入自动化批量执行模式（other_loop），按脚本批量录制与分析。
        - 若未设置任何运行模式参数，将跳过该方法的主流程，不进行任务分发。
        - 在初始化过程中会加载分析模型（如 --keras），并确认相关设备状态。
        - 模型加载失败不会中止主流程，但将不执行 Keras 分析路径。
        """

        async def anything_film(device_list: list["Device"], report: "Report") -> list[list]:
            """
            初始化所有设备，计算窗口布局并启动对应的视频录制任务。

            Parameters
            ----------
            device_list : list of Device
                所有待启动录制的设备对象列表。

            report : Report
                报告实例对象，用于配置路径、生成标题和视频存储目录。

            Returns
            -------
            list of list
                每个任务的参数集合，包含录制临时路径、传输对象、
                视频与数据存储路径、报告标题等信息，供后续分析使用。

            Notes
            -----
            - 该函数还负责根据屏幕尺寸计算窗口摆放布局，避免设备画面重叠。
            - 每个设备会以异步方式开启录制任务，并按窗口位置进行排列。
            - 返回的列表中，每一项对应一个设备的完整录制上下文。
            """
            monitor_task = asyncio.create_task(monitor.monitor_stable())
            await self.design.multi_load_ripple_vision(monitor)
            await monitor_task

            for device in device_list:
                logger.debug(f"Wait Device Online -> {device.tag} {device.sn}")
            await asyncio.gather(
                *(device.device_online() for device in device_list)
            )

            media_screen_w, media_screen_h = ScreenMonitor.screen_size()
            logger.debug(f"Media Screen W={media_screen_w} H={media_screen_h}")

            todo_list = []
            format_folder = time.strftime("%Y%m%d%H%M%S")

            margin_x, margin_y = 50, 75
            window_x, window_y = 50, 75
            max_y_height = 0
            for device in device_list:
                device_x, device_y = device.display[device.id]
                device_x, device_y = int(device_x * 0.25), int(device_y * 0.25)

                # 检查是否需要换行
                if window_x + device_x + margin_x > media_screen_w:
                    # 重置当前行的开始位置
                    window_x = 50
                    if (new_y_height := window_y + max_y_height) + device_y > media_screen_h:
                        # 如果新行加设备高度超出屏幕底部，则只增加一个 margin_y
                        window_y += margin_y
                    else:
                        # 否则按计划设置新行的起始位置
                        window_y = new_y_height
                    # 重置当前行的最大高度
                    max_y_height = 0
                # 更新当前行的最大高度
                max_y_height = max(max_y_height, device_y)
                # 位置确认
                location = window_x, window_y, device_x, device_y
                # 移动到下一个设备的起始位置
                window_x += device_x + margin_x
                # 延时投屏
                await asyncio.sleep(0.5)

                report.query = os.path.join(format_folder, device.sn)

                video_temp, transports = await record.ask_start_record(
                    device, report.video_path, location=location
                )
                todo_list.append(
                    [video_temp, transports, report.total_path, report.title, report.query_path,
                     report.query, report.frame_path, report.extra_path, report.proto_path]
                )

            return todo_list

        async def anything_over(device_list: list["Device"], task_list: list[list]) -> None:
            """
            完成录制任务后的清理流程，关闭录制并剔除无效任务。

            Parameters
            ----------
            device_list : list of Device
                当前参与录制的设备对象列表。

            task_list : list of list
                每个设备对应的录制任务信息，包括录制文件路径和传输对象等。

            Notes
            -----
            - 每台设备调用关闭录制方法 `ask_close_record`，回收资源并返回状态。
            - 若某个任务标记为“录制失败”，该任务将从 `task_list` 中移除。
            - 所有处理结果将以调试信息的形式显示在控制面板中。
            """
            effective_list = await asyncio.gather(
                *(record.ask_close_record(device, transports)
                  for device, (_, transports, *_) in zip(device_list, task_list))
            )

            for (idx, effective), (video_temp, *_) in zip(enumerate(effective_list), task_list):
                logger.debug(f"{Path(video_temp).name} status={effective}")
                if isinstance(effective, Exception):
                    try:
                        task = task_list.pop(idx)
                        logger.debug(f"移除录制失败项: {Path(task[0]).name}")
                    except IndexError as e:
                        logger.debug(e)

        async def anything_well(task_list: list[list], report: "Report") -> None:
            """
            根据分析模式对录制任务进行处理，支持快速模式、基础模式和模型模式。

            Parameters
            ----------
            task_list : list of list
                每个任务包含录制文件路径、传输对象及报告路径等相关信息。

            report : Report
                当前分析过程关联的报告对象。

            Notes
            -----
            - 若任务列表为空，则终止处理流程并提示。
            - 分析模式依据实例的 `speed`、`basic` 或 `keras` 属性自动选择：
                - `speed` 模式：快速执行分析，不加载时间戳。
                - `basic` 模式：加载基础时间信息，支持可解释分析。
                - `keras` 模式：使用深度模型分析，识别视频内容区域。
            """
            if len(task_list) == 0:
                logger.debug(tip := f"没有有效任务")
                return self.design.show_panel(tip, Wind.KEEPER)

            # Notes: Analyzer
            if self.speed:
                await self.als_speed(
                    deploy, clipix, report, task_list
                )
            elif self.basic or self.keras:
                await self.als_basic_or_keras(
                    deploy, clipix, report, task_list, option=option, alynex=alynex
                )
            else:
                logger.debug(f"**<* 影像捕手 *>**")
                self.design.show_panel(Wind.MOVIE_TEXT, Wind.MOVIE)

        async def load_carry(carry: str) -> dict:
            """
            加载并解析 carry 指令字符串，提取并返回对应任务的执行命令字典。

            Parameters
            ----------
            carry : str
                用逗号、分号或空格分隔的 carry 指令字符串。第一个元素应为路径，其余为任务键。

            Returns
            -------
            dict
                包含待执行命令的键值对字典，key 为任务标识，value 为执行参数。

            Raises
            ------
            FramixError
                - 参数不足或格式错误；
                - 文件不存在；
                - 文件结构非法或 JSON 解析失败；
                - 指定键不存在。
            """
            if len(parts := re.split(r",|;|!|\s", carry)) >= 2:
                loc_file, *key_list = parts

                exec_dict = await load_fully(loc_file)

                try:
                    return {key: value for key in key_list if (value := exec_dict.get(key, None))}
                except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                    raise FramixError(e)

            raise FramixError(f"参数错误: {carry}")

        async def load_fully(fully: str) -> dict:
            """
            异步加载并解析完整的 JSON 命令文件，提取格式化后的执行指令集合。

            Parameters
            ----------
            fully : str
                JSON 脚本文件路径，用于批量加载多组命令数据。

            Returns
            -------
            dict
                执行指令的字典，每个 key 对应一组命令，每组命令包含 parser、header、action 等字段。

            Raises
            ------
            FramixError
                - 当文件不存在、格式不符合要求，或无法解析为合法 JSON 时抛出。
            """
            fully = await Craft.revise_path(fully)

            try:
                async with aiofiles.open(fully, "r", encoding=const.CHARSET) as f:
                    file_list = json.loads(await f.read())["command"]
                    exec_dict = {
                        file_key: {
                            **({"parser": cmds["parser"]} if cmds.get("parser") else {}),
                            **({"header": cmds["header"]} if cmds.get("header") else {}),
                            **({"change": cmds["change"]} if cmds.get("change") else {}),
                            **({"looper": cmds["looper"]} if cmds.get("looper") else {}),
                            **({"prefix": [c for c in cmds.get("prefix", []) if c["cmds"]]} if any(
                                c["cmds"] for c in cmds.get("prefix", [])) else {}),
                            **({"action": [c for c in cmds.get("action", []) if c["cmds"]]} if any(
                                c["cmds"] for c in cmds.get("action", [])) else {}),
                            **({"suffix": [c for c in cmds.get("suffix", []) if c["cmds"]]} if any(
                                c["cmds"] for c in cmds.get("suffix", [])) else {}),
                        } for file_dict in file_list for file_key, cmds in file_dict.items()
                        if any(c["cmds"] for c in cmds.get("action", []))
                    }
            except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                raise FramixError(e)

            return exec_dict

        async def call_commands(
                bean: typing.Any, live_devices: dict,
                exec_func: str, exec_vals: list, exec_args: list, exec_kwds: dict
        ) -> typing.Any:
            """
            执行指定对象的指令函数（异步或同步），并捕获运行结果与异常。

            Parameters
            ----------
            bean : typing.Any
                可执行指令的对象实例（如 Device 或 Player 实例）。

            live_devices : dict
                当前存活的设备字典，key 为设备编号（sn），value 为设备实例。

            exec_func : str
                待执行的函数名称字符串。

            exec_vals : list
                函数调用的主参数（位于 args 前的显式参数值）。

            exec_args : list
                函数调用的扩展位置参数（*args）。

            exec_kwds : dict
                函数调用的关键字参数（**kwargs）。

            Returns
            -------
            typing.Any
                如果函数有返回值则返回，若无返回值或异常中断则返回 None。

            Raises
            ------
            FramixError
                - 当函数执行异常或找不到目标函数时抛出 FramixError 异常。
            """
            if not (callable(function := getattr(bean, exec_func, None))):
                logger.debug(tip := f"No callable {exec_func}")
                return self.design.show_panel(tip, Wind.KEEPER)

            sn = getattr(bean, "sn", bean.__class__.__name__)
            try:
                logger.debug(tip := f"{sn} {function.__name__} {exec_vals}")
                self.design.show_panel(tip, Wind.EXPLORER)

                if inspect.iscoroutinefunction(function):
                    if call_result := await function(*exec_vals, *exec_args, **exec_kwds):
                        logger.debug(tip := f"Returns: {call_result}")
                        return self.design.show_panel(tip, Wind.EXPLORER)

            except asyncio.CancelledError:
                live_devices.pop(sn)
                logger.debug(tip := f"{sn} Call Commands Exit")
                self.design.show_panel(tip, Wind.EXPLORER)
            except Exception as e:
                return FramixError(e)

        async def pack_commands(resolve_list: list) -> list:
            """
            将脚本中解析得到的命令描述列表，转换为标准格式的指令元组序列。

            Parameters
            ----------
            resolve_list : list
                从脚本中解析得到的命令配置，每个元素是包含 `cmds`, `vals`, `args`, `kwds` 的 dict。

            Returns
            -------
            list
                一个列表，包含已打包好的命令执行单元，每个单元是 (func, vals, args, kwds) 的元组序列。
                每个设备任务为一组命令元组的列表，最终形成二维结构 list[list[tuple]]。

            Notes
            -----
            - 此方法确保输入不为空字符串，并为缺失参数补全默认空值。
            - 所有输入会被校验并标准化为 list 或 dict。
            """
            exec_pairs_list = []

            for resolve in resolve_list:
                device_cmds_list = resolve.get("cmds", [])
                if all(isinstance(device_cmds, str) and device_cmds != "" for device_cmds in device_cmds_list):
                    # 因为要并发，所以不能去除重复命令
                    # device_cmds_list = list(dict.fromkeys(device_cmds_list))

                    # 解析 vals 参数
                    device_vals_list = resolve.get("vals", [])
                    device_vals_list = [
                        d_vals if isinstance(d_vals, list) else ([] if d_vals == "" else [d_vals])
                        for d_vals in device_vals_list
                    ]
                    device_vals_list += [[]] * (len(device_cmds_list) - len(device_vals_list))

                    # 解析 args 参数
                    device_args_list = resolve.get("args", [])
                    device_args_list = [
                        d_args if isinstance(d_args, list) else ([] if d_args == "" else [d_args])
                        for d_args in device_args_list
                    ]
                    device_args_list += [[]] * (len(device_cmds_list) - len(device_args_list))

                    # 解析 kwds 参数
                    device_kwds_list = resolve.get("kwds", [])
                    device_kwds_list = [
                        d_kwds if isinstance(d_kwds, dict) else ({} if d_kwds == "" else {"None": d_kwds})
                        for d_kwds in device_kwds_list
                    ]
                    device_kwds_list += [{}] * (len(device_cmds_list) - len(device_kwds_list))

                    exec_pairs_list.append(
                        list(zip(device_cmds_list, device_vals_list, device_args_list, device_kwds_list))
                    )

            return exec_pairs_list

        async def exec_commands(device_list: list["Device"], exec_pairs_list: list, *args) -> None:
            """
            在多个设备上并发执行一组命令任务，支持通配符参数替换。

            Parameters
            ----------
            device_list : list of Device
                活跃的设备列表，每个设备将尝试执行指定的命令序列。

            exec_pairs_list : list
                每组为多个命令元组的列表，形式为 (exec_func, exec_vals, exec_args, exec_kwds)。

            *args : any
                用于替换命令参数中 "*" 的外部输入数据，逐一映射到对应位置。

            Notes
            -----
            - 如果 exec_func 为 "audio_player"，将使用 player 对象单独执行。
            - 其他命令会在所有设备上并发执行，失败设备将被移除。
            - 命令支持通配符 "*" 替换，通过传入 *args 实现参数动态注入。
            - 每一轮命令执行结束后清除任务，异常将记录到日志中但不中断主流程。
            """

            # 替换列表中所有 `*` 为给定参数列表中的值，支持嵌入字符串中的 `*` 多次替换。
            replace_star: typing.Callable[[list], list] = lambda x, y=iter(args): [
                "".join(next(y, "*") if c == "*" else c for c in i)
                if isinstance(i, str) else (next(y, "*") if i == "*" else i) for i in x
            ]

            live_devices = {device.sn: device for device in device_list}.copy()

            exec_tasks: dict[str, "asyncio.Task"] = {}
            stop_tasks: list["asyncio.Task"] = []

            for device in device_list:
                stop_tasks.append(
                    asyncio.create_task(
                        record.check_event(device, exec_tasks), name="stop"
                    )
                )

            for exec_pairs in exec_pairs_list:
                if len(live_devices) == 0:
                    return Design.notes(f"[bold #F0FFF0 on #000000]All tasks canceled ...")

                for exec_func, exec_vals, exec_args, exec_kwds in exec_pairs:
                    exec_vals = replace_star(exec_vals)

                    if exec_func == "audio_player":
                        await call_commands(
                            player, live_devices, exec_func, exec_vals, exec_args, exec_kwds
                        )
                    else:
                        for device in live_devices.values():
                            exec_tasks[device.sn] = asyncio.create_task(
                                call_commands(
                                    device, live_devices, exec_func, exec_vals, exec_args, exec_kwds
                                )
                            )

                try:
                    exec_status_list = await asyncio.gather(
                        *exec_tasks.values(), return_exceptions=True
                    )
                except asyncio.CancelledError:
                    return Design.notes(f"[bold #F0FFF0 on #000000]All tasks canceled ...")
                finally:
                    exec_tasks.clear()

                for status in exec_status_list:
                    if isinstance(status, Exception):
                        logger.debug(status)
                        self.design.show_panel(status, Wind.KEEPER)

            for stop in stop_tasks:
                stop.cancel()

        async def flick_loop() -> typing.Coroutine | None:
            """
            启动交互式的录制与分析主循环，用于手动控制录制流程与配置参数。

            Returns
            -------
            Coroutine or None
                若执行成功则返回协程；若中断或退出则返回 None。

            Notes
            -----
            - 支持设置定时器、修改标题、编辑配置文件、切换设备等命令。
            - 用户输入控制整个录制流程，支持多轮录制与分析循环。
            - 根据 self.whist 控制定时范围；用户输入 header/create/deploy 等命令进行动态控制。
            - 每次录制完成后自动检测异常并重置设备列表。
            - 若使用 digest 指令，可在所有轮次录制完成后统一进行视频分析。
            """
            device_list = await manage_.operate_device()

            report = Report(option.total_place)
            report.title = f"{input_title_}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

            lower_bound, upper_bound = (8, 300) if self.whist else (5, 300)
            amount = lower_bound

            choices = list(titles_.keys())

            await self.design.wave_converge_animation()
            Design.tips_document()

            while True:
                try:
                    await manage_.display_device(ctrl_)
                    run_tip = f"<<<按 Enter 开始 [bold #D7FF5F]{amount}[/] 秒>>>"

                    if action := Prompt.ask(f"[bold #5FD7FF]{run_tip}", console=Design.console):
                        if (select := action.strip().lower()) == "device":
                            device_list = await manage_.another_device()
                            continue

                        elif select == "cancel":
                            Design.console.print()
                            Design.exit()
                            sys.exit(Design.closure())

                        elif "header" in select:
                            if match := re.search(r"(?<=header\s).*", select):
                                if hd := match.group().strip():
                                    src_hd, a, b = f"{input_title_}_{time.strftime('%Y%m%d_%H%M%S')}", 10000, 99999
                                    Design.notes(f"{const.SUC}New title set successfully")
                                    report.title = f"{src_hd}_{hd}" if hd else f"{src_hd}_{random.randint(a, b)}"
                                    continue
                            raise FramixError(f"命名方式应为 header .*")

                        elif select == "digest":
                            if any((self.speed, self.basic, self.keras)):
                                selected = next(
                                    (attr for attr in choices if getattr(self, attr, False)), None
                                )
                                raise FramixError(f"当前处于 {selected} 分析模式中，请使用 create 指令生成报告")

                            choices += ["88"]
                            while mode := Prompt.ask(
                                    f"[bold #ADFF2F]Choose analysis mode [bold #FFA07A][{choices[-1]}]返回",
                                    console=Design.console, choices=choices, default=choices[-1]
                            ):
                                if mode == choices[-1]:
                                    choices.pop()
                                    break

                                choices.pop()
                                for key in choices:
                                    setattr(self, key, key == mode)
                                return await self.video_data_task(
                                    [Path(report.total_path).parent], option, deploy
                                )

                            continue

                        elif select == "create":
                            if any((self.speed, self.basic, self.keras)):
                                return await self.combine(report)
                            raise FramixError(f"当前处于 影像捕手 模式，请使用 digest 指令进行分析并生成报告")

                        elif select == "deploy":
                            Design.notes(f"{const.WRN}请完全退出编辑器再继续操作")
                            deploy.dump_deploy(self.initial_deploy)
                            if sys.platform == "win32":
                                first = ["notepad++"] if shutil.which("notepad++") else ["Notepad"]
                            else:
                                first = ["open", "-W", "-a", "TextEdit"]
                            await Terminal.cmd_line(first + [self.initial_deploy])
                            deploy.load_deploy(self.initial_deploy)
                            deploy.view_deploy()
                            await self.design.neural_sync_loading()
                            continue

                        elif select.isdigit():
                            timer_value = int(select)
                            if timer_value > upper_bound or timer_value < lower_bound:
                                bound_tips = f"{lower_bound} <= [bold #FFD7AF]Time[/] <= {upper_bound}"
                                Design.notes(f"[bold #FFFF87]{bound_tips}")
                            amount = max(lower_bound, min(upper_bound, timer_value))

                        else:
                            raise FramixError(f"未知命令 {select}")

                except FramixError as e:
                    Design.notes(f"{const.WRN}{e}")
                    Design.simulation_progress("Try again")
                    Design.tips_document()

                else:
                    record.record_events = {
                        device.sn: {
                            **{i: asyncio.Event() for i in ["head", "done", "stop", "fail"]},
                            "remain": 0, "notify": f"等待同步"
                        } for device in device_list
                    }
                    record_ui_task = asyncio.create_task(
                        self.design.display_record_ui(record.record_events, amount)
                    )

                    task_list = await anything_film(device_list, report)  # 开始录制

                    await asyncio.gather(
                        *(record.check_timer(device, amount) for device in device_list)  # 启动计时
                    )
                    await anything_over(device_list, task_list)  # 结束录制
                    await record_ui_task

                    await anything_well(task_list, report)  # 任务结束

                    if await record.flunk_event():
                        device_list = await manage_.operate_device()

                finally:
                    await record.clean_event()

        async def other_loop() -> typing.Coroutine | None:
            """
            加载并执行 carry 或 fully 脚本文件中的批处理任务指令，并完成自动化录制与分析流程。

            Returns
            -------
            Coroutine or None
                异步执行任务流程的协程对象，若执行被中断或无内容返回 None。

            Notes
            -----
            - carry 表示指定任务子集；fully 表示完整的脚本任务集，二者互斥使用。
            - 自动激活所有连接设备的自动化引擎（automator）。
            - 每个任务组支持执行 parser 参数解析、header 命名、change 数据扩展、looper 循环次数。
            - 分为 prefix（前置指令）、action（主要操作）、suffix（后置指令）三个阶段。
            - 所有指令集会被自动解析为设备可执行命令并分发调度。
            - 若开启 `--shine` 模式，所有任务将统一收集后集中分析；否则逐轮分析。
            - 在执行所有脚本任务后，如设置了 speed/basic/keras 分析模式，会合并生成最终报告。
            """
            load_script_data = await asyncio.gather(
                *(load_carry(c) for c in self.carry) if self.carry else (load_fully(f) for f in self.fully)
            )

            if not (script_storage := [script_data_ for script_data_ in load_script_data]):
                raise FramixError(f"Script content is empty")

            device_list = await manage_.operate_device()

            for device in device_list:
                logger.debug(tip := f"{device.sn} Automator Activation")
                self.design.show_panel(tip, Wind.EXPLORER)

            try:
                await asyncio.gather(
                    *(device.automator_activation() for device in device_list)
                )
            except Exception as e:
                raise FramixError(e)

            await self.design.batch_runner_task_grid()

            await manage_.display_device(ctrl_)

            for script_dict in script_storage:
                report = Report(option.total_place)
                for script_key, script_value in script_dict.items():
                    logger.debug(tip := f"Batch Exec: {script_key}")
                    self.design.show_panel(tip, Wind.EXPLORER)

                    # 根据 script_value_ 中的 parser 参数更新 deploy 配置
                    if (parser := script_value.get("parser", {})) and type(parser) is dict:
                        for deploy_key, deploy_value in deploy.deploys.items():
                            logger.debug(f"Current Key {deploy_key}")
                            for d_key, d_value in deploy_value.items():
                                # 以命令行参数为第一优先级
                                if any(line_.lower().startswith(f"--{d_key}") for line_ in self.wires):
                                    logger.debug(f"    Line First <{d_key}> = {getattr(deploy, d_key)}")
                                    continue
                                setattr(deploy, d_key, parser.get(deploy_key, {}).get(d_key, {}))
                                logger.debug(f"    Parser Set <{d_key}>  {d_value} -> {getattr(deploy, d_key)}")

                    # 处理 script_value_ 中的 header 参数
                    header = header if type(
                        header := script_value.get("header", [])
                    ) is list else ([header] if type(header) is str else [time.strftime("%Y%m%d%H%M%S")])

                    # 处理 script_value_ 中的 change 参数
                    if change := script_value.get("change", []):
                        change = change if type(change) is list else (
                            [change] if type(change) is str else [str(change)])

                    # 处理 script_value_ 中的 looper 参数
                    try:
                        looper = int(looper) if (looper := script_value.get("looper", None)) else 1
                    except ValueError as e:
                        logger.debug(tip := f"重置循环次数 {(looper := 1)} {e}")
                        self.design.show_panel(tip, Wind.EXPLORER)

                    # 处理 script_value 中的 prefix 参数
                    if prefix_list := script_value.get("prefix", []):
                        prefix_list = await pack_commands(prefix_list)
                    # 处理 script_value 中的 action 参数
                    if action_list := script_value.get("action", []):
                        action_list = await pack_commands(action_list)
                    # 处理 script_value 中的 suffix 参数
                    if suffix_list := script_value.get("suffix", []):
                        suffix_list = await pack_commands(suffix_list)

                    # 遍历 header 并执行任务
                    for hd in header:
                        report.title = f"{input_title_}_{script_key}_{hd}"
                        extend_task_list = []

                        for _ in range(looper):
                            # prefix 前置任务
                            if prefix_list:
                                await exec_commands(device_list, prefix_list)

                            # start record 开始录屏
                            task_list = await anything_film(device_list, report)

                            # action 主要任务
                            if action_list:
                                change_list = [hd + c for c in change] if change else [hd]
                                await exec_commands(device_list, action_list, *change_list)

                            # close record 结束录屏
                            await anything_over(device_list, task_list)

                            # 检查事件并更新设备列表，清除所有事件
                            if await record.flunk_event():
                                device_list = await manage_.operate_device()
                            await record.clean_event()

                            # suffix 提交后置任务
                            suffix_task_list = []
                            if suffix_list:
                                suffix_task_list.append(
                                    asyncio.create_task(
                                        exec_commands(device_list, suffix_list), name="suffix"
                                    )
                                )

                            # 根据参数判断是否分析视频以及使用哪种方式分析
                            if self.shine:
                                extend_task_list.extend(task_list)
                            else:
                                await anything_well(task_list, report)

                            # 等待后置任务完成
                            await asyncio.gather(*suffix_task_list)

                        # 分析视频集合
                        if task_list := (extend_task_list if self.shine else []):
                            await anything_well(task_list, report)

                # 如果需要，结合多种模式生成最终报告
                if any((self.speed, self.basic, self.keras)):
                    await self.combine(report)

            Design.console.print()

        # Notes: Start from here
        manage_ = Manage(self.adb)

        if any((self.speed, self.basic, self.keras)):
            clipix = Clipix(self.fmp, self.fpb)

            matrix = option.model_place if self.keras else None
            alynex = Alynex(matrix, option, deploy, self.design)
            try:
                await alynex.ask_model_load()
            except FramixError as err_:
                logger.debug(err_)
                self.design.show_panel(err_, Wind.KEEPER)

        titles_ = {"speed": "Speed", "basic": "Basic", "keras": "Keras"}
        input_title_ = next((title_ for key_, title_ in titles_.items() if getattr(self, key_)), "Video")

        ctrl_ = f"静默守护模式" if self.whist else f"{('独立' if self.alone else '全局')}控制模式"

        record = Record(
            tp_ver, alone=self.alone, whist=self.whist, frate=deploy.frate
        )
        player = Player()
        monitor = SourceMonitor()

        return await flick_loop() if self.flick else await other_loop()


class Clipix(object):
    """
    Clipix Engine

    该类封装 FFmpeg 与 FFprobe 的路径配置，用于执行视频剪辑、编码转换、
    元信息分析等基础任务，作为 Framix 视频处理流程的底层工具引擎。

    Attributes
    ----------
    fmp : str
        FFmpeg 的可执行文件路径，用于处理视频转码与剪辑等操作。

    fpb : str
        FFprobe 的可执行文件路径，用于提取视频的元数据信息（如帧率、时长、分辨率等）。
    """

    def __init__(self, fmp: str, fpb: str):
        self.fmp = fmp  # 表示 ffmpeg 的路径
        self.fpb = fpb  # 表示 ffprobe 的路径

    async def vision_content(
            self,
            video_temp: str,
            start: typing.Optional[str],
            close: typing.Optional[str],
            limit: typing.Optional[str],
    ) -> tuple[str, str, float, tuple, dict]:
        """
        异步提取视频的关键内容信息，并计算可视时间点。

        该函数分析指定的视频文件，提取实际与平均帧率、视频总时长、原始分辨率等元数据，
        并根据起止时间与限制时间推导出处理区域的时间范围。

        Parameters
        ----------
        video_temp : str
            视频文件的路径。

        start : Optional[str]
            起始时间点，格式为 HH:MM:SS。如果为 None，则从视频开头开始处理。

        close : Optional[str]
            结束时间点，格式为 HH:MM:SS。如果为 None，则处理到视频结尾。

        limit : Optional[str]
            限制处理的最大时长，格式为 HH:MM:SS。如果为 None，则不限制时长。

        Returns
        -------
        tuple
            包含以下五项的数据元组：

            - rlt : str
                实际帧率字符串（例如 "30/1"）。
            - avg : str
                平均帧率字符串（例如 "29/1"）。
            - duration : float
                视频总时长（单位为秒）。
            - original : tuple
                视频的原始分辨率及其他元信息（如宽高）。
            - vision_point : dict
                包含处理区域的起始、结束、时长限制点，格式化为字符串（如 "00:01:23"）。

        Notes
        -----
        - 此函数主要用于视频处理前的预分析，用于指导后续的剪辑与分析流程。
        - 使用 `Parser.parse_mills` 将时间字符串转换为毫秒数，用于精准时间计算。
        - 所有返回时间点均为格式化字符串，便于在日志与报告中呈现。
        - 若输入时间超出视频实际时长，`Switch.ask_magic_point` 将自动进行容错处理。

        Workflow
        --------
        1. 调用 `Switch.ask_video_stream` 获取视频帧率、时长、原始分辨率等信息。
        2. 将传入的 start / close / limit 转换为毫秒数。
        3. 使用 `Switch.ask_magic_point` 对齐计算真实的处理时间点范围。
        4. 格式化时间点为字符串表示，并打包成 vision_point 字典。
        5. 返回分析结果。
        """
        video_streams = await Switch.ask_video_stream(self.fpb, video_temp)

        rlt = video_streams.get("rlt_frame_rate", "0/0")
        avg = video_streams.get("avg_frame_rate", "0/0")
        duration = video_streams.get("duration", 0.0)
        original = video_streams.get("original", (0, 0))

        vision_start, vision_close, vision_limit = await Switch.ask_magic_point(
            Parser.parse_mills(start),
            Parser.parse_mills(close),
            Parser.parse_mills(limit),
            duration
        )
        vision_start: str = Parser.parse_times(vision_start)
        vision_close: str = Parser.parse_times(vision_close)
        vision_limit: str = Parser.parse_times(vision_limit)

        vision_point = {"start": vision_start, "close": vision_close, "limit": vision_limit}

        return rlt, avg, duration, original, vision_point

    async def vision_balance(
            self, duration: float, standard: float, src: str, frate: float
    ) -> tuple[str, str]:
        """
        异步调整视频时长以匹配指定的标准时长，通过裁剪视频的起始和结束时间。

        此函数计算原视频与标准时长的差值，基于这一差值调整视频的开始和结束时间点，以生成新的视频文件，保证其总时长接近标准时长。
        适用于需要统一视频播放长度的场景，例如批量训练数据处理、对齐播放时长或模型输入长度统一化等任务。

        Parameters
        ----------
        duration : float
            原视频的总时长（秒）。

        standard : float
            目标视频的标准时长（秒）。

        src : str
            原视频文件的路径。

        frate : float
            目标视频的帧率。

        Returns
        -------
        tuple[str, str]
            包含两个元素：
            - video_dst：调整后生成的新视频文件路径。
            - video_blc：对视频时长裁剪操作的描述字符串。

        Notes
        -----
        - 如果原始视频小于标准时长，将无法进行裁剪，请提前判断。
        - 输出文件将保存在原路径目录中，文件名中包含帧率和随机编号以避免覆盖。
        - 使用 `Switch.ask_video_tailor` 执行实际的视频裁剪操作。

        Workflow
        --------
        1. 计算应从何时开始裁剪以满足目标时长；
        2. 构建裁剪时间段（start ~ limit）；
        3. 构造输出文件路径及裁剪描述；
        4. 调用视频裁剪方法生成新视频；
        5. 返回新视频路径和裁剪描述信息。
        """
        start_time_point = (limit_time_point := duration) - standard
        start_delta = datetime.timedelta(seconds=start_time_point)
        limit_delta = datetime.timedelta(seconds=limit_time_point)

        video_dst = os.path.join(
            os.path.dirname(src), f"tailor_fps{frate}_{random.randint(100, 999)}.mp4"
        )

        target = f"[{start_delta.total_seconds():.6f} - {limit_delta.total_seconds():.6f}]"
        video_blc = f"{os.path.basename(src)} [{duration:.6f}] {target} -> {os.path.basename(video_dst)}"

        await Switch.ask_video_tailor(
            self.fmp, src, video_dst, start=str(start_delta), limit=str(limit_delta)
        )

        return video_dst, video_blc

    @staticmethod
    async def vision_improve(
            deploy: "Deploy", original: tuple, filters: list
    ) -> list:
        """
        异步方法，用于改进视频的视觉效果，通过调整尺寸并组合过滤器列表。

        根据提供的目标尺寸（shape）或缩放因子（scale），自动计算输出分辨率，并将其添加到过滤器列表中。
        最终返回一个完整的过滤器命令集合，用于后续视频处理流程。

        Parameters
        ----------
        deploy : Deploy
            视频处理配置对象，包含尺寸调整（shape）或缩放比例（scale）等参数。
        original : tuple
            原始视频的分辨率，格式为 (width, height)。
        filters : list
            初始过滤器命令列表，如 ["eq=contrast=1.2", "unsharp"]。

        Returns
        -------
        list
            组合后的完整过滤器命令列表，包含尺寸调整指令。

        Raises
        ------
        ValueError
            如果传入的原始尺寸不是元组类型，或过滤器列表不是列表类型。

        Notes
        -----
        - 若 `shape` 参数存在，则优先使用该参数进行尺寸转换；
        - 若未指定 `shape`，则使用 `scale` 缩放比例对原始尺寸进行调整；
        - 所有输出分辨率将自动转换为偶数，以满足编码器要求。

        Workflow
        --------
        1. 判断是否指定 `shape`，调用 `ask_magic_frame` 获取目标尺寸；
        2. 调整为偶数尺寸并生成 scale 过滤器；
        3. 若未指定 `shape`，则回退为缩放比例处理；
        4. 返回合并后的过滤器列表。
        """
        if deploy.shape:
            w, h, ratio = await Switch.ask_magic_frame(original, deploy.shape)
            w, h = w - 1 if w % 2 != 0 else w, h - 1 if h % 2 != 0 else h
            video_filter_list = filters + [f"scale={w}:{h}"]
        else:
            deploy.scale = deploy.scale or const.DEFAULT_SCALE
            video_filter_list = filters + [f"scale=iw*{deploy.scale}:ih*{deploy.scale}"]

        return video_filter_list

    async def pixels(
            self,
            function: "typing.Callable",
            video_filter: list, src: str, dst: str, **kwargs
    ) -> tuple[str]:
        """
        执行视频过滤处理函数，应用指定的过滤器列表，将源视频转换为目标视频。

        本方法封装了视频处理调用逻辑，通过提供的函数与过滤器配置，实现视频的转码、调整或增强等处理操作。

        Parameters
        ----------
        function : typing.Callable
            视频处理函数，必须接受过滤器、源路径、目标路径等参数，并异步执行。

        video_filter : list
            字符串形式的过滤器列表，例如 ["scale=1280:720", "eq=contrast=1.2"]。

        src : str
            源视频的完整路径。

        dst : str
            目标视频的输出路径。

        **kwargs : dict
            传递给处理函数的额外关键字参数，如帧率、编码选项等。

        Returns
        -------
        tuple[str]
            包含处理结果信息的元组，例如执行日志、输出路径、状态描述等。

        Notes
        -----
        - 该方法本身不包含视频处理逻辑，仅作为封装调用器；
        - `function` 必须为异步函数，且支持过滤器和路径参数；
        - 返回结果依赖于 `function` 的实现，通常为处理状态说明。

        Workflow
        --------
        1. 接收处理函数和参数；
        2. 异步调用目标处理函数，传入 ffmpeg 路径和过滤器；
        3. 返回其执行结果。
        """
        return await function(self.fmp, video_filter, src, dst, **kwargs)


class Alynex(object):
    """
    Alynex Engine

    该类作为模型驱动的分析器核心，负责组织模型结构、参数配置和视频解析流程。
    在运行时通过注入 Option、Deploy 和 Design 等上下文配置，实现深度学习模型
    在视频任务中的推理、标签处理与调度。

    Attributes
    ----------
    matrix : Optional[str]
        模型文件路径，通常为 `.h5` 或 `.keras` 格式，用于 Keras 模型加载。

    option : Option
        程序运行选项，包含模型选择、运行模式、输出开关等控制参数。

    deploy : Deploy
        部署配置对象，包含视频路径、帧率、分辨率、帧提取策略等处理参数。

    design : Design
        标注设计对象，封装标签结构、类名映射、颜色标记等语义设计信息。

    __ks : Optional[KerasStruct]
        内部模型结构缓存对象，封装加载后的模型与结构描述。
    """

    __ks: typing.Optional["KerasStruct"] = KerasStruct()

    def __init__(
            self,
            matrix: typing.Optional[str],
            option: "Option",
            deploy: "Deploy",
            design: "Design"
    ):

        self.matrix = matrix  # 模型文件路径
        self.option = option  # 程序运行选项
        self.deploy = deploy  # 部署配置对象
        self.design = design  # 标注设计对象

    @property
    def ks(self) -> typing.Optional["KerasStruct"]:
        """
        模型结构访问属性。

        该属性用于获取或设置 Alynex 内部使用的 Keras 模型结构对象，
        通常在模型加载或替换过程中使用。

        Returns
        -------
        Optional[KerasStruct]
            当前的模型结构对象，可能为 None。
        """

        return self.__ks

    @ks.setter
    def ks(self, value: typing.Optional["KerasStruct"]) -> None:
        """
        设置模型结构对象。

        用于将外部加载的 KerasStruct 模型注入到 Alynex 内部缓存，
        以供后续分析或推理调用使用。

        Parameters
        ----------
        value : Optional[KerasStruct]
            要设置的模型结构实例，可以为 None 表示清除缓存。
        """

        self.__ks = value

    async def ask_model_load(self) -> None:
        """
        异步加载模型到 KerasStruct 实例中。

        该方法用于在分析任务执行前加载指定路径下的 Keras 模型，并根据部署参数验证模型输入通道的正确性。
        若加载失败或验证不通过，将抛出异常并清除模型状态。
        模型状态通过 `self.ks.model` 保持。
        """
        try:
            if mp := self.matrix:
                assert os.path.isdir(self.option.model_place), f"The model must be a directory {mp}"
                assert self.ks, f"Must be loaded first model"

                assert os.path.isdir(
                    final_model := os.path.join(
                        mp,
                        self.option.color_model if self.deploy.color else self.option.faint_model
                    )
                ), f"No configuration model file {final_model.format()}"
                self.ks.load_model(final_model.format())

                channel = self.ks.model.input_shape[-1]
                if self.deploy.color:
                    assert channel == 3, f"彩色模式需要匹配彩色模型 Model color channel={channel}"
                else:
                    assert channel == 1, f"灰度模式需要匹配灰度模型 Model color channel={channel}"
        except (OSError, TypeError, ValueError, AssertionError, AttributeError) as e:
            self.ks.model = None
            raise FramixError(e)

    async def ask_video_load(self, vision: str, src_size: tuple) -> "VideoObject":
        """
        异步加载视频帧并返回 VideoObject 实例。

        该方法用于从指定路径读取视频文件，根据配置调整视频尺寸或缩放比例，并加载视频的所有帧数据，
        生成包含帧信息的 `VideoObject` 实例。

        Parameters
        ----------
        vision : str
            视频文件的绝对路径。

        src_size : tuple
            原始视频的宽高尺寸，格式为 (width, height)。

        Returns
        -------
        VideoObject
            返回加载后的视频对象，包含全部帧数据及视频元信息。

        Notes
        -----
        - 当 `deploy.shape` 被定义时，将以固定尺寸 `shape=(width, height)` 加载视频帧；
        - 否则使用 `deploy.scale` 执行等比缩放；
        - 视频帧将根据配置决定是否保留彩色；
        - 加载完成后会显示帧总数、尺寸、加载耗时等信息。
        """
        start_time_ = time.time()  # 开始计时

        video = VideoObject(vision)  # 创建 VideoObject 对象

        # 记录视频帧长度和尺寸
        logger.debug(f"{(task_name_ := '视频帧长度: ' f'{video.frame_count}')}")
        logger.debug(f"{(task_info_ := '视频帧尺寸: ' f'{video.frame_size}')}")
        logger.debug(f"{(task_desc_ := '加载视频帧: ' f'{video.name}')}")
        self.design.show_panel(f"{task_name_}\n{task_info_}\n{task_desc_}", Wind.LOADER)

        # 调整视频尺寸，保持宽高比
        if self.deploy.shape:
            w, h, ratio = await Switch.ask_magic_frame(src_size, self.deploy.shape)
            shape, scale = (w, h), None
        else:
            shape, scale = None, self.deploy.scale or const.DEFAULT_SCALE

        logger.debug(f"调整视频帧: Shape={shape} Scale={scale}")

        # 加载视频帧
        video.load_frames(
            scale=scale, shape=shape, color=self.deploy.color
        )

        # 记录视频帧加载完成后的详细信息和耗时
        logger.debug(f"{(task_name := '视频帧加载完成: ' f'{video.frame_details(video.frames_data)}')}")
        logger.debug(f"{(task_info := '视频帧加载耗时: ' f'{time.time() - start_time_:.2f} 秒')}")
        self.design.show_panel(f"{task_name}\n{task_info}", Wind.LOADER)

        return video  # 返回 VideoObject 对象

    @staticmethod
    async def ask_frame_grid(vision: str) -> typing.Optional[str]:
        """
        检查视频文件或文件夹，返回首个可成功打开的视频路径。

        该方法用于判断传入路径是视频文件还是包含视频的目录，并尝试使用 OpenCV 打开首个可用视频。
        如果成功打开，则返回对应路径，否则返回 None。

        Parameters
        ----------
        vision : str
            视频文件路径或包含视频文件的目录路径。

        Returns
        -------
        Optional[str]
            若找到可成功打开的视频文件，返回该文件路径；否则返回 None。

        Notes
        -----
        - 此方法使用 `cv2.VideoCapture` 尝试加载视频；
        - 对目录的处理仅检查第一个可用的文件，未对文件类型做扩展名判断；
        - 无法打开的视频不会抛出异常，只返回 None。
        """
        target_screen = None

        # 检查 vision 是否为文件路径
        if os.path.isfile(vision):
            screen = cv2.VideoCapture(vision)
            if screen.isOpened():
                target_screen = vision
            screen.release()

        # 检查 vision 是否为目录路径
        elif os.path.isdir(vision):
            file_list = [
                file for file in os.listdir(vision) if os.path.isfile(os.path.join(vision, file))
            ]
            if len(file_list) >= 1:
                screen = cv2.VideoCapture(open_file := os.path.join(vision, file_list[0]))
                if screen.isOpened():
                    target_screen = open_file
                screen.release()

        return target_screen

    async def ask_exercise(self, vision: str, *args) -> typing.Optional[str]:
        """
        执行视频分析任务，提取并保存关键帧。

        该方法主要用于分析指定视频文件，获取稳定区域，通过压缩和切片的方式提取出若干具有代表性的帧图像，并将其保存到指定目录。

        Parameters
        ----------
        vision : str
            视频文件路径或包含视频文件的目录路径。

        *args : Any
            附加参数列表，其中第一个参数应为关键帧的保存路径，第二个为原始视频尺寸等分析配置。

        Returns
        -------
        Optional[str]
            若成功提取帧并保存，返回保存目录路径；若视频损坏或分析失败，返回 None。

        Raises
        ------
        FramixError
            视频加载或帧处理过程中可能抛出 FramixError，自定义异常未在此方法中主动抛出但可能间接发生。

        Notes
        -----
        - 此方法会自动判断输入是文件还是目录；
        - 视频帧压缩通过 VideoCutter 实现，并根据稳定性分析选出关键帧；
        - 提取过程不会修改原视频文件，所有输出保存在 query_path 路径中。
        """
        if not (target_vision := await self.ask_frame_grid(vision)):
            logger.debug(tip := f"视频文件损坏: {os.path.basename(vision)}")
            return self.design.show_panel(tip, Wind.KEEPER)

        query_path, src_size, *_ = args

        video = await self.ask_video_load(target_vision, src_size)

        logger.debug(f"引擎初始化: slide={self.deploy.slide}")
        cutter = VideoCutter(step=self.deploy.slide)

        logger.debug(f"{(cut_name := '视频帧长度: ' f'{video.frame_count}')}")
        logger.debug(f"{(cut_part := '视频帧片段: ' f'{video.frame_count - 1}')}")
        logger.debug(f"{(cut_info := '视频帧尺寸: ' f'{video.frame_size}')}")
        logger.debug(f"{(cut_desc := '压缩视频帧: ' f'{video.name}')}")
        self.design.show_panel(
            f"{cut_name}\n{cut_part}\n{cut_info}\n{cut_desc}", Wind.CUTTER
        )

        cut_start_time = time.time()

        logger.debug(
            f"压缩视频中: block={self.deploy.block}, scope={self.deploy.scope}, grade={self.deploy.grade}"
        )
        cut_range = cutter.cut(
            video=video,
            block=self.deploy.block,
            window_size=self.deploy.scope,
            window_coefficient=self.deploy.grade
        )

        logger.debug(f"{(cut_name := '视频帧压缩完成: ' f'{video.name}')}")
        logger.debug(f"{(cut_info := '视频帧压缩耗时: ' f'{time.time() - cut_start_time:.2f} 秒')}")
        self.design.show_panel(f"{cut_name}\n{cut_info}", Wind.CUTTER)

        stable, unstable = cut_range.get_range(
            threshold=self.deploy.thres, offset=self.deploy.shift
        )

        frame_count = 20

        pick_frame_path = cut_range.pick_and_save(
            stable, frame_count, query_path,
            meaningful_name=True, compress_rate=None, target_size=None, not_grey=True
        )

        return pick_frame_path

    async def ask_analyzer(self, vision: str, *args) -> typing.Optional["Review"]:
        """
        分析视频文件，提取视频帧并进行处理，返回分析结果。

        Parameters
        ----------
        vision : str
            视频文件路径或包含视频文件的目录路径。

        *args : Any
            可变参数，包含帧保存路径、额外图片路径、原始尺寸等。

        Returns
        -------
        typing.Optional[Review]
            包含分析结果的 Review 对象，如果失败则返回 None。

        Notes
        -----
        - 若 Keras 模型已加载，将执行结构化处理和分类分析。
        - 若无模型，则只进行基础帧提取并分析。
        """

        async def frame_forge(frame: "VideoFrame") -> typing.Any:
            """
            保存视频帧为图片文件。

            Parameters
            ----------
            frame : VideoFrame
                视频帧对象，包含帧数据和元信息。

            Returns
            -------
            dict
                如果成功，返回包含帧 ID 和图片保存路径的字典。

            Exception
                如果保存过程中发生异常，则返回异常对象。

            Notes
            -----
            每帧保存为 PNG 格式，文件名包含帧 ID 和时间戳。
            """
            try:
                # 构建图片文件名：帧ID + 帧时间戳（精确到小数点后5位）
                picture = f"{frame.frame_id}_{format(round(frame.timestamp, 5), '.5f')}.png"
                # 将帧图像数据编码为 PNG 格式的二进制数据
                _, codec = cv2.imencode(".png", frame.data)
                # 使用 aiofiles 异步写入文件（避免阻塞主线程）
                async with aiofiles.open(os.path.join(frame_path, picture), "wb") as f:
                    await f.write(codec.tobytes())
            except Exception as e:
                # 捕获异常并返回异常对象用于上层处理
                return FramixError(e)

            # 返回帧信息，图像文件相对路径用于后续展示
            return {"id": frame.frame_id, "picture": os.path.join(os.path.basename(frame_path), picture)}

        async def frame_flick() -> tuple:
            """
            提取视频的关键帧信息。

            Returns
            -------
            tuple
                返回 (开始帧ID, 结束帧ID, 时间成本) 三元组，分别表示关键帧范围及其持续时间。

            Notes
            -----
            - 起始与结束关键帧基于 `self.deploy.begin` 和 `self.deploy.final` 配置索引。
            - 如果索引无效（如越界或顺序错误），将回退到默认关键帧范围。
            - 展示帧信息面板，便于终端可视化关键帧提取结果。
            """

            # 从 deploy 配置中提取起始帧索引 (阶段索引, 阶段内帧索引)
            begin_stage_index, begin_frame_index = self.deploy.begin
            final_stage_index, final_frame_index = self.deploy.final

            # 打印提取关键帧的起始和结束阶段及帧索引
            logger.debug(
                f"{(extract := f'取关键帧: begin={list(self.deploy.begin)} final={list(self.deploy.final)}')}"
            )
            self.design.show_panel(extract, Wind.FASTER)

            try:
                # 获取视频的阶段信息并打印
                logger.debug(f"{(stage_name := f'阶段划分: {struct.get_ordered_stage_set()}')}")
                self.design.show_panel(stage_name, Wind.FASTER)
                # 获取所有非稳定阶段的帧范围（二维数组，按阶段分组）
                unstable_stage_range = struct.get_not_stable_stage_range()
                # 获取起始帧与结束帧对象
                begin_frame = unstable_stage_range[begin_stage_index][begin_frame_index]
                final_frame = unstable_stage_range[final_stage_index][final_frame_index]
            except (AssertionError, IndexError) as e:
                # 如果索引无效，使用默认首尾关键帧
                logger.debug(e)
                self.design.show_panel(e, Wind.KEEPER)
                begin_frame = struct.get_important_frame_list()[0]
                final_frame = struct.get_important_frame_list()[-1]

            # 若起始帧 ID 大于等于结束帧 ID，说明取值顺序出错，回退为完整帧范围
            if final_frame.frame_id <= begin_frame.frame_id:
                logger.debug(tip := f"{final_frame} <= {begin_frame}")
                self.design.show_panel(tip, Wind.KEEPER)
                begin_frame, end_frame = struct.data[0], struct.data[-1]

            # 计算帧间耗时（单位：秒）
            time_cost = final_frame.timestamp - begin_frame.timestamp

            # 获取帧 ID 和时间戳
            begin_id, begin_ts = begin_frame.frame_id, begin_frame.timestamp
            final_id, final_ts = final_frame.frame_id, final_frame.timestamp

            # 组织并展示格式化输出
            begin_fr, final_fr = f"{begin_id} - {begin_ts:.5f}", f"{final_id} - {final_ts:.5f}"
            logger.debug(f"开始帧:[{begin_fr}] 结束帧:[{final_fr}] 总耗时:[{(stage_cs := f'{time_cost:.5f}')}]")
            self.design.assort_frame(begin_fr, final_fr, stage_cs)

            # 返回关键帧 ID 和持续时间
            return begin_frame.frame_id, final_frame.frame_id, time_cost

        async def frame_hold() -> list:
            """
            获取并返回视频的所有帧数据列表。

            Returns
            -------
            list
                视频帧列表，每个元素为一个 `VideoFrame` 对象。

            Notes
            -----
            - 若 `struct` 不存在（即未进行结构分析），则直接返回原始帧；
            - 若启用 `boost` 参数，则额外添加非关键帧（如不稳定片段）以增强数据覆盖；
            - 使用 `toolbox.show_progress` 展示处理进度条。
            """

            # 如果 struct 不存在，说明没有结构分析，直接返回原始帧列表
            if not struct:
                return [i for i in video.frames_data]

            frames_list = []  # 最终帧列表

            important_frames = struct.get_important_frame_list()  # 获取关键帧列表

            # 如果启用了 boost 模式，将忽略连续地稳定帧
            if self.deploy.boost:
                # 使用进度条展示帧处理进度
                pbar = toolbox.show_progress(total=struct.get_length(), color=50)

                # 添加第一个关键帧
                frames_list.append(previous := important_frames[0])
                pbar.update(1)

                # 遍历其余关键帧
                for current in important_frames[1:]:
                    # 添加当前关键帧
                    frames_list.append(current)
                    pbar.update(1)

                    # 计算前后关键帧之间的帧距
                    frames_diff = current.frame_id - previous.frame_id
                    # 如果两帧都是不稳定帧，并且之间有空隙，则添加中间帧
                    if not previous.is_stable() and not current.is_stable() and frames_diff > 1:
                        for sample in struct.data[previous.frame_id: current.frame_id - 1]:
                            frames_list.append(sample)
                            pbar.update(1)

                    previous = current  # 更新上一帧
                pbar.close()  # 关闭进度条

            else:
                # 若未启用 boost，则直接返回所有结构帧
                for current in toolbox.show_progress(items=struct.data, color=50):
                    frames_list.append(current)

            return frames_list

        async def frame_flow() -> typing.Optional["ClassifierResult"]:
            """
            处理视频帧的裁剪、过滤与分类操作。

            Returns
            -------
            Optional[ClassifierResult]
                包含分类结果的结构化数据对象，若处理失败则返回 None。

            Notes
            -----
            - 使用 VideoCutter 添加多个 Hook 对视频帧进行裁剪与保存；
            - 支持尺寸调整、裁剪区域、忽略区域、保存图片等操作；
            - 过滤后的帧将被保存到 `extra_path` 指定目录；
            - 若启用了 Keras 模型，将对帧序列执行分类操作。
            """
            logger.debug(f"引擎初始化: slide={self.deploy.slide}")
            cutter = VideoCutter(step=self.deploy.slide)

            panel_hook_list = []  # 用于记录所有 hook 描述信息并展示

            # 添加尺寸调整 Hook（统一处理为等比例缩放）
            size_hook = FrameSizeHook(1.0, None, True)
            cutter.add_hook(size_hook)
            logger.debug(
                f"{(cut_size := f'视频帧处理: {size_hook.__class__.__name__} {[1.0, None, True]}')}"
            )
            panel_hook_list.append(cut_size)

            # 遍历 deploy 中配置的裁剪区域，添加裁剪 Hook
            if len(crop_list := self.deploy.crops) > 0 and sum([j for i in crop_list for j in i.values()]) > 0:
                for crop in crop_list:
                    x, y, x_size, y_size = crop.values()
                    crop_hook = PaintCropHook((y_size, x_size), (y, x))
                    cutter.add_hook(crop_hook)
                    logger.debug(
                        f"{(cut_crop := f'视频帧处理: {crop_hook.__class__.__name__} {x, y, x_size, y_size}')}"
                    )
                    panel_hook_list.append(cut_crop)

            # 遍历 deploy 中配置的忽略区域，添加忽略 Hook
            if len(omit_list := self.deploy.omits) > 0 and sum([j for i in omit_list for j in i.values()]) > 0:
                for omit in omit_list:
                    x, y, x_size, y_size = omit.values()
                    omit_hook = PaintOmitHook((y_size, x_size), (y, x))
                    cutter.add_hook(omit_hook)
                    logger.debug(
                        f"{(cut_omit := f'视频帧处理: {omit_hook.__class__.__name__} {x, y, x_size, y_size}')}"
                    )
                    panel_hook_list.append(cut_omit)

            # 添加保存 Hook，将处理后的帧图像保存到 extra_path
            save_hook = FrameSaveHook(extra_path)
            cutter.add_hook(save_hook)
            logger.debug(
                f"{(cut_save := f'视频帧处理: {save_hook.__class__.__name__} {[os.path.basename(extra_path)]}')}"
            )
            panel_hook_list.append(cut_save)

            # 输出所有 Hook 配置面板信息
            self.design.show_panel("\n".join(panel_hook_list), Wind.CUTTER)

            # 打印视频基础信息
            logger.debug(f"{(cut_name := '视频帧长度: ' f'{video.frame_count}')}")
            logger.debug(f"{(cut_part := '视频帧片段: ' f'{video.frame_count - 1}')}")
            logger.debug(f"{(cut_info := '视频帧尺寸: ' f'{video.frame_size}')}")
            logger.debug(f"{(cut_desc := '压缩视频帧: ' f'{video.name}')}")
            self.design.show_panel(
                f"{cut_name}\n{cut_part}\n{cut_info}\n{cut_desc}", Wind.CUTTER
            )
            cut_start_time = time.time()

            # 开始裁剪视频帧块
            logger.debug(
                f"压缩视频中: block={self.deploy.block}, scope={self.deploy.scope}, grade={self.deploy.grade}"
            )
            cut_range = cutter.cut(
                video=video,
                block=self.deploy.block,
                window_size=self.deploy.scope,
                window_coefficient=self.deploy.grade
            )

            # 裁剪完成后展示信息
            logger.debug(f"{(cut_name := '视频帧压缩完成: ' f'{video.name}')}")
            logger.debug(f"{(cut_info := '视频帧压缩耗时: ' f'{time.time() - cut_start_time:.2f} 秒')}")
            self.design.show_panel(f"{cut_name}\n{cut_info}", Wind.CUTTER)

            # 根据阈值与偏移值获取稳定与不稳定帧段
            stable, unstable = cut_range.get_range(
                threshold=self.deploy.thres, offset=self.deploy.shift
            )

            # 获取裁剪后保存的所有图片文件
            file_list = os.listdir(extra_path)
            file_list.sort(key=lambda n: int(n.split("(")[0]))
            total_images, desired_count = len(file_list), 12

            # 保留目标数量的图片索引
            if total_images <= desired_count:
                retain_indices = range(total_images)
            else:
                retain_indices = [int(i * (total_images / desired_count)) for i in range(desired_count)]
                if len(retain_indices) < desired_count:
                    retain_indices.append(total_images - 1)
                elif len(retain_indices) > desired_count:
                    retain_indices = retain_indices[:desired_count]

            # 删除不在保留列表中的图片文件
            for index, file in enumerate(file_list):
                if index not in retain_indices:
                    os.remove(os.path.join(extra_path, file))

            # 依次绘制裁剪图片文件的网格辅助线
            for draw in toolbox.show_progress(items=os.listdir(extra_path), color=146):
                toolbox.draw_line(os.path.join(extra_path, draw).format())

            try:
                # 使用 keras 模型进行结构分类
                struct_data = self.ks.classify(
                    video=video, valid_range=stable, keep_data=True
                )
            except AssertionError as e:
                logger.debug(e)
                return self.design.show_panel(e, Wind.KEEPER)

            return struct_data

        async def analytics_basic() -> tuple:
            """
            执行基础视频分析，保存帧图片并计算时间成本。

            Returns
            -------
            tuple
                返回一个元组，包含以下信息：
                - 开始帧 ID（int）
                - 结束帧 ID（int）
                - 分析耗时（float）
                - 帧 ID 与图片路径的映射字典（dict）
                - None：表示未使用模型结构分析

            Notes
            -----
            - 该函数将所有帧按每100张进行分块，并发保存为 PNG 图片。
            - 分析结果中不会包含结构体或模型输出，仅返回原始帧范围及对应图片路径。
            - 遇到保存失败时将记录异常，但整体流程不中断。
            """
            forge_tasks = [
                [frame_forge(frame) for frame in chunk] for chunk in
                [frames[i:i + 100] for i in range(0, len(frames), 100)]
            ]

            forge_result = await asyncio.gather(
                *(asyncio.gather(*forge) for forge in forge_tasks)
            )

            scores = {}
            for result in forge_result:
                for r in result:
                    if isinstance(r, Exception):
                        logger.debug(r)
                        self.design.show_panel(r, Wind.KEEPER)
                    else:
                        scores[r["id"]] = r["picture"]

            begin_frame, final_frame = frames[0], frames[-1]
            time_cost = final_frame.timestamp - begin_frame.timestamp

            return begin_frame.frame_id, final_frame.frame_id, time_cost, scores, None

        async def analytics_keras() -> tuple:
            """
            执行基于 Keras 模型的视频分析，保存帧图片并计算时间成本。

            Returns
            -------
            tuple
                返回一个元组，包含以下信息：
                - 开始帧 ID（int）
                - 结束帧 ID（int）
                - 分析耗时（float）
                - 帧 ID 与图片路径的映射字典（dict）
                - 分析结构体（ClassifierResult）

            Notes
            -----
            - 使用 Keras 模型结构提取关键帧，并生成结构体用于后续分析。
            - 帧保存操作与基础分析类似，也采用分块并发保存。
            - `frame_flick()` 负责返回关键帧范围及耗时，在异步中提前调度以节省时间。
            """
            flick_tasks = asyncio.create_task(frame_flick())

            forge_tasks = [
                [frame_forge(frame) for frame in chunk] for chunk in
                [frames[i:i + 100] for i in range(0, len(frames), 100)]
            ]

            forge_result = await asyncio.gather(
                *(asyncio.gather(*forge) for forge in forge_tasks)
            )

            scores = {}
            for result in forge_result:
                for r in result:
                    if isinstance(r, Exception):
                        logger.debug(r)
                        self.design.show_panel(r, Wind.KEEPER)
                    else:
                        scores[r["id"]] = r["picture"]

            begin_frame_id, final_frame_id, time_cost = await flick_tasks

            return begin_frame_id, final_frame_id, time_cost, scores, struct

        # 检查并获取有效的视频文件路径
        if not (target_vision := await self.ask_frame_grid(vision)):
            logger.debug(tip_ := f"视频文件损坏: {os.path.basename(vision)}")
            return self.design.show_panel(tip_, Wind.KEEPER)

        # 解包 args，依次是帧图片保存路径、额外图片保存路径、原始尺寸
        frame_path, extra_path, src_size, *_ = args

        # 加载视频帧对象 VideoObject，包含解码后所有帧
        video = await self.ask_video_load(target_vision, src_size)

        # 如果存在模型，则执行 frame_flow（分类并裁剪视频）
        struct = await frame_flow() if self.ks.model else None

        # 获取帧数据，如果存在 struct，则使用其结构提取关键帧与非关键帧
        frames = await frame_hold()

        # 根据是否有模型结构，选择执行 keras 模式或 basic 分析
        return Review(
            *(await analytics_keras())
        ) if struct else Review(*(await analytics_basic()))


async def main() -> typing.Coroutine | None:
    """
    命令分发调度器，根据命令行参数执行对应任务模块。
    """

    async def _scheduling() -> typing.Coroutine | typing.Any:
        """
        检查 scrcpy 是否已安装，如果未安装则显示安装提示并退出程序。
        """
        if shutil.which(third_party_app := "scrcpy"):
            return await Terminal.cmd_line([third_party_app, "--version"])
        raise FramixError("Install first https://github.com/Genymobile/scrcpy")

    async def _authorized() -> typing.Coroutine | None:
        """
        检查目录下的所有文件是否具备执行权限，如果文件没有执行权限，则自动添加 +x 权限。
        """
        if _platform != "darwin":
            return None

        if not (ensure := [
            kit for kit in [_adb, _fmp, _fpb] if not (Path(kit).stat().st_mode & stat.S_IXUSR)
        ]):
            return None

        for auth in ensure:
            logger.debug(f"Authorizing: {auth}")

        for resp in await asyncio.gather(
            *(Terminal.cmd_line(["chmod", "+x", kit]) for kit in ensure), return_exceptions=True
        ):
            logger.debug(f"Authorize: {resp}")

    async def _arithmetic(function: "typing.Callable", parameters: list[str]) -> typing.Coroutine | None:
        """
        执行通用异步任务函数，并预处理参数路径。
        """
        parameters = [(await Craft.revise_path(param)) for param in parameters]
        parameters = list(dict.fromkeys(parameters))
        await function(parameters, _option, _deploy)

    # Notes: Start from here
    await random.choice(
        [Design.engine_topology_wave, Design.stellar_glyph_binding]
    )(_level)  # 启动仪式

    await _authorized()  # 应用授权

    if _video_list := _lines.video:
        await _arithmetic(_missions.video_file_task, _video_list)

    elif _stack_list := _lines.stack:
        await _arithmetic(_missions.video_data_task, _stack_list)

    elif _train_list := _lines.train:
        await _arithmetic(_missions.train_model, _train_list)

    elif _build_list := _lines.build:
        await _arithmetic(_missions.build_model, _build_list)

    elif _lines.flick or _lines.carry or _lines.fully:
        tp_ver = await _scheduling()
        await _missions.analysis(_option, _deploy, tp_ver)

    elif _lines.paint:
        await _missions.painting(_option, _deploy)

    elif _lines.union:
        await _missions.combine_view(_lines.union)

    elif _lines.merge:
        await _missions.combine_main(_lines.merge)

    else:
        Design.help_document()

    await Design.engine_starburst(_level)  # 结尾动画


if __name__ == '__main__':
    """  
    **应用程序入口点，根据命令行参数初始化并运行主进程**

    ***********************
    *                     *
    *  Welcome to Framix  *
    *                     *
    ***********************    
    """

    try:
        signal.signal(signal.SIGINT, signal_processor)  # 设置 Ctrl + C 信号处理方式

        # 如果没有提供命令行参数，则显示应用程序标志和帮助文档，并退出程序
        if len(system_parameter_list := sys.argv) == 1:
            Design.minor_logo()
            Design.help_document()
            Design.done()
            sys.exit(Design.closure())

        Design.specially_logo()

        _wires = system_parameter_list[1:]  # 获取命令行参数（去掉第一个参数，即脚本名称）

        # 获取当前操作系统平台和应用名称
        _platform = sys.platform.strip().lower()
        _software = os.path.basename(os.path.abspath(sys.argv[0])).strip().lower()
        _sys_symbol = os.sep
        _env_symbol = os.path.pathsep

        # 根据应用名称确定工作目录和配置目录
        if _software == f"{const.NAME}.exe":
            # Windows
            _fx_work = os.path.dirname(os.path.abspath(sys.argv[0]))
            _fx_feasible = os.path.dirname(_fx_work)
        elif _software == f"{const.NAME}":
            # MacOS
            _fx_work = os.path.dirname(sys.executable)
            _fx_feasible = os.path.dirname(_fx_work)
        elif _software == f"{const.NAME}.py":
            # IDE
            _fx_work = os.path.dirname(os.path.abspath(__file__))
            _fx_feasible = _fx_work
        else:
            raise FramixError(f"{const.DESC} compatible with {const.NAME} command")

        # 设置模板文件路径
        _atom_total_temp = os.path.join(_fx_work, const.F_SCHEMATIC, "templates", "template_atom_total.html")
        _main_share_temp = os.path.join(_fx_work, const.F_SCHEMATIC, "templates", "template_main_share.html")
        _main_total_temp = os.path.join(_fx_work, const.F_SCHEMATIC, "templates", "template_main_total.html")
        _view_share_temp = os.path.join(_fx_work, const.F_SCHEMATIC, "templates", "template_view_share.html")
        _view_total_temp = os.path.join(_fx_work, const.F_SCHEMATIC, "templates", "template_view_total.html")

        # 检查每个模板文件是否存在，如果缺失则显示错误信息并退出程序
        for _tmp in (
                _temps := [
                    _atom_total_temp, _main_share_temp, _main_total_temp, _view_share_temp, _view_total_temp
                ]
        ):
            if os.path.isfile(_tmp) and os.path.basename(_tmp).endswith(".html"):
                continue
            _tmp_name = os.path.basename(_tmp)
            raise FramixError(f"{const.DESC} missing files {_tmp_name}")

        _turbo = os.path.join(_fx_work, const.F_SCHEMATIC, "supports").format()  # 设置工具源路径

        # 根据平台设置工具路径
        if _platform == "win32":
            # Windows
            _supports = os.path.join(_turbo, "Windows").format()
            _adb, _fmp, _fpb = "adb.exe", "ffmpeg.exe", "ffprobe.exe"
        else:
            # MacOS
            _supports = os.path.join(_turbo, "MacOS").format()
            _adb, _fmp, _fpb = "adb", "ffmpeg", "ffprobe"

        _adb = os.path.join(_supports, "platform-tools", _adb)
        _fmp = os.path.join(_supports, "ffmpeg", "bin", _fmp)
        _fpb = os.path.join(_supports, "ffmpeg", "bin", _fpb)

        # 将工具路径添加到系统 PATH 环境变量中
        for _tls in (_tools := [_adb, _fmp, _fpb]):
            os.environ["PATH"] = os.path.dirname(_tls) + _env_symbol + os.environ.get("PATH", "")

        # 检查每个工具是否存在，如果缺失则显示错误信息并退出程序
        for _tls in _tools:
            if not shutil.which((_tls_name := os.path.basename(_tls))):
                raise FramixError(f"{const.DESC} missing files {_tls_name}")

        # 设置初始路径
        if not os.path.exists(
                _initial_source := os.path.join(_fx_feasible, const.F_STRUCTURE).format()
        ):
            os.makedirs(_initial_source, exist_ok=True)

        # 设置模型路径
        if not os.path.exists(
                _src_model_place := os.path.join(_initial_source, const.F_SRC_MODEL_PLACE).format()
        ):
            os.makedirs(_src_model_place, exist_ok=True)

        # 设置报告路径
        if not os.path.exists(
                _src_total_place := os.path.join(_initial_source, const.F_SRC_TOTAL_PLACE).format()
        ):
            os.makedirs(_src_total_place, exist_ok=True)

        # Notes: 在 Windows 平台上启动多进程时确保冻结的可执行文件可以正确运行。
        freeze_support()

        # Notes: 此代码块必须在 `__main__` 块下调用，否则可能会导致多进程模块无法正确加载。
        _parser = Parser()
        _lines = _parser.parse_cmd

        # 激活日志记录功能，设置日志级别
        Active.active(_level := "DEBUG" if _lines.debug else "INFO")

        # 输出调试信息
        logger.debug(f"操作系统: {_platform}")
        logger.debug(f"应用名称: {_software}")
        logger.debug(f"系统路径: {_sys_symbol}")
        logger.debug(f"环境变量: {_env_symbol}")
        logger.debug(f"工具路径: {_turbo}")
        logger.debug(f"命令参数: {_wires}")
        logger.debug(f"日志等级: {_level}\n")

        logger.debug(f"* 环境 * {'=' * 30}")
        for _env in os.environ["PATH"].split(_env_symbol):
            logger.debug(f"{_env}")
        logger.debug(f"* 环境 * {'=' * 30}\n")

        logger.debug(f"* 工具 * {'=' * 30}")
        for _tls in _tools:
            logger.debug(f"{os.path.basename(_tls):7}: {_tls}")
        logger.debug(f"* 工具 * {'=' * 30}\n")

        logger.debug(f"* 模版 * {'=' * 30}")
        for _tmp in _temps:
            logger.debug(f"Html-Template: {_tmp}")
        logger.debug(f"* 模版 * {'=' * 30}\n")

        # 设置初始配置文件路径
        _initial_option = os.path.join(_initial_source, const.F_SRC_OPERA_PLACE, const.F_OPTION)
        _initial_deploy = os.path.join(_initial_source, const.F_SRC_OPERA_PLACE, const.F_DEPLOY)
        _initial_script = os.path.join(_initial_source, const.F_SRC_OPERA_PLACE, const.F_SCRIPT)
        logger.debug(f"配置文件路径: {_initial_option}")
        logger.debug(f"部署文件路径: {_initial_deploy}")
        logger.debug(f"脚本文件路径: {_initial_script}")

        # 加载初始配置
        _option = Option(_initial_option)
        _option.total_place = _option.total_place or _src_total_place
        _option.model_place = _option.model_place or _src_model_place
        _option.faint_model = _option.faint_model or const.FAINT_MODEL
        _option.color_model = _option.color_model or const.COLOR_MODEL
        for _attr_key, _attribute_value in _option.options.items():
            logger.debug(f"{_option.__class__.__name__} Current Key {_attr_key}")
            # Notes: 如果命令行中包含配置参数，无论是否存在配置文件，都将覆盖配置文件，以命令行参数为第一优先级。
            if any(_line.lower().startswith(f"--{(_attr_adapt := _attr_key.split('_')[0])}") for _line in _wires):
                setattr(_option, _attr_key, getattr(_lines, _attr_adapt))
                logger.debug(f"  Set <{_attr_key}> {_attribute_value} -> {getattr(_option, _attr_key)}")

        logger.debug(f"报告文件路径: {_option.total_place}")
        logger.debug(f"模型文件路径: {_option.model_place}")

        # 获取处理器核心数
        logger.debug(f"处理器核心数: {(_power := os.cpu_count())}")

        # 从命令行参数覆盖部署配置
        _deploy = Deploy(_initial_deploy)
        for _attr_key, _attribute_value in _deploy.deploys.items():
            logger.debug(f"{_deploy.__class__.__name__} Current Key {_attr_key}")
            for _attr, _attribute in _attribute_value.items():
                # 如果命令行中包含部署参数，无论是否存在部署文件，都将覆盖部署文件，以命令行参数为第一优先级
                if any(_line.lower().startswith(f"--{_attr}") for _line in _wires):
                    setattr(_deploy, _attr, getattr(_lines, _attr))
                    logger.debug(f"  {_attr_key} Set <{_attr}> {_attribute} -> {getattr(_deploy, _attr)}")

        """
        **将命令行参数解析结果转换为基本数据类型**

        注意:
            将命令行参数解析器解析得到的结果存储在基本数据类型的变量中。
            这样做的目的是避免在多进程环境中向子进程传递不可序列化的对象。
            因为这些对象在传递过程中可能会导致 `pickle.PicklingError` 错误。
        """
        _flick, _carry, _fully = _lines.flick, _lines.carry, _lines.fully
        _speed, _basic, _keras = _lines.speed, _lines.basic, _lines.keras
        _alone, _whist = _lines.alone, _lines.whist
        _alike, _shine = _lines.alike, _lines.shine
        _group = _lines.group

        # 打包位置参数
        _positions = _flick, _carry, _fully, _speed, _basic, _keras, _alone, _whist, _alike, _shine, _group

        # 打包关键字参数
        _keywords = {
            "atom_total_temp": _atom_total_temp,
            "main_share_temp": _main_share_temp,
            "main_total_temp": _main_total_temp,
            "view_share_temp": _view_share_temp,
            "view_total_temp": _view_total_temp,
            "initial_option": _initial_option,
            "initial_deploy": _initial_deploy,
            "initial_script": _initial_script,
            "adb": _adb,
            "fmp": _fmp,
            "fpb": _fpb
        }

        # Notes: 初始化主要任务对象
        _missions = Missions(_wires, _level, _power, *_positions, **_keywords)

        # Notes: Start from here
        asyncio.run(main())

    except KeyboardInterrupt:
        Design.exit()
        sys.exit(Design.closure())
    except (OSError, RuntimeError, MemoryError):
        Design.console.print_exception()
        Design.fail()
        sys.exit(1)
    except FramixError as _error:
        Design.console.print(_error)
        Design.fail()
        sys.exit(1)
    else:
        Design.done()
        sys.exit(Design.closure())
