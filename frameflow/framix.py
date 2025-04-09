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

"""
版权所有 (c) 2024  Framix(画帧秀)
此文件受 Framix(画帧秀) 许可证的保护。您可以在 LICENSE.md 文件中查看详细的许可条款。

Copyright (c) 2024  Framix(画帧秀)
This file is licensed under the Framix(画帧秀) License. See the LICENSE.md file for more details.
"""

__all__ = ["Clipix", "Alynex"]  # 接口

# ====[ 内置模块 ]====
import os
import re
import sys
import json
import time
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
from functools import partial
from multiprocessing import freeze_support
from concurrent.futures import ProcessPoolExecutor

# ====[ from: 第三方库 ]====
from loguru import logger
from rich.prompt import Prompt

# ====[ from: 本地模块 ]====
from engine.tinker import (
    Craft, Finder, Active, Review, FramixError
)
from engine.switch import Switch
from engine.terminal import Terminal
from frameflow.argument import Wind
from frameflow.skills.show import Show
from frameflow.skills.cubicle import DB
from frameflow.skills.parser import Parser
from frameflow.skills.profile import (
    Deploy, Option
)
from nexaflow import const
from nexaflow import toolbox
from nexaflow.report import Report
from nexaflow.video import (
    VideoObject, VideoFrame
)
from nexaflow.cutter.cutter import VideoCutter
from nexaflow.hook import (
    FrameSizeHook, FrameSaveHook, PaintCropHook, PaintOmitHook
)
from nexaflow.classifier.base import ClassifierResult
from nexaflow.classifier.keras_classifier import KerasStruct

_T = typing.TypeVar("_T")  # 定义类型变量


class Missions(object):
    """Missions"""

    # """Initialization"""
    def __init__(self, wires: list, level: str, power: int, *args, **kwargs):
        self.wires = wires  # 命令参数
        self.level = level  # 日志级别
        self.power = power  # 最大进程

        self.flick, self.carry, self.fully, self.speed, self.basic, self.keras, *_ = args
        *_, self.alone, self.whist, self.alike, self.shine, self.group = args

        self.atom_total_temp = kwargs["atom_total_temp"]
        self.main_share_temp = kwargs["main_share_temp"]
        self.main_total_temp = kwargs["main_total_temp"]
        self.view_share_temp = kwargs["view_share_temp"]
        self.view_total_temp = kwargs["view_total_temp"]
        self.initial_option = kwargs["initial_option"]
        self.initial_deploy = kwargs["initial_deploy"]
        self.initial_script = kwargs["initial_script"]
        self.adb = kwargs["adb"]
        self.fmp = kwargs["fmp"]
        self.fpb = kwargs["fpb"]

    # """Child Process"""
    def amazing(self, option: "Option", deploy: "Deploy", vision: str, *args) -> "_T":
        """
        异步分析视频的子进程方法。

        该方法在异步进程执行器中执行，用于加载 Keras 模型并分析视频。

        参数:
            option (Option): 选项对象，包含模型路径和其他运行时选项配置。
            deploy (Deploy): 配置信息对象，包含视频处理的各项配置。
            vision (str): 视频文件路径。
            *args: 传递给分析器的其他参数。

        返回:
            处理完成的结果。

        注意:
            避免传递复杂对象，或者传递的对象是可以序列化的，或者传递基本数据类型。
            该方法需要在异步进程执行器中执行。

        代码逻辑:
            1. 获取当前事件循环。
            2. 根据 `option` 和 `self.keras` 决定模型路径，并创建 Alynex 实例。
            3. 尝试加载 Keras 模型，如果失败则捕获并处理异常。
            4. 运行 Alynex 的 `ask_analyzer` 方法，传递视频路径和其他参数。
            5. 等待异步操作完成，并返回结果。
        """
        loop = asyncio.get_event_loop()

        matrix = option.model_place if self.keras else None
        alynex = Alynex(matrix, option, deploy, self.level)
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

        该方法在异步进程执行器中执行，用于分析视频，并利用 Alynex 工具进行处理。

        参数:
            option (Option): 选项对象，包含模型路径和其他运行时选项配置。
            deploy (Deploy): 配置信息对象，包含视频处理的各项配置。
            vision (str): 视频文件路径。
            *args: 传递给分析器的其他参数。

        返回:
            处理完成的结果。

        注意:
            避免传递复杂对象，或者传递的对象是可以序列化的，或者传递基本数据类型。
            该方法需要在异步进程执行器中执行。

        代码逻辑:
            1. 获取当前事件循环。
            2. 创建 Alynex 实例，使用提供的配置进行初始化。
            3. 运行 Alynex 的 `ask_exercise` 方法，传递视频路径和其他参数。
            4. 等待异步操作完成，并返回结果。
        """
        loop = asyncio.get_event_loop()

        alynex = Alynex(None, option, deploy, self.level)

        loop_complete = loop.run_until_complete(
            alynex.ask_exercise(vision, *args)
        )
        return loop_complete

    @staticmethod
    async def enforce(db: "DB", style: str, total: str, title: str, nest: str) -> None:
        """
        将分析数据插入数据库表中，并确保表结构存在。

        参数:
            db (DB): 数据库连接对象，必须实现 `create` 和 `insert` 方法。
            style (str): 分析方式，用于指定数据处理或分析的类型。
            total (str): 报告的根目录，表示存储分析结果的根文件夹路径。
            title (str): 标题信息，通常是数据集的名称。
            nest (str): 嵌套信息，表示可能包含的子结构或子数据的标识。

        返回:
            None: 此方法不返回任何内容。

        异常:
            TypeError: 如果 `db` 对象未实现 `create` 或 `insert` 方法。
            DatabaseError: 如果数据库操作失败，可能抛出特定的数据库异常。
            ValueError: 如果任何参数的值不符合预期的格式或范围。

        功能:
            1. 创建一个包含 `style`、`total`、`title` 和 `nest` 列的表结构。如果表已存在，跳过创建步骤。
            2. 将提供的分析方式、报告根目录、标题和嵌套信息插入到数据库的对应列中。

        说明:
            - 此方法是异步的，在调用时不会阻塞主线程或事件循环。
            - 数据库对象 `db` 必须具有 `create` 和 `insert` 方法。
            - 在插入数据之前，此方法将确保表结构已经创建。
        """
        await db.create(column_list := ["style", "total", "title", "nest"])
        await db.insert(column_list, [style, total, title, nest])

    async def fst_track(
            self, deploy: "Deploy", clipix: "Clipix", task_list: list[list], *_, **__
    ) -> tuple[list, list]:
        """
        异步执行视频的处理追踪，包括内容提取和平衡视频长度等功能。

        此函数用于根据指定的部署配置，提取视频内容，并尝试将多个视频的长度调整为一致。主要处理包括解析视频信息、视频内容提取、时间平衡和删除临时文件。

        参数:
            deploy (Deploy): 配置信息对象，包含视频处理的起始、结束、限制时间和帧率等。
            clipix (Clipix): 视频处理工具对象，负责具体的视频内容提取和平衡操作。
            task_list (list[list]): 包含视频信息的任务列表，每个列表项包括视频模板和其他参数。

        返回:
            tuple[list, list]: 返回处理后的原始视频列表和指示信息列表。

        注意:
            - 该函数为异步函数，需要在异步环境中运行。
            - 函数内部使用了多个异步 gather 来并行处理视频操作，提高效率。
            - 确保提供的每个视频都符合 `Deploy` 中定义的处理标准。
            - 异常处理：确保处理过程中捕获并妥善处理可能发生的任何异常，以避免程序中断。

        处理流程:
            1. 初始化事件循环并解析视频处理的起始、结束和限制时间。
            2. 异步提取视频内容，获取视频尺寸、实际帧率、平均帧率、视频时长等信息。
            3. 如果需要，将多个视频的长度调整为一致，并记录相关日志信息。
            4. 删除临时文件，确保资源被妥善释放。

        功能细节:
            - 提取视频内容：调用 `clipix.vision_content` 方法解析视频信息，包括帧率、视频时长等。
            - 平衡视频长度：如果需要，将多个视频的长度调整为一致。
            - 日志记录：记录视频处理的详细信息，并在控制台输出。
            - 删除临时文件：确保在处理完成后删除不再需要的临时文件。

        代码逻辑:
            1. 获取事件循环，并解析视频处理的时间参数。
            2. 使用 `clipix.vision_content` 方法异步提取视频内容。
            3. 记录视频处理的详细信息，包括视频尺寸、帧率、时长等。
            4. 如果需要，调用 `clipix.vision_balance` 方法平衡视频长度。
            5. 删除临时文件，释放资源。
            6. 返回处理后的原始视频列表和指示信息列表。
        """
        looper = asyncio.get_event_loop()

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
            if self.level == const.SHOW_LEVEL:
                Show.content_pose(
                    rlt, avg, f"{dur:.6f}", org, vd_start, vd_close, vd_limit, video_temp, deploy.frate
                )
        *_, durations, originals, indicates = zip(*content_list)

        # Video Balance
        eliminate = []
        if self.alike and len(task_list) > 1:
            logger.debug(tip := f"平衡时间: [{(standard := min(durations)):.6f}] [{Parser.parse_times(standard)}]")
            Show.show_panel(self.level, tip, Wind.STANDARD)
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
            Show.show_panel(self.level, "\n".join(panel_blc_list), Wind.TAILOR)
        await asyncio.gather(*eliminate, return_exceptions=True)

        return originals, indicates

    async def fst_waves(
            self, deploy: "Deploy", clipix: "Clipix", task_list: list[list], originals: list, *_, **__
    ) -> list:
        """
        异步执行视频的过滤和改进操作。

        根据提供的配置参数，应用一系列过滤器对视频进行处理，改进视频质量。处理包括调整帧率、颜色、模糊和锐化等操作。

        参数:
            deploy (Deploy): 包含处理参数的部署配置对象。
            clipix (Clipix): 视频处理工具对象，负责具体的视频内容调整操作。
            task_list (list[list]): 包含视频任务信息的列表，每个列表项包括视频路径和其他相关参数。
            originals (list): 原始视频列表，用于提取和处理视频内容。

        返回:
            Tuple: 包含处理后的视频过滤列表。

        处理流程:
            1. 根据配置参数初始化过滤器列表。
            2. 异步执行视频过滤操作，对每个原始视频应用过滤器。
            3. 记录和显示过滤操作的日志信息。

        注意:
            - 该方法是异步的，需要在异步环境中调用。
            - 确保提供的每个视频都符合 Deploy 中定义的处理标准。
            - 异常处理：确保处理过程中捕获并妥善处理可能发生的任何异常，以避免程序中断。

        功能细节:
            - 初始化过滤器：根据 deploy 配置中的帧率、颜色格式、高斯模糊和锐化参数设置过滤器列表。
            - 异步处理：使用 `asyncio.gather` 并行执行视频过滤操作，对每个原始视频应用过滤器。
            - 日志记录：记录每个过滤操作的详细信息，并在控制台输出。

        代码逻辑:
            1. 根据 `deploy` 配置初始化过滤器列表，包括帧率调整、颜色格式转换、模糊和锐化操作。
            2. 使用 `clipix.vision_improve` 方法异步处理每个原始视频，应用过滤器。
            3. 记录并显示过滤操作的日志信息。
            4. 返回处理后的视频过滤列表。
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
        Show.show_panel(self.level, "\n".join(panel_filter_list), Wind.FILTER)

        return video_filter_list

    async def als_speed(
            self, deploy: "Deploy", clipix: "Clipix", report: "Report", task_list: list[list], *_, **__
    ) -> None:
        """
        异步执行视频的速度分析和调整，包括视频过滤、尺寸调整等功能。

        此函数根据指定的部署配置（deploy）处理视频内容，进行视频尺寸和帧率的调整，并记录视频处理过程中的关键信息。
        主要操作包括应用视频过滤器，改变视频尺寸和帧率，以及处理并输出视频处理的细节和结果。

        参数:
            deploy (Deploy): 配置信息对象，包含视频处理的帧率、颜色格式、尺寸等配置。
            clipix (Clipix): 视频处理工具对象，负责具体的视频内容调整操作。
            report (Report): 报告处理对象，负责记录和展示处理结果。
            task_list (list[list]): 包含视频和其他相关参数的任务列表。
            originals (list): 原始视频列表，用于提取和处理视频内容。
            indicates (list): 指示信息列表，包含视频处理的具体指标和参数。

        返回:
            None: 此函数没有返回值，所有结果通过日志和报告对象进行记录和展示。

        注意:
            - 该函数为异步函数，需要在异步环境中运行。
            - 函数内部使用了多个异步gather来并行处理视频操作，提高效率。
            - 确保提供的每个视频都符合Deploy中定义的处理标准。
            - 异常处理：确保处理过程中捕获并妥善处理可能发生的任何异常，以避免程序中断。
        功能细节:
            - 初始设置：通过 `deploy` 和 `clipix` 对象进行初始设置和参数调整。
            - 视频过滤：调用 `als_waves` 函数对视频进行预处理和过滤。
            - 视频拆帧：根据 `task_list` 中的任务配置，对视频进行拆帧操作。
            - 结果处理：使用 `asyncio.gather` 并行处理视频操作，生成分析报告。

        代码逻辑:
            1. 获取事件循环，并初始化相关对象。
            2. 使用 `als_track` 函数获取原始视频和指示信息。
            3. 使用 `als_waves` 函数对视频进行过滤处理。
            4. 设置视频目标路径，并调用 `clipix.pixels` 函数处理视频帧。
            5. 处理拆帧结果，并记录相关信息。
            6. 渲染分析结果并生成报告，存储在指定路径中。
        """
        logger.debug(f"**<* 光速穿梭 *>**")
        Show.show_panel(self.level, Wind.SPEED_TEXT, Wind.SPEED)

        originals, indicates = await self.fst_track(deploy, clipix, task_list)

        video_filter_list = await self.fst_waves(deploy, clipix, task_list, originals)

        video_target_list = [
            (flt, frame_path) for flt, (*_, frame_path, _, _) in zip(video_filter_list, task_list)
        ]

        detach_result = await asyncio.gather(
            *(clipix.pixels(
                Switch.ask_video_detach, video_filter, video_temp, target, **points
            ) for (video_filter, target), (video_temp, *_), points in zip(video_target_list, task_list, indicates))
        )

        for detach, (video_temp, *_) in zip(detach_result, task_list):
            message_list = []
            for message in detach.splitlines():
                if matcher := re.search(r"frame.*fps.*speed.*", message):
                    discover: typing.Any = lambda x: re.findall(r"(\w+)=\s*([\w.\-:/x]+)", x)
                    message_list.append(
                        format_msg := " ".join([f"{k}={v}" for k, v in discover(matcher.group())])
                    )
                    logger.debug(format_msg)
                elif re.search(r"Error", message, re.IGNORECASE):
                    Show.show_panel(self.level, "\n".join(message_list), Wind.KEEPER)
            Show.show_panel(self.level, "\n".join(message_list), Wind.METRIC)

        async def render_speed(todo_list: list[list]):
            total_path: typing.Any
            query_path: typing.Any
            frame_path: typing.Any
            extra_path: typing.Any
            proto_path: typing.Any

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
            logger.debug(f"Speeder: {(nest := json.dumps(result, ensure_ascii=False))}")
            await report.load(result)

            return result.get("style"), result.get("total"), result.get("title"), nest

        # Speed Analyzer Result
        render_result = await asyncio.gather(
            *(render_speed(todo_list) for todo_list in task_list)
        )

        async with DB(os.path.join(report.reset_path, const.DB_FILES_NAME).format()) as db:
            await asyncio.gather(
                *(self.enforce(db, *ns) for ns in render_result)
            )

    async def als_keras(
            self, deploy: "Deploy", clipix: "Clipix", report: "Report", task_list: list[list], *_, **kwargs
    ) -> None:
        """
        异步执行视频的 Keras 模式分析或基本模式分析，包括视频过滤、尺寸调整和动态模板渲染等功能。

        此函数根据部署配置（deploy）调整视频帧率和尺寸，执行视频分析，并根据分析结果采用不同模式处理视频。
        如果启用了 Keras 模型，执行深度学习模型分析；否则执行基本分析。

        参数:
            deploy (Deploy): 配置信息对象，包含视频处理的帧率、颜色格式、尺寸等配置。
            clipix (Clipix): 视频处理工具对象，负责具体的视频内容调整和分析操作。
            report (Report): 报告处理对象，负责记录和展示处理结果。
            task_list (list[list]): 包含视频和其他相关参数的任务列表。
            *_: 接受并忽略所有传入的位置参数。
            **kwargs: 其他可选参数，包括以下关键字参数：
                option (Option): 选项对象，包含各种运行时选项配置，用于控制分析和处理流程。
                alynex (Alynex): 模型分析工具，决定使用 Keras 模型还是基础分析。

        注意:
            - 该函数为异步函数，需要在异步环境中运行。
            - 函数内部使用了多个异步 gather 来并行处理视频操作，提高效率。
            - 函数的执行路径依赖于 `alynex.ks.model` 的状态，确保 Alynex 实例正确初始化。
            - 异常处理：确保处理过程中捕获并妥善处理可能发生的任何异常，以避免程序中断。

        功能细节:
            - 初始设置：通过 `deploy` 和 `clipix` 对象进行初始设置和参数调整。
            - 视频过滤：调用 `als_waves` 函数对视频进行预处理和过滤。
            - 视频拆帧：根据 `task_list` 中的任务配置，对视频进行拆帧操作。
            - 结果处理：使用 `asyncio.gather` 并行处理视频操作，生成分析报告。

        代码逻辑:
            1. 获取事件循环，并从 kwargs 中提取选项和分析工具实例。
            2. 使用 `als_track` 函数获取原始视频和指示信息。
            3. 使用 `als_waves` 函数对视频进行过滤处理。
            4. 设置视频目标路径，并调用 `clipix.pixels` 函数处理视频帧。
            5. 处理拆帧结果，并使用 `os.remove` 删除临时文件。
            6. 根据 `alynex.ks.model` 的状态决定调用深度学习分析模型还是基础分析。
            7. 渲染分析结果并生成报告，存储在指定路径中。
        """
        looper = asyncio.get_event_loop()

        option = kwargs["option"]
        alynex = kwargs["alynex"]

        logger.debug(f"**<* 思维导航 *>**" if alynex.ks.model else f"**<* 基石阵地 *>**")
        Show.show_panel(
            self.level,
            Wind.KERAS_TEXT if alynex.ks.model else Wind.BASIC_TEXT,
            Wind.KERAS if alynex.ks.model else Wind.BASIC
        )

        originals, indicates = await self.fst_track(deploy, clipix, task_list)

        video_filter_list = await self.fst_waves(deploy, clipix, task_list, originals)

        video_target_list = [
            (flt, os.path.join(
                os.path.dirname(video_temp), f"vision_fps{deploy.frate}_{random.randint(100, 999)}.mp4")
             ) for flt, (video_temp, *_) in zip(video_filter_list, task_list)
        ]

        change_result = await asyncio.gather(
            *(clipix.pixels(
                Switch.ask_video_change, video_filter, video_temp, target, **points
            ) for (video_filter, target), (video_temp, *_), points in zip(video_target_list, task_list, indicates))
        )

        eliminate = []
        for change, (video_temp, *_) in zip(change_result, task_list):
            message_list = []
            for message in change.splitlines():
                if matcher := re.search(r"frame.*fps.*speed.*", message):
                    discover: typing.Any = lambda x: re.findall(r"(\w+)=\s*([\w.\-:/x]+)", x)
                    message_list.append(
                        format_msg := " ".join([f"{k}={v}" for k, v in discover(matcher.group())])
                    )
                    logger.debug(format_msg)
                elif re.search(r"Error", message, re.IGNORECASE):
                    Show.show_panel(self.level, "\n".join(message_list), Wind.KEEPER)
            Show.show_panel(self.level, "\n".join(message_list), Wind.METRIC)
            eliminate.append(
                looper.run_in_executor(None, os.remove, video_temp)
            )
        await asyncio.gather(*eliminate, return_exceptions=True)

        if alynex.ks.model:
            deploy.view_deploy()

        # Ask Analyzer
        if len(task_list) == 1:
            task = [
                alynex.ask_analyzer(target, frame_path, extra_path, src_size)
                for (_, target), src_size, (*_, frame_path, extra_path, _)
                in zip(video_target_list, originals, task_list)
            ]
            futures = await asyncio.gather(*task)

        else:
            this_level = self.level
            self.level = "ERROR"
            func = partial(self.amazing,  option, deploy)
            with ProcessPoolExecutor(self.power, None, Active.active, ("ERROR",)) as exe:
                task = [
                    looper.run_in_executor(exe, func, target, frame_path, extra_path, src_size)
                    for (_, target), src_size, (*_, frame_path, extra_path, _)
                    in zip(video_target_list, originals, task_list)
                ]
                futures = await asyncio.gather(*task)
            self.level = this_level

        # Template
        if isinstance(atom_tmp := await Craft.achieve(self.atom_total_temp), Exception):
            logger.debug(tip := f"{atom_tmp}")
            return Show.show_panel(self.level, tip, Wind.KEEPER)

        async def render_keras(future: "Review", todo_list: list[list]):
            total_path: str
            query_path: str
            frame_path: str
            extra_path: str
            proto_path: str
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
                logger.debug(tip_ := f"模版引擎渲染完成 {os.path.basename(stages_inform)}")
                Show.show_panel(self.level, tip_, Wind.REPORTER)
                result["extra"] = os.path.basename(extra_path)
                result["proto"] = os.path.basename(stages_inform)
                result["style"] = "keras"
            else:
                result["style"] = "basic"

            logger.debug(f"Restore: {(nest := json.dumps(result, ensure_ascii=False))}")
            await report.load(result)

            return result.get("style"), result.get("total"), result.get("title"), nest

        # Keras Analyzer Result
        render_result = await asyncio.gather(
            *(render_keras(future, todo_list) for future, todo_list in zip(futures, task_list) if future)
        )

        async with DB(os.path.join(report.reset_path, const.DB_FILES_NAME).format()) as db:
            await asyncio.gather(
                *(self.enforce(db, *ns) for ns in render_result)
            )

    async def combine(self, report: "Report") -> None:
        """
        异步生成组合报告的方法。

        该方法用于检查报告对象的范围列表，并根据配置调用适当的报告生成方法来创建组合报告。

        参数:
            report (Report): 报告处理对象，包含处理和展示结果的路径和范围列表。

        返回:
            None: 此函数没有返回值，所有结果通过日志和报告对象进行记录和展示。

        注意:
            - 该函数为异步函数，需要在异步环境中运行。
            - 如果范围列表为空，则记录并展示没有可生成的报告信息。
            - 根据 `self.speed` 配置，调用不同的报告生成方法 (`combine_view` 或 `combine_main`)。
            - 异常处理：确保处理过程中捕获并妥善处理可能发生的任何异常，以避免程序中断。
        """
        if report.range_list:
            function = getattr(self, "combine_view" if self.speed else "combine_main")
            return await function([os.path.dirname(report.total_path)])

        logger.debug(tip := f"没有可以生成的报告")
        return Show.show_panel(self.level, tip, Wind.KEEPER)

    async def combine_crux(self, share_temp: str, total_temp: str, merge: list) -> None:
        """
        异步生成汇总报告的方法。

        该方法用于根据共享模板和汇总模板生成汇总报告，并将多个子报告合并成一个总报告。

        参数:
            share_temp (str): 共享模板路径，用于生成部分共享报告内容。
            total_temp (str): 汇总模板路径，用于生成完整的汇总报告。
            merge (list): 需要合并的子报告列表。

        返回:
            None: 此函数没有返回值，所有结果通过日志和报告对象进行记录和展示。

        注意:
            - 该函数为异步函数，需要在异步环境中运行。
            - 异步获取模板内容，并确保模板获取成功。
            - 异常处理：确保处理过程中捕获并妥善处理可能发生的任何异常，以避免程序中断。
            - 生成汇总报告，并记录和展示处理结果。
        """
        template_list = await asyncio.gather(
            Craft.achieve(share_temp), Craft.achieve(total_temp),
            return_exceptions=True
        )

        if isinstance(template_list, Exception):
            logger.debug(tip := f"{template_list}")
            return Show.show_panel(self.level, tip, Wind.KEEPER)

        share_form, total_form = template_list

        logger.debug(tip := f"正在生成汇总报告 ...")
        Show.show_panel(self.level, tip, Wind.REPORTER)
        state_list: list[str | Exception] = await asyncio.gather(
            *(Report.ask_create_total_report(m, self.group, share_form, total_form) for m in merge),
            return_exceptions=True
        )

        for state in state_list:
            if isinstance(state, Exception):
                tip_state, tip_style = f"{state}", Wind.KEEPER
            else:
                tip_state, tip_style = f"成功生成汇总报告 {os.path.basename(state)}", Wind.REPORTER
            logger.debug(tip_state)
            logger.debug(state)
            Show.show_panel(self.level, tip_state, tip_style)
            Show.show_panel(self.level, state, tip_style)

    # """时空纽带分析系统"""
    async def combine_view(self, merge: list) -> None:
        # 合并视图数据。
        await self.combine_crux(
            self.view_share_temp, self.view_total_temp, merge
        )

    # """时序融合分析系统"""
    async def combine_main(self, merge: list) -> None:
        # 合并视图数据。
        await self.combine_crux(
            self.main_share_temp, self.main_total_temp, merge
        )

    # """视频解析探索"""
    async def video_file_task(self, video_file_list: list, option: "Option", deploy: "Deploy") -> None:
        """
        异步处理视频文件任务，并根据配置选项进行分析。

        参数:
            video_file_list (list): 包含视频文件路径的列表。
            option (Option): 配置选项对象，包含分析任务的相关配置。
            deploy (Deploy): 部署配置对象，包含视频处理的具体参数。

        返回:
            None: 任务完成后没有返回值，结果将通过日志和报告面板展示。

        功能说明:
            1. 检查视频文件列表中的有效文件，如果没有有效文件，将记录并显示错误信息。
            2. 初始化Clipix和Report对象，用于视频处理和报告生成。
            3. 将视频文件复制到报告目录中，并生成相关任务列表。
            4. 根据speed选项决定执行何种分析方式:
                - 如果启用speed选项，调用`als_speed`进行快速分析。
                - 否则，初始化Alynex并加载模型，调用`als_keras`进行深度学习分析。
            5. 任务结束后，生成最终报告。

        注意:
            - 如果视频文件列表为空，将直接返回并记录相应的日志信息。
            - 在执行Keras分析时，若模型加载失败，将捕获异常并记录。

        主要流程:
            1. 过滤有效视频文件并复制到目标目录。
            2. 创建和配置Clipix和Report对象。
            3. 根据配置执行相应的分析操作（快速分析或深度学习分析）。
            4. 生成并展示分析报告。
        """
        if not (video_file_list := [video_file for video_file in video_file_list if os.path.isfile(video_file)]):
            logger.debug(tip := f"没有有效任务")
            return Show.show_panel(self.level, tip, Wind.KEEPER)

        clipix = Clipix(self.fmp, self.fpb)
        report = Report(option.total_place)
        report.title = f"{const.DESC}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

        # Profession
        task_list = []
        for video_file in video_file_list:
            report.query = f"{os.path.basename(video_file).split('.')[0]}_{time.strftime('%Y%m%d%H%M%S')}"
            new_video_path = os.path.join(report.video_path, os.path.basename(video_file))
            shutil.copy(video_file, new_video_path.format())
            task_list.append(
                [new_video_path, None, report.total_path, report.title, report.query_path,
                 report.query, report.frame_path, report.extra_path, report.proto_path]
            )

        # Pack Argument
        attack = deploy, clipix, report, task_list

        if self.speed:
            # Speed Analyzer
            await self.als_speed(*attack)
        else:
            # Initial Alynex
            matrix = option.model_place if self.keras else None
            alynex = Alynex(matrix, option, deploy, self.level)
            try:
                await alynex.ask_model_load()
            except FramixError as e:
                logger.debug(e)
                Show.show_panel(self.level, e, Wind.KEEPER)

            # Keras Analyzer
            await self.als_keras(*attack, option=option, alynex=alynex)

        # Create Report
        await self.combine(report)

    # """影像堆叠导航"""
    async def video_data_task(self, video_data_list: list, option: "Option", deploy: "Deploy") -> None:
        """
        异步处理视频数据任务，并根据配置选项进行分析。

        参数:
            video_data_list (list): 包含视频数据路径的列表。
            option (Option): 配置选项对象，包含分析任务的相关配置。
            deploy (Deploy): 部署配置对象，包含视频处理的具体参数。

        功能说明:
            1. 使用`finder`对象加速搜索视频数据文件。
            2. 对每个有效的搜索结果生成对应的报告和任务列表。
            3. 根据speed选项决定执行何种分析方式:
                - 如果启用speed选项，调用`als_speed`进行快速分析。
                - 否则，初始化Alynex并加载模型，调用`als_keras`进行深度学习分析。
            4. 每次任务完成后，生成最终报告。

        注意:
            - 如果`finder`搜索结果返回异常，将记录日志并显示错误信息。
            - 在执行Keras分析时，若模型加载失败，将捕获异常并记录。

        主要流程:
            1. 使用`finder`加速搜索视频数据，并过滤有效结果。
            2. 生成对应的报告对象并初始化任务列表。
            3. 根据配置执行相应的分析操作（快速分析或深度学习分析）。
            4. 生成并展示分析报告。
        """

        async def load_entries():
            # 加载视频数据条目。
            for video_data in video_data_list:
                finder_result = finder.accelerate(video_data)
                if isinstance(finder_result, Exception):
                    logger.debug(finder_result)
                    Show.show_panel(self.level, finder_result, Wind.KEEPER)
                    continue
                tree, collection_list = finder_result
                Show.console.print(tree)
                yield collection_list[0]

        finder = Finder()
        clipix = Clipix(self.fmp, self.fpb)

        # Profession
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

                    # Pack Argument
                    attack = deploy, clipix, report, task_list

                    if self.speed:
                        # Speed Analyzer
                        await self.als_speed(*attack)
                    else:
                        # Initial Alynex
                        matrix = option.model_place if self.keras else None
                        alynex = Alynex(matrix, option, deploy, self.level)
                        try:
                            await alynex.ask_model_load()
                        except FramixError as e:
                            logger.debug(e)
                            Show.show_panel(self.level, e, Wind.KEEPER)

                        # Keras Analyzer
                        await self.als_keras(*attack, option=option, alynex=alynex)

                # Create Report
                await self.combine(report)

    # """模型训练大师"""
    async def train_model(self, video_file_list: list, option: "Option", deploy: "Deploy") -> None:
        """
        异步模型训练任务。

        参数:
            video_file_list (list): 包含视频文件路径的列表。
            option (Option): 配置选项对象，包含模型训练的相关配置。
            deploy (Deploy): 部署配置对象，包含视频处理的具体参数。

        功能说明:
            1. 过滤无效视频文件，并生成对应的报告和任务列表。
            2. 根据部署配置，执行视频跟踪、过滤和调整操作。
            3. 调用Alynex执行模型训练任务，并根据任务数量选择单任务或多任务模式。
            4. 处理视频转换后的结果，并清理临时文件。

        处理步骤:
            1. 验证并过滤有效的视频文件路径。
            2. 使用`Clipix`对象和`Report`对象对视频进行处理，并生成任务列表。
            3. 执行视频跟踪(`als_track`)、过滤(`als_waves`)和视频调整操作。
            4. 根据任务数量选择合适的分析方式:
                - 单任务模式下直接调用`Alynex`进行分析。
                - 多任务模式下，使用多进程池执行分析任务，并处理返回结果。
            5. 记录分析结果并清理临时生成的视频文件。

        注意:
            - 如果视频文件列表为空，将记录日志并显示错误信息。
            - 在多任务模式下，设置`self.level`为`ERROR`级别以确保多进程中的正确日志记录。
            - 临时文件在任务完成后被清理以释放存储空间。
        """
        if not (video_file_list := [video_file for video_file in video_file_list if os.path.isfile(video_file)]):
            logger.debug(tip := f"没有有效任务")
            return Show.show_panel(self.level, tip, Wind.KEEPER)

        import uuid

        looper = asyncio.get_event_loop()

        clipix = Clipix(self.fmp, self.fpb)
        report = Report(option.total_place)

        # Profession
        task_list = []
        for video_file in video_file_list:
            report.title = f"Model_{uuid.uuid4()}"
            new_video_path = os.path.join(report.video_path, os.path.basename(video_file))
            shutil.copy(video_file, new_video_path.format())
            task_list.append(
                [new_video_path, None, report.total_path, report.title, report.query_path,
                 report.query, report.frame_path, report.extra_path, report.proto_path]
            )

        # Information
        originals, indicates = await self.fst_track(deploy, clipix, task_list)

        video_filter_list = await self.fst_waves(deploy, clipix, task_list, originals)

        video_target_list = [
            (flt, os.path.join(
                report.query_path, f"tmp_fps{deploy.frate}_{random.randint(10000, 99999)}.mp4")
             ) for flt, (video_temp, *_) in zip(video_filter_list, task_list)
        ]

        change_result = await asyncio.gather(
            *(clipix.pixels(
                Switch.ask_video_change, video_filter, video_temp, target, **points
            ) for (video_filter, target), (video_temp, *_), points in zip(video_target_list, task_list, indicates))
        )

        eliminate = []
        for change, (video_temp, *_) in zip(change_result, task_list):
            message_list = []
            for message in change.splitlines():
                if matcher := re.search(r"frame.*fps.*speed.*", message):
                    discover: typing.Any = lambda x: re.findall(r"(\w+)=\s*([\w.\-:/x]+)", x)
                    message_list.append(
                        format_msg := " ".join([f"{k}={v}" for k, v in discover(matcher.group())])
                    )
                    logger.debug(format_msg)
                elif re.search(r"Error", message, re.IGNORECASE):
                    Show.show_panel(self.level, "\n".join(message_list), Wind.KEEPER)
            Show.show_panel(self.level, "\n".join(message_list), Wind.METRIC)
            eliminate.append(
                looper.run_in_executor(None, os.remove, video_temp)
            )
        await asyncio.gather(*eliminate, return_exceptions=True)

        # Initial Alynex
        alynex = Alynex(None, option, deploy, self.level)

        # Ask Analyzer
        if len(task_list) == 1:
            task = [
                alynex.ask_exercise(target, query_path, src_size)
                for (_, target), src_size, (_, _, _, _, query_path, *_)
                in zip(video_target_list, originals, task_list)
            ]
            futures = await asyncio.gather(*task)

        else:
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

        pick_info_list = []
        for future in futures:
            if future:
                logger.debug(tip := f"保存: {os.path.basename(future)}")
                pick_info_list.append(tip)
        Show.show_panel(self.level, "\n".join(pick_info_list), Wind.PROVIDER)

        await asyncio.gather(
            *(looper.run_in_executor(None, os.remove, target) for (_, target) in video_target_list)
        )

    # """模型编译大师"""
    async def build_model(self, video_data_list: list, option: "Option", deploy: "Deploy") -> None:
        """
        异步模型构建任务。

        参数:
            video_data_list (list): 包含视频数据文件夹路径的列表。
            option (Option): 配置选项对象，包含模型构建的相关配置。
            deploy (Deploy): 部署配置对象，包含视频处理的具体参数。

        功能说明:
            1. 过滤无效的视频数据文件夹，生成报告和任务列表。
            2. 执行视频数据的文件夹搜索、图片通道分析和模型构建操作。
            3. 根据任务数量选择单任务或多任务模式进行模型构建。
            4. 记录构建结果并显示模型构建的详细信息。

        处理步骤:
            1. 验证并过滤有效的视频数据文件夹路径。
            2. 使用`conduct`函数搜索文件夹中的数据，根据子文件夹的命名规则排序并生成列表。
            3. 使用`channel`函数对文件夹中的图片进行通道分析，确定图片的色彩信息和图像形状。
            4. 根据分析结果生成模型构建任务列表，并调用Alynex的`ks.build`函数进行模型构建。
            5. 根据任务数量选择合适地执行模式:
                - 单任务模式下直接在主进程中执行模型构建。
                - 多任务模式下使用多进程池进行并行构建。
            6. 记录和展示模型构建的成功信息。

        注意:
            - 如果视频数据文件夹列表为空，将记录日志并显示错误信息。
            - 在多任务模式下，设置`self.level`为`ERROR`级别以确保多进程中的正确日志记录。
            - 处理过程中可能抛出的异常将记录并展示为错误信息。
        """
        if not (video_data_list := [video_data for video_data in video_data_list if os.path.isdir(video_data)]):
            logger.debug(tip := f"没有有效任务")
            return Show.show_panel(self.level, tip, Wind.KEEPER)

        looper = asyncio.get_event_loop()

        async def conduct():
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

        async def channel():
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
                Show.show_panel(self.level, "\n".join(channel_list), Wind.DESIGNER)
            return image_learn, image_color, image_aisle

        async def measure(image):
            if image.ndim != 3:
                return "grayscale", 1, f"Image: {list(image.shape)} is grayscale image"
            if numpy.array_equal(image[:, :, 0], image[:, :, 1]) and numpy.array_equal(image[:, :, 1], image[:, :, 2]):
                return "grayscale", 1, f"Image: {list(image.shape)} is grayscale image, stored in RGB format"
            return "rgb", image.ndim, f"Image: {list(image.shape)} is color image"

        alynex = Alynex(None, option, deploy, self.level)
        report = Report(option.total_place)

        task_list = []
        for video_data in video_data_list:
            logger.debug(tip := f"搜索文件夹: {os.path.basename(video_data)}")
            Show.show_panel(self.level, tip, Wind.DESIGNER)
            if dirs_list := await conduct():
                logger.debug(tip := f"分类文件夹: {os.path.basename(cf_src := os.path.dirname(dirs_list[0]))}")
                Show.show_panel(self.level, tip, Wind.DESIGNER)
                try:
                    ready_image, ready_color, ready_aisle = await channel()
                    image_shape = deploy.shape if deploy.shape else ready_image.shape
                except Exception as e:
                    logger.debug(e)
                    Show.show_panel(self.level, e, Wind.KEEPER)
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
                Show.show_panel(self.level, tip, Wind.KEEPER)

        if len(task_list) == 0:
            logger.debug(tip := f"没有有效任务")
            return Show.show_panel(self.level, tip, Wind.KEEPER)

        # Ask Analyzer
        if len(task_list) == 1:
            task = [
                looper.run_in_executor(None, alynex.ks.build, *compile_data)
                for compile_data in task_list
            ]
            futures = await asyncio.gather(*task)

        else:
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

        final_model_list = []
        for future in futures:
            logger.debug(tip := f"Model saved successfully {os.path.basename(future)}")
            final_model_list.append(tip)
        Show.show_panel(self.level, "\n".join(final_model_list), Wind.DESIGNER)

    # """线迹创造者"""
    async def painting(self, option: "Option", deploy: "Deploy") -> None:
        """
        使用设备截图进行绘制操作，并在图像上添加网格线。

        参数:
            deploy ("Deploy"): 部署对象，包含绘制配置。

        功能描述:
            1. 初始化绘图所需的库和路径。
            2. 获取设备列表，并对每个设备进行以下操作:
                - 获取设备截图并保存到本地临时目录。
                - 根据配置将图像转换为灰度或保持彩色。
                - 对图像进行裁剪或省略操作。
                - 调整图像大小并添加网格线。
                - 显示最终处理的图像。
            3. 处理完成后，询问用户是否保存图片，并根据用户选择保存图片。
        """

        import PIL.Image
        import PIL.ImageDraw
        import PIL.ImageFont

        async def paint_lines(device):
            """
            处理单个设备的截图，进行图像操作并添加网格线。

            参数:
                device (object): 设备对象，包含设备的相关信息。

            返回:
                PIL.Image: 处理后的图像对象。

            功能描述:
                1. 获取设备截图并保存到本地临时目录。
                2. 根据配置将图像转换为灰度或保持彩色。
                3. 对图像进行裁剪或省略操作。
                4. 调整图像大小并添加网格线。
                5. 显示最终处理的图像。
            """
            image_folder = "/sdcard/Pictures/Shots"
            image = f"{time.strftime('%Y%m%d%H%M%S')}_{random.randint(100, 999)}_" + "Shot.png"
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

                image_file = PIL.Image.open(image_save_path)
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
                Show.show_panel(self.level, tip, Wind.DRAWER)

                if new_w == new_h:
                    x_line_num, y_line_num = 10, 10
                elif new_w > new_h:
                    x_line_num, y_line_num = 10, 20
                else:
                    x_line_num, y_line_num = 20, 10

                resized = image_file.resize((new_w, new_h))

                draw = PIL.ImageDraw.Draw(resized)
                font = PIL.ImageFont.load_default()

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

        manage = Manage(self.adb)
        device_list = await manage.operate_device()

        resized_result = await asyncio.gather(
            *(paint_lines(device) for device in device_list)
        )

        while True:
            action = Prompt.ask(
                f"[bold]保存图片([bold #5FD700]Y[/]/[bold #FF87AF]N[/])?[/]",
                console=Show.console, default="Y"
            )
            if action.strip().upper() == "Y":
                report = Report(option.total_place)
                report.title = f"Hooks_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
                for device, resize_img in zip(device_list, resized_result):
                    img_save_path = os.path.join(
                        report.query_path, f"hook_{device.sn}_{random.randint(10000, 99999)}.png"
                    )
                    resize_img.save(img_save_path)
                    logger.debug(tip_ := f"保存图片: {os.path.basename(img_save_path)}")
                    Show.show_panel(self.level, tip_, Wind.DRAWER)
                break
            elif action.strip().upper() == "N":
                break
            else:
                tip_ = f"没有该选项,请重新输入\n"
                Show.show_panel(self.level, tip_, Wind.KEEPER)

    # """循环节拍器 | 脚本驱动者 | 全域执行者"""
    async def analysis(self, option: "Option", deploy: "Deploy") -> None:

        async def anything_film():

            # 初始化并启动设备的视频录制任务
            async def wait_for_device(device):
                # 等待设备上线
                Show.notes(f"[bold #FAFAD2]Wait Device Online -> {device.tag} {device.sn}")
                await Terminal.cmd_line([self.adb, "-s", device.sn, "wait-for-device"])

            Show.notes(f"**<* {('独立' if self.alone else '全局')}控制模式 *>**")

            await source.monitor()

            await asyncio.gather(
                *(wait_for_device(device) for device in device_list)
            )

            media_screen_w, media_screen_h = ScreenMonitor.screen_size()
            Show.notes(f"Media Screen W={media_screen_w} H={media_screen_h}")

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

        async def anything_over():
            # 完成任务后的处理
            effective_list = await asyncio.gather(
                *(record.ask_close_record(device, video_temp, transports)
                  for device, (video_temp, transports, *_) in zip(device_list, task_list))
            )

            check_list = []
            for idx, (effective, video_name) in enumerate(effective_list):
                if "视频录制失败" in effective:
                    try:
                        task = task_list.pop(idx)
                        logger.debug(tip := f"{effective}: {video_name} 移除: {os.path.basename(task[0])}")
                        check_list.append(tip)
                    except IndexError:
                        continue
                else:
                    logger.debug(tip := f"{effective}: {video_name}")
                    check_list.append(tip)
            Show.show_panel(self.level, "\n".join(check_list), Wind.EXPLORER)

        async def anything_well():
            # 执行任务处理，根据不同模式选择适当的分析方法
            if len(task_list) == 0:
                logger.debug(tip := f"没有有效任务")
                return Show.show_panel(self.level, tip, Wind.KEEPER)

            # Pack Argument
            attack = deploy, clipix, report, task_list

            if self.speed:
                # Speed Analyzer
                await self.als_speed(*attack)
            elif self.basic or self.keras:
                # Keras Analyzer
                await self.als_keras(*attack, option=option, alynex=alynex)
            else:
                logger.debug(tip := f"**<* 录制模式 *>**")
                Show.show_panel(self.level, tip, Wind.EXPLORER)

        async def load_timer():
            # 并行执行定时任务，对设备列表中的每个设备进行计时操作
            await asyncio.gather(
                *(record.check_timer(device, timer_mode) for device in device_list)
            )

        # 加载并解析传入的 carry 字符串，返回包含执行指令的字典或异常
        async def load_carry(carry):
            # 解析传入的 carry 字符串，分割为路径和关键字两部分
            if len(parts := re.split(r",|;|!|\s", carry)) >= 2:
                loc_file, *key_list = parts
                # 异步加载执行字典
                if isinstance(exec_dict := await load_fully(loc_file), Exception):
                    return exec_dict

                try:
                    # 查找关键字对应的执行指令
                    return {key: value for key in key_list if (value := exec_dict.get(key, None))}
                # 如果 carry 字符串格式不正确，抛出值错误
                except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                    return e

            raise ValueError("参数错误")

        # 异步加载和解析完整的命令文件
        async def load_fully(fully):
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
                return e
            return exec_dict

        async def call_commands(bean, live_devices, exec_func, exec_vals, exec_args, exec_kwds):
            """
            异步执行命令。

            参数:
                - bean: 要操作的对象实例，通常包含需要调用的方法。
                - live_devices: 活动设备的列表或字典，用于管理当前正在处理的设备。
                - exec_func: 字符串类型，表示要调用的函数名称。
                - exec_vals: 位置参数列表，传递给目标函数。
                - exec_args: 额外的参数列表，与 `exec_vals` 一起传递给目标函数。
                - exec_kwds: 关键字参数字典，传递给目标函数。

            功能说明:
                1. 动态获取 `bean` 对象中的指定方法 (`exec_func`) 并检查其可调用性。
                2. 如果方法不可调用，记录调试信息并显示提示。
                3. 获取对象实例的序列号 (`sn`) 或类名，用于日志记录和显示。
                4. 调用指定的可调用方法 (`function`) 并传递所有参数 (`exec_vals`, `exec_args`, `exec_kwds`)。
                5. 处理异步函数的执行结果，记录和显示返回值。
                6. 在捕获到 `asyncio.CancelledError` 异常时，将设备从 `live_devices` 中移除并记录退出信息。
                7. 如果发生其他异常，捕获并返回异常对象。

            返回值:
                - 成功时返回函数的执行结果。
                - 如果方法不可调用或在执行过程中发生异常，返回异常对象。
                - 如果发生取消错误 (`asyncio.CancelledError`)，从 `live_devices` 中移除设备并退出。
            """
            if not (callable(function := getattr(bean, exec_func, None))):
                logger.debug(tip := f"No callable {exec_func}")
                return Show.show_panel(self.level, tip, Wind.KEEPER)

            sn = getattr(bean, "sn", bean.__class__.__name__)
            try:
                logger.debug(tip := f"{sn} {function.__name__} {exec_vals}")
                Show.show_panel(self.level, tip, Wind.EXPLORER)
                if inspect.iscoroutinefunction(function):
                    if call_result := await function(*exec_vals, *exec_args, **exec_kwds):
                        logger.debug(tip := f"Returns: {call_result}")
                        return Show.show_panel(self.level, tip, Wind.EXPLORER)
            except asyncio.CancelledError:
                live_devices.pop(sn)
                logger.debug(tip := f"{sn} Call Commands Exit")
                Show.show_panel(self.level, tip, Wind.EXPLORER)
            except Exception as e:
                return e

        async def pack_commands(resolve_list):
            """
            异步命令打包函数。

            参数:
                - resolve_list (list): 包含解析信息的列表，每个元素都是字典，通常包含`cmds`、`vals`、`args`和`kwds`等键。

            功能说明:
                1. 遍历 `resolve_list` 列表，处理每个解析项中的命令、值、参数和关键字参数。
                2. 对 `cmds` 列表中的每个命令进行检查，确保它们是非空的字符串。
                3. 对 `vals`、`args` 和 `kwds` 列表进行规范化处理，确保它们的每个元素符合预期的类型（列表或字典）。
                4. 如果 `cmds` 列表长度与其他列表不匹配，使用空列表或空字典进行补充，以确保各列表长度一致。
                5. 将处理后的命令、值、参数和关键字参数组合成元组，并追加到 `exec_pairs_list` 列表中。

            返回值:
                - list: 包含命令、值、参数和关键字参数配对的列表。每个元素都是一个四元组，格式为 `(cmd, vals, args, kwds)`。
            """
            exec_pairs_list = []
            for resolve in resolve_list:
                device_cmds_list = resolve.get("cmds", [])
                if all(isinstance(device_cmds, str) and device_cmds != "" for device_cmds in device_cmds_list):
                    # 去除重复命令
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

        async def exec_commands(exec_pairs_list, *change):
            """
            异步执行命令函数。

            参数:
                - exec_pairs_list (list): 包含命令和其参数配对的列表。每个元素都是一个四元组，格式为 `(exec_func, exec_vals, exec_args, exec_kwds)`。
                - *change: 可选的变化参数，用于在命令执行过程中动态替换某些值。

            功能说明:
                1. 定义 `substitute_star` 内部函数，用于替换 `exec_args` 中的 "*" 为 `change` 中的相应值。
                2. 初始化 `live_devices` 字典，包含当前活动设备的副本，以保证设备状态的一致性。
                3. 创建用于停止任务的异步任务列表 `stop_tasks`，这些任务会在所有命令执行完毕后被取消。
                4. 遍历 `exec_pairs_list`，对每对命令及其参数进行处理：
                    - 如果 `exec_func` 是 "audio_player"，直接调用对应的 `call_commands` 方法处理。
                    - 对于其他 `exec_func`，为每个设备创建异步任务，并使用 `asyncio.create_task` 启动这些任务。
                5. 使用 `asyncio.gather` 并发执行所有任务，并捕获任务执行状态。
                6. 在所有任务执行完成后，清空任务字典 `exec_tasks`，并记录或显示异常信息（如果有）。
                7. 在任务结束后，取消所有停止任务，以确保所有异步操作都已安全终止。
            """

            async def substitute_star(replaces):
                substitute = iter(change)
                return [
                    "".join(next(substitute, "*") if c == "*" else c for c in i)
                    if isinstance(i, str) else (next(substitute, "*") if i == "*" else i) for i in replaces
                ]

            live_devices = {device.sn: device for device in device_list}.copy()

            exec_tasks: dict[str, "asyncio.Task"] = {}
            stop_tasks: list["asyncio.Task"] = []

            for device in device_list:
                stop_tasks.append(
                    asyncio.create_task(record.check_event(device, exec_tasks), name="stop"))

            for exec_pairs in exec_pairs_list:
                if len(live_devices) == 0:
                    return Show.notes(f"[bold #F0FFF0 on #000000]All tasks canceled")
                for exec_func, exec_vals, exec_args, exec_kwds in exec_pairs:
                    exec_vals = await substitute_star(exec_vals)
                    if exec_func == "audio_player":
                        await call_commands(player, live_devices, exec_func, exec_vals, exec_args, exec_kwds)
                    else:
                        for device in live_devices.values():
                            exec_tasks[device.sn] = asyncio.create_task(
                                call_commands(device, live_devices, exec_func, exec_vals, exec_args, exec_kwds))

                try:
                    exec_status_list = await asyncio.gather(*exec_tasks.values())
                except asyncio.CancelledError:
                    return Show.notes(f"[bold #F0FFF0 on #000000]All tasks canceled")
                finally:
                    exec_tasks.clear()

                for status in exec_status_list:
                    if isinstance(status, Exception):
                        logger.debug(status)
                        Show.show_panel(self.level, status, Wind.KEEPER)

            for stop in stop_tasks:
                stop.cancel()

        # 初始化操作，为后续的程序运行做准备
        manage_ = Manage(self.adb)
        device_list = await manage_.operate_device()

        clipix = Clipix(self.fmp, self.fpb)

        matrix = option.model_place if self.keras else None
        alynex = Alynex(matrix, option, deploy, self.level)
        try:
            await alynex.ask_model_load()
        except FramixError as e_:
            logger.debug(e_)
            Show.show_panel(self.level, e_, Wind.KEEPER)

        titles_ = {"speed": "Speed", "basic": "Basic", "keras": "Keras"}
        input_title_ = next((title_ for key_, title_ in titles_.items() if getattr(self, key_)), "Video")

        vs_ = await Terminal.cmd_line(["scrcpy", "--version"])
        record = Record(
            vs_, alone=self.alone, whist=self.whist, frate=deploy.frate
        )
        player = Player()
        source = SourceMonitor()

        lower_bound_, upper_bound_ = (8, 300) if self.whist else (5, 300)

        # Flick Loop 处理控制台应用程序中的复杂交互过程，主要负责管理设备显示、设置报告以及通过命令行界面处理各种用户输入
        if self.flick:
            report = Report(option.total_place)
            report.title = f"{input_title_}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

            timer_mode = lower_bound_

            while True:
                try:
                    await manage_.display_device()
                    start_tips_ = f"<<<按 Enter 开始 [bold #D7FF5F]{timer_mode}[/] 秒>>>"
                    if action_ := Prompt.ask(f"[bold #5FD7FF]{start_tips_}", console=Show.console):
                        if (select_ := action_.strip().lower()) == "device":
                            device_list = await manage_.another_device()
                            continue
                        elif select_ == "cancel":
                            Show.exit()
                            sys.exit(Show.closure())
                        elif "header" in select_:
                            if match_ := re.search(r"(?<=header\s).*", select_):
                                if hd_ := match_.group().strip():
                                    src_hd_, a_, b_ = f"{input_title_}_{time.strftime('%Y%m%d_%H%M%S')}", 10000, 99999
                                    Show.notes(f"{const.SUC}New title set successfully")
                                    report.title = f"{src_hd_}_{hd_}" if hd_ else f"{src_hd_}_{random.randint(a_, b_)}"
                                    continue
                            raise FramixError
                        elif select_ == "create":
                            await self.combine(report)
                            break
                        elif select_ == "deploy":
                            Show.notes(f"{const.WRN}请完全退出编辑器再继续操作")
                            deploy.dump_deploy(self.initial_deploy)
                            if sys.platform == "win32":
                                first_ = ["notepad++"] if shutil.which("notepad++") else ["Notepad"]
                            else:
                                first_ = ["open", "-W", "-a", "TextEdit"]
                            await Terminal.cmd_line(first_ + self.initial_deploy)
                            deploy.load_deploy(self.initial_deploy)
                            deploy.view_deploy()
                            continue
                        elif select_.isdigit():
                            timer_value_ = int(select_)
                            if timer_value_ > upper_bound_ or timer_value_ < lower_bound_:
                                bound_tips_ = f"{lower_bound_} <= [bold #FFD7AF]Time[/] <= {upper_bound_}"
                                Show.notes(f"[bold #FFFF87]{bound_tips_}")
                            timer_mode = max(lower_bound_, min(upper_bound_, timer_value_))
                        else:
                            raise FramixError
                except FramixError:
                    Show.tips_document()
                    continue
                else:
                    task_list = await anything_film()
                    await load_timer()
                    await anything_over()
                    await anything_well()
                    check_ = await record.flunk_event()
                    device_list = await manage_.operate_device() if check_ else device_list
                finally:
                    await record.clean_event()

        # Other Loop 执行批量脚本任务，并根据脚本中的配置进行操作
        elif self.carry or self.fully:

            if self.carry:
                load_script_data_ = await asyncio.gather(
                    *(load_carry(carry_) for carry_ in self.carry), return_exceptions=True
                )
            elif self.fully:
                load_script_data_ = await asyncio.gather(
                    *(load_fully(fully_) for fully_ in self.fully), return_exceptions=True
                )
            else:
                raise FramixError(f"Script file does not exist")

            for device_ in device_list:
                logger.debug(tip_ := f"{device_.sn} Automator Activating")
                Show.show_panel(self.level, tip_, Wind.EXPLORER)

            try:
                await asyncio.gather(
                    *(device_.automator_activation() for device_ in device_list)
                )
            except Exception as e_:
                raise FramixError(e_)

            for device_ in device_list:
                logger.debug(tip_ := f"{device_.sn} Automator Activation Success")
                Show.show_panel(self.level, tip_, Wind.EXPLORER)

            for script_data_ in load_script_data_:
                if isinstance(script_data_, Exception):
                    raise FramixError(script_data_)
            script_storage_ = [script_data_ for script_data_ in load_script_data_]

            await manage_.display_device()
            for script_dict_ in script_storage_:
                report = Report(option.total_place)
                for script_key_, script_value_ in script_dict_.items():
                    logger.debug(tip_ := f"Batch Exec: {script_key_}")
                    Show.show_panel(self.level, tip_, Wind.EXPLORER)

                    # 根据 script_value_ 中的 parser 参数更新 deploy 配置
                    if (parser_ := script_value_.get("parser", {})) and type(parser_) is dict:
                        for deploy_key_, deploy_value_ in deploy.deploys.items():
                            logger.debug(f"Current Key {deploy_key_}")
                            for d_key_, d_value_ in deploy_value_.items():
                                # 以命令行参数为第一优先级
                                if any(line_.lower().startswith(f"--{d_key_}") for line_ in self.wires):
                                    logger.debug(f"    Line First <{d_key_}> = {getattr(deploy, d_key_)}")
                                    continue
                                setattr(deploy, d_key_, parser_.get(deploy_key_, {}).get(d_key_, {}))
                                logger.debug(f"    Parser Set <{d_key_}>  {d_value_} -> {getattr(deploy, d_key_)}")

                    # 处理 script_value_ 中的 header 参数
                    header_ = header_ if type(
                        header_ := script_value_.get("header", [])
                    ) is list else ([header_] if type(header_) is str else [time.strftime("%Y%m%d%H%M%S")])

                    # 处理 script_value_ 中的 change 参数
                    if change_ := script_value_.get("change", []):
                        change_ = change_ if type(change_) is list else (
                            [change_] if type(change_) is str else [str(change_)])

                    # 处理 script_value_ 中的 looper 参数
                    try:
                        looper_ = int(looper_) if (looper_ := script_value_.get("looper", None)) else 1
                    except ValueError as e_:
                        logger.debug(tip_ := f"重置循环次数 {(looper_ := 1)} {e_}")
                        Show.show_panel(self.level, tip_, Wind.EXPLORER)

                    # 处理 script_value_ 中的 prefix 参数
                    if prefix_list_ := script_value_.get("prefix", []):
                        prefix_list_ = await pack_commands(prefix_list_)
                    # 处理 script_value_ 中的 action 参数
                    if action_list_ := script_value_.get("action", []):
                        action_list_ = await pack_commands(action_list_)
                    # 处理 script_value_ 中的 suffix 参数
                    if suffix_list_ := script_value_.get("suffix", []):
                        suffix_list_ = await pack_commands(suffix_list_)

                    # 遍历 header 并执行任务
                    for hd_ in header_:
                        report.title = f"{input_title_}_{script_key_}_{hd_}"
                        extend_task_list_ = []

                        for _ in range(looper_):
                            # prefix 前置任务
                            if prefix_list_:
                                await exec_commands(prefix_list_)

                            # start record 开始录屏
                            task_list = await anything_film()

                            # action 主要任务
                            if action_list_:
                                change_list_ = [hd_ + c_ for c_ in change_] if change_ else [hd_]
                                await exec_commands(action_list_, *change_list_)

                            # close record 结束录屏
                            await anything_over()

                            # 检查事件并更新设备列表，清除所有事件
                            check_ = await record.flunk_event()
                            device_list = await manage_.operate_device() if check_ else device_list
                            await record.clean_event()

                            # suffix 提交后置任务
                            suffix_task_list_ = []
                            if suffix_list_:
                                suffix_task_list_.append(
                                    asyncio.create_task(exec_commands(suffix_list_), name="suffix"))

                            # 根据参数判断是否分析视频以及使用哪种方式分析
                            if self.shine:
                                extend_task_list_.extend(task_list)
                            else:
                                await anything_well()

                            # 等待后置任务完成
                            await asyncio.gather(*suffix_task_list_)

                        # 分析视频集合
                        if task_list := (extend_task_list_ if self.shine else []):
                            await anything_well()

                # 如果需要，结合多种模式生成最终报告
                if any((self.speed, self.basic, self.keras)):
                    await self.combine(report)

        # Empty Loop
        else:
            raise FramixError(f"Command does not exist")


class Clipix(object):
    """Clipix"""

    def __init__(self, fmp: str, fpb: str):
        self.fmp = fmp  # 表示 `ffmpeg` 的路径
        self.fpb = fpb  # 表示 `ffprobe` 的路径

    async def vision_content(
            self, video_temp: str, start: typing.Optional[str], close: typing.Optional[str], limit: typing.Optional[str],
    ) -> tuple[str, str, float, tuple, dict]:
        """
        异步获取特定视频文件的内容分析，包括实际和平均帧率、视频时长及其视觉处理点。

        此函数分析视频文件，提供关键视频流参数，如帧率和时长，并根据输入的起始、结束和限制时间点计算处理后的视觉时间点。

        参数:
            video_temp (str): 视频文件的路径。
            start (Optional[str]): 视频处理的起始时间点（如 "00:00:10" 表示从第10秒开始）。如果为None，则从视频开始处处理。
            close (Optional[str]): 视频处理的结束时间点。如果为None，则处理到视频结束。
            limit (Optional[str]): 处理视频的最大时长限制。如果为None，则没有时间限制。

        返回:
            tuple[str, str, float, tuple, dict]: 返回一个包含以下元素的元组：
                - rlt (str): 实际帧率。
                - avg (str): 平均帧率。
                - duration (float): 视频总时长（秒）。
                - original (tuple): 原始视频分辨率和其他基础数据。
                - vision_point (dict): 处理后的起始、结束和限制时间点（格式化为字符串如"00:00:10"）。
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

        此函数计算原视频与标准时长的差值，基于这一差值调整视频的开始和结束时间点，以生成新的视频文件，保证其总时长接近标准时长。适用于需要统一视频播放长度的场景。

        参数:
            duration (float): 原视频的总时长（秒）。
            standard (float): 目标视频的标准时长（秒）。
            src (str): 原视频文件的路径。
            frate (float): 目标视频的帧率。

        返回:
            tuple[str, str]: 包含两个元素：
                - video_dst (str): 调整时长后生成的新视频文件的路径。
                - video_blc (str): 描述视频时长调整详情的字符串。
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
        异步方法，用于改进视频的视觉效果，通过调整视频尺寸和应用过滤器列表。

        参数:
            - deploy (Deploy): 包含处理参数的部署配置对象。
            - original (tuple): 原始视频的尺寸，格式为 (原始宽度, 原始高度)。
            - filters (list): 初始过滤器列表，可以包括例如 'blur', 'contrast' 等过滤器命令。

        返回:
            - list: 包含所有过滤器命令的列表，包括用于调整尺寸的 'scale' 过滤器。

        抛出:
            - ValueError: 如果输入的参数类型不符合预期。
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
            self, function: "typing.Callable", video_filter: list, src: str, dst: str, **kwargs
    ) -> tuple[str]:
        """
        执行视频过滤处理函数，应用指定的视频过滤参数，从源视频生成目标视频。

        此方法用于调用具体的视频处理函数，该函数根据提供的过滤参数列表，源视频路径和目标视频路径进行视频处理。

        参数:
            function (typing.Callable): 要执行的视频处理函数，接收视频处理器、过滤参数、源视频路径和目标视频路径等参数。
            video_filter (list): 视频过滤参数列表，每个参数为字符串形式。
            src (str): 源视频文件路径。
            dst (str): 目标视频文件路径。
            **kwargs: 传递给视频处理函数的额外关键字参数。

        返回:
            tuple[str]: 包含处理结果的元组，通常包括处理日志或其他输出信息。
        """
        return await function(self.fmp, video_filter, src, dst, **kwargs)


class Alynex(object):
    """Alynex"""

    __ks: typing.Optional["KerasStruct"] = KerasStruct()

    def __init__(
            self, matrix: typing.Optional[str], option: "Option", deploy: "Deploy", extent: "typing.Any"
    ):
        self.matrix = matrix
        self.option = option
        self.deploy = deploy
        self.extent = extent

    @property
    def ks(self) -> typing.Optional["KerasStruct"]:
        return self.__ks

    async def ask_model_load(self) -> None:
        """
        异步加载模型到 KerasStruct 实例。

        该方法主要用于加载训练好的模型，以便在后续分析过程中使用。
        根据配置选项选择彩色或灰度模型，并确保模型结构和输入数据的兼容性。

        详细描述:
            1. 确认 `model_place` 路径存在，并且是一个有效的目录。
            2. 确认 `ks` (KerasStruct 实例) 已初始化。
            3. 根据用户选择的模型类型（彩色或灰度），构造模型的最终路径并尝试加载。
            4. 验证模型的输入通道是否与选定的模型类型匹配。
            5. 若加载或验证过程中发生错误，清空 `ks.model` 并抛出 `FramixAnalyzerError`。

        异常处理:
            - OSError: 当模型路径无效或无法访问时抛出。
            - TypeError: 当模型文件类型不正确时抛出。
            - ValueError: 当模型加载过程中发生数据错误时抛出。
            - AssertionError: 当模型通道不匹配或 `ks` 实例未正确初始化时抛出。
            - AttributeError: 当尝试访问未初始化的属性时抛出。
            - FramixAnalyzerError: 当模型路径无效或加载失败时，抛出此异常。
        """
        try:
            if mp := self.matrix:
                assert os.path.isdir(self.option.model_place), f"Invalid Model {mp}"
                assert self.ks, f"First Load KerasStruct()"

                assert os.path.isdir(
                    final_model := os.path.join(
                        mp,
                        mn := self.option.color_model if self.deploy.color else self.option.faint_model
                    )
                ), f"Invalid Model {mn}"
                self.ks.load_model(final_model.format())

                channel = self.ks.model.input_shape[-1]
                if self.deploy.color:
                    assert channel == 3, f"彩色模式需要匹配彩色模型 Model Color Channel={channel}"
                else:
                    assert channel == 1, f"灰度模式需要匹配灰度模型 Model Color Channel={channel}"
        except (OSError, TypeError, ValueError, AssertionError, AttributeError) as e:
            self.ks.model = None
            raise FramixError(e)

    async def ask_video_load(self, vision: str, src_size: tuple) -> "VideoObject":
        """
        加载并处理视频帧信息，返回 VideoObject 对象。

        参数:
            vision (str): 视频文件路径。
            src_size (tuple): 视频原始尺寸。

        返回:
            VideoObject: 包含视频帧信息的对象。

        功能描述:
            1. 创建 VideoObject 对象并记录视频的基本信息（帧长度、帧尺寸）。
            2. 调用 load_frames 方法加载视频帧。
            3. 记录视频帧加载完成后的详细信息和耗时。
        """

        start_time_ = time.time()  # 开始计时

        video = VideoObject(vision)  # 创建 VideoObject 对象

        # 记录视频帧长度和尺寸
        logger.debug(f"{(task_name_ := '视频帧长度: ' f'{video.frame_count}')}")
        logger.debug(f"{(task_info_ := '视频帧尺寸: ' f'{video.frame_size}')}")
        logger.debug(f"{(task_desc_ := '加载视频帧: ' f'{video.name}')}")
        Show.show_panel(self.extent, f"{task_name_}\n{task_info_}\n{task_desc_}", Wind.LOADER)

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
        Show.show_panel(self.extent, f"{task_name}\n{task_info}", Wind.LOADER)

        return video  # 返回 VideoObject 对象

    @staticmethod
    async def ask_frame_grid(vision: str) -> typing.Optional[str]:
        """
        检查视频文件或目录，返回可用的视频文件路径。

        参数:
            vision (str): 视频文件路径或包含视频文件的目录路径。

        返回:
            typing.Optional[str]: 如果存在可用的视频文件，返回视频文件路径，否则返回 None。

        功能描述:
            1. 检查 vision 是否为文件路径，如果是则尝试打开该文件。
            2. 如果 vision 为目录路径，则获取目录中的第一个文件并尝试打开。
            3. 如果视频文件成功打开，返回该文件的路径。
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
        执行视频分析任务，提取并保存视频中的关键帧。

        参数:
            vision (str): 视频文件路径或目录。
            *args: 其他附加参数，其中第一个参数应为保存提取帧的路径。

        返回:
            typing.Optional[str]: 保存提取帧的路径，如果视频文件损坏则返回 None。

        功能描述:
            1. 检查并获取目标视频文件或目录。
            2. 加载视频帧信息。
            3. 使用 VideoCutter 对视频进行分割和压缩。
            4. 获取视频中稳定和不稳定的帧范围。
            5. 提取并保存指定数量的稳定帧。
        """
        if not (target_vision := await self.ask_frame_grid(vision)):
            logger.debug(tip := f"视频文件损坏: {os.path.basename(vision)}")
            return Show.show_panel(self.extent, tip, Wind.KEEPER)

        query_path, src_size, *_ = args

        video = await self.ask_video_load(target_vision, src_size)

        cutter = VideoCutter()
        logger.debug(f"{(cut_name := '视频帧长度: ' f'{video.frame_count}')}")
        logger.debug(f"{(cut_part := '视频帧片段: ' f'{video.frame_count - 1}')}")
        logger.debug(f"{(cut_info := '视频帧尺寸: ' f'{video.frame_size}')}")
        logger.debug(f"{(cut_desc := '压缩视频帧: ' f'{video.name}')}")
        Show.show_panel(self.extent, f"{cut_name}\n{cut_part}\n{cut_info}\n{cut_desc}", Wind.CUTTER)
        cut_start_time = time.time()

        cut_range = cutter.cut(
            video=video, block=self.deploy.block
        )

        logger.debug(f"{(cut_name := '视频帧压缩完成: ' f'{video.name}')}")
        logger.debug(f"{(cut_info := '视频帧压缩耗时: ' f'{time.time() - cut_start_time:.2f} 秒')}")
        Show.show_panel(self.extent, f"{cut_name}\n{cut_info}", Wind.CUTTER)

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

        参数:
            vision (str): 视频文件路径或包含视频文件的目录路径。
            *args: 额外的参数，用于路径和其他配置信息。

        返回:
            typing.Optional["Review"]: 包含分析结果的 Review 对象，如果失败则返回 None。

        功能描述:
            1. 检查并获取有效的视频文件路径。
            2. 加载视频帧信息。
            3. 根据是否有模型加载结果执行相应的视频处理流程。
            4. 根据处理流程返回视频分析结果。
        """

        async def frame_forge(frame: "VideoFrame") -> typing.Union[dict, Exception]:
            """
            保存视频帧为图片文件。

            参数:
                frame: 视频帧对象。

            返回:
                dict: 包含帧ID和图片路径的字典。
            """
            try:
                picture = f"{frame.frame_id}_{format(round(frame.timestamp, 5), '.5f')}.png"
                _, codec = cv2.imencode(".png", frame.data)
                async with aiofiles.open(os.path.join(frame_path, picture), "wb") as f:
                    await f.write(codec.tobytes())
            except Exception as e:
                return e
            return {"id": frame.frame_id, "picture": os.path.join(os.path.basename(frame_path), picture)}

        async def frame_flick() -> tuple:
            """
            提取视频的关键帧信息。

            功能:
                - 提取并记录视频的关键帧（begin_frame 和 final_frame），以及它们的时间戳和帧 ID。
                - 计算关键帧之间的时间成本。

            返回:
                tuple: 包含开始帧ID、结束帧ID和时间成本的元组。
            """
            begin_stage_index, begin_frame_index = self.deploy.begin
            final_stage_index, final_frame_index = self.deploy.final
            # 打印提取关键帧的起始和结束阶段及帧索引
            logger.debug(
                f"{(extract := f'取关键帧: begin={list(self.deploy.begin)} final={list(self.deploy.final)}')}"
            )
            Show.show_panel(self.extent, extract, Wind.FASTER)

            try:
                # 获取视频的阶段信息并打印
                logger.debug(f"{(stage_name := f'阶段划分: {struct.get_ordered_stage_set()}')}")
                Show.show_panel(self.extent, stage_name, Wind.FASTER)
                # 获取视频的不稳定阶段范围
                unstable_stage_range = struct.get_not_stable_stage_range()
                # 获取起始帧和结束帧
                begin_frame = unstable_stage_range[begin_stage_index][begin_frame_index]
                final_frame = unstable_stage_range[final_stage_index][final_frame_index]
            except (AssertionError, IndexError) as e:
                # 捕获异常并记录日志，使用默认的第一个和最后一个重要帧
                logger.debug(e)
                Show.show_panel(self.extent, e, Wind.KEEPER)
                begin_frame = struct.get_important_frame_list()[0]
                final_frame = struct.get_important_frame_list()[-1]

            # 检查是否开始帧的ID小于等于结束帧的ID，若是则使用默认的第一个和最后一个帧
            if final_frame.frame_id <= begin_frame.frame_id:
                logger.debug(tip := f"{final_frame} <= {begin_frame}")
                Show.show_panel(self.extent, tip, Wind.KEEPER)
                begin_frame, end_frame = struct.data[0], struct.data[-1]

            # 计算起始帧和结束帧之间的时间成本
            time_cost = final_frame.timestamp - begin_frame.timestamp

            # 获取起始帧和结束帧的ID和时间戳
            begin_id, begin_ts = begin_frame.frame_id, begin_frame.timestamp
            final_id, final_ts = final_frame.frame_id, final_frame.timestamp

            # 打印开始帧和结束帧的详细信息以及总时间成本
            begin_fr, final_fr = f"{begin_id} - {begin_ts:.5f}", f"{final_id} - {final_ts:.5f}"
            logger.debug(f"开始帧:[{begin_fr}] 结束帧:[{final_fr}] 总耗时:[{(stage_cs := f'{time_cost:.5f}')}]")
            if self.extent == const.SHOW_LEVEL:
                Show.assort_frame(begin_fr, final_fr, stage_cs)

            # 返回关键帧信息和时间成本
            return begin_frame.frame_id, final_frame.frame_id, time_cost

        async def frame_hold() -> list:
            """
            获取并返回视频的所有帧数据。

            功能:
                如果视频帧结构（struct）为空，则直接返回视频的所有帧数据。
                否则，获取视频的所有关键帧，并根据配置参数（boost）决定是否包含非关键帧数据。

            返回:
                list: 包含视频帧对象的列表。

            详细说明:
                - 当视频帧结构（struct）为空时，直接返回视频的所有帧数据。
                - 如果 boost 参数为真，则在获取所有关键帧的基础上，额外包含关键帧之间的非关键帧数据。
                - 使用进度条显示帧处理进度。
            """
            if not struct:
                return [i for i in video.frames_data]

            frames_list = []
            important_frames = struct.get_important_frame_list()
            if self.deploy.boost:
                # 使用进度条显示帧处理进度
                pbar = toolbox.show_progress(total=struct.get_length(), color=50)
                # 将第一个关键帧添加到帧列表中
                frames_list.append(previous := important_frames[0])
                pbar.update(1)
                # 遍历剩余的关键帧
                for current in important_frames[1:]:
                    # 将当前关键帧添加到帧列表中
                    frames_list.append(current)
                    pbar.update(1)
                    # 计算当前帧与前一帧之间的帧差
                    frames_diff = current.frame_id - previous.frame_id
                    # 如果前后帧都不稳定且帧差大于1，则添加中间的关键帧
                    if not previous.is_stable() and not current.is_stable() and frames_diff > 1:
                        for sample in struct.data[previous.frame_id: current.frame_id - 1]:
                            frames_list.append(sample)
                            pbar.update(1)
                    # 更新前一帧为当前帧
                    previous = current
                # 关闭进度条
                pbar.close()
            else:
                for current in toolbox.show_progress(items=struct.data, color=50):
                    frames_list.append(current)

            return frames_list

        async def frame_flow() -> typing.Optional["ClassifierResult"]:
            """
            处理视频帧，包括裁剪和保存。

            返回:
                typing.Optional["ClassifierResult"]: 处理后的视频帧结构数据，没有获取则返回 None。
            """
            cutter = VideoCutter()

            panel_hook_list = []

            size_hook = FrameSizeHook(1.0, None, True)
            cutter.add_hook(size_hook)
            logger.debug(
                f"{(cut_size := f'视频帧处理: {size_hook.__class__.__name__} {[1.0, None, True]}')}"
            )
            panel_hook_list.append(cut_size)

            if len(crop_list := self.deploy.crops) > 0 and sum([j for i in crop_list for j in i.values()]) > 0:
                for crop in crop_list:
                    x, y, x_size, y_size = crop.values()
                    crop_hook = PaintCropHook((y_size, x_size), (y, x))
                    cutter.add_hook(crop_hook)
                    logger.debug(
                        f"{(cut_crop := f'视频帧处理: {crop_hook.__class__.__name__} {x, y, x_size, y_size}')}"
                    )
                    panel_hook_list.append(cut_crop)

            if len(omit_list := self.deploy.omits) > 0 and sum([j for i in omit_list for j in i.values()]) > 0:
                for omit in omit_list:
                    x, y, x_size, y_size = omit.values()
                    omit_hook = PaintOmitHook((y_size, x_size), (y, x))
                    cutter.add_hook(omit_hook)
                    logger.debug(
                        f"{(cut_omit := f'视频帧处理: {omit_hook.__class__.__name__} {x, y, x_size, y_size}')}"
                    )
                    panel_hook_list.append(cut_omit)

            save_hook = FrameSaveHook(extra_path)
            cutter.add_hook(save_hook)
            logger.debug(
                f"{(cut_save := f'视频帧处理: {save_hook.__class__.__name__} {[os.path.basename(extra_path)]}')}"
            )
            panel_hook_list.append(cut_save)

            Show.show_panel(self.extent, "\n".join(panel_hook_list), Wind.CUTTER)

            logger.debug(f"{(cut_name := '视频帧长度: ' f'{video.frame_count}')}")
            logger.debug(f"{(cut_part := '视频帧片段: ' f'{video.frame_count - 1}')}")
            logger.debug(f"{(cut_info := '视频帧尺寸: ' f'{video.frame_size}')}")
            logger.debug(f"{(cut_desc := '压缩视频帧: ' f'{video.name}')}")
            Show.show_panel(self.extent, f"{cut_name}\n{cut_part}\n{cut_info}\n{cut_desc}", Wind.CUTTER)
            cut_start_time = time.time()

            cut_range = cutter.cut(
                video=video, block=self.deploy.block
            )

            logger.debug(f"{(cut_name := '视频帧压缩完成: ' f'{video.name}')}")
            logger.debug(f"{(cut_info := '视频帧压缩耗时: ' f'{time.time() - cut_start_time:.2f} 秒')}")
            Show.show_panel(self.extent, f"{cut_name}\n{cut_info}", Wind.CUTTER)

            stable, unstable = cut_range.get_range(
                threshold=self.deploy.thres, offset=self.deploy.shift
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

            for draw in toolbox.show_progress(items=os.listdir(extra_path), color=146):
                toolbox.draw_line(os.path.join(extra_path, draw).format())

            try:
                struct_data = self.ks.classify(
                    video=video, valid_range=stable, keep_data=True
                )
            except AssertionError as e:
                logger.debug(e)
                return Show.show_panel(self.extent, e, Wind.KEEPER)

            return struct_data

        async def analytics_basic() -> tuple:
            """
            执行基础视频分析，保存帧图片并计算时间成本。

            返回:
                tuple: 包含开始帧ID、结束帧ID、时间成本、评分和结构数据的元组。
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
                        Show.show_panel(self.extent, r, Wind.KEEPER)
                    else:
                        scores[r["id"]] = r["picture"]

            begin_frame, final_frame = frames[0], frames[-1]
            time_cost = final_frame.timestamp - begin_frame.timestamp
            return begin_frame.frame_id, final_frame.frame_id, time_cost, scores, None

        async def analytics_keras() -> tuple:
            """
            执行基于Keras模型的视频分析，保存帧图片并计算时间成本。

            返回:
                tuple: 包含开始帧ID、结束帧ID、时间成本、评分和结构数据的元组。
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
                        Show.show_panel(self.extent, r, Wind.KEEPER)
                    else:
                        scores[r["id"]] = r["picture"]

            begin_frame_id, final_frame_id, time_cost = await flick_tasks
            return begin_frame_id, final_frame_id, time_cost, scores, struct

        # 检查并获取有效的视频文件路径
        if not (target_vision := await self.ask_frame_grid(vision)):
            logger.debug(tip_ := f"视频文件损坏: {os.path.basename(vision)}")
            return Show.show_panel(self.extent, tip_, Wind.KEEPER)

        # 解包额外的参数
        frame_path, extra_path, src_size, *_ = args

        # 加载视频帧信息
        video = await self.ask_video_load(target_vision, src_size)

        # 根据是否有模型加载结果执行相应的视频处理流程
        struct = await frame_flow() if self.ks.model else None

        # 获取帧数据
        frames = await frame_hold()

        # 根据处理流程返回视频分析结果
        if struct:
            return Review(*(await analytics_keras()))
        return Review(*(await analytics_basic()))


# 任务处理器
async def arithmetic(function: "typing.Callable", parameters: list[str]) -> None:
    """
    异步执行函数，并处理参数路径修正。

    参数:
        function (typing.Callable): 要执行的函数。
        parameters (list[str]): 参数列表。
    """

    # 修正参数路径
    parameters = [(await Craft.revise_path(param)) for param in parameters]
    # 去除重复参数
    parameters = list(dict.fromkeys(parameters))
    # 执行函数
    await function(parameters, _option, _deploy)


# 任务处理器
async def scheduling() -> None:
    """
    根据命令行参数调度并执行相应的任务。

    任务包括:
        - 视频分析
        - 图像绘制
        - 合并视图
        - 合并文件
    """

    async def _already_installed():
        # 检查 scrcpy 是否已安装，如果未安装则显示安装提示并退出程序
        if shutil.which("scrcpy"):
            return None
        raise FramixError("Install first https://github.com/Genymobile/scrcpy")

    if _lines.flick or _lines.carry or _lines.fully:
        await _already_installed()
        await _missions.analysis(_option, _deploy)

    elif _lines.paint:
        await _missions.painting(_option, _deploy)

    elif _lines.union:
        await _missions.combine_view(_lines.union)

    elif _lines.merge:
        await _missions.combine_main(_lines.merge)

    else:
        Show.help_document()


# 信号处理器
def signal_processor(*_, **__) -> None:
    """
    处理信号，用于在特定情况下触发应用程序的退出。

    参数:
        *_: 接受并忽略所有传入的位置参数。
        **__: 接受并忽略所有传入的关键字参数。

    返回值:
        None: 函数不会返回任何值，因为在 `sys.exit()` 调用后，程序会终止执行。

    注意:
        - 此函数的主要功能是接受外部传入的信号并在适当的条件下终止程序的运行。
        - `_` 和 `__` 的命名表示这些参数未被使用。
    """
    Show.exit()
    sys.exit(Show.closure())


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
            Show.minor_logo()
            Show.help_document()
            Show.done()
            sys.exit(Show.closure())

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
            _fx_feasible = os.path.dirname(_fx_work)
        else:
            raise FramixError(f"{const.DESC} compatible with {const.NAME} command")

        # 设置模板文件路径
        _atom_total_temp = os.path.join(_fx_work, const.F_SCHEMATIC, "templates", "template_atom_total.html")
        _main_share_temp = os.path.join(_fx_work, const.F_SCHEMATIC, "templates", "template_main_share.html")
        _main_total_temp = os.path.join(_fx_work, const.F_SCHEMATIC, "templates", "template_main_total.html")
        _view_share_temp = os.path.join(_fx_work, const.F_SCHEMATIC, "templates", "template_view_share.html")
        _view_total_temp = os.path.join(_fx_work, const.F_SCHEMATIC, "templates", "template_view_total.html")

        # 检查每个模板文件是否存在，如果缺失则显示错误信息并退出程序
        for _tmp in (_temps := [_atom_total_temp, _main_share_temp, _main_total_temp, _view_share_temp, _view_total_temp]):
            if os.path.isfile(_tmp) and os.path.basename(_tmp).endswith(".html"):
                continue
            _tmp_name = os.path.basename(_tmp)
            raise FramixError(f"{const.DESC} missing files {_tmp_name}")

        _turbo = os.path.join(_fx_work, const.F_SCHEMATIC, "supports").format()  # 设置工具源路径

        # 根据平台设置工具路径
        if _platform == "win32":
            # Windows
            _adb = os.path.join(_turbo, "Windows", "platform-tools", "adb.exe")
            _fmp = os.path.join(_turbo, "Windows", "ffmpeg", "bin", "ffmpeg.exe")
            _fpb = os.path.join(_turbo, "Windows", "ffmpeg", "bin", "ffprobe.exe")
        elif _platform == "darwin":
            # MacOS
            _adb = os.path.join(_turbo, "MacOS", "platform-tools", "adb")
            _fmp = os.path.join(_turbo, "MacOS", "ffmpeg", "bin", "ffmpeg")
            _fpb = os.path.join(_turbo, "MacOS", "ffmpeg", "bin", "ffprobe")
        else:
            raise FramixError(f"{const.DESC} tool compatible with Windows or MacOS")

        # 将工具路径添加到系统 PATH 环境变量中
        for _tls in (_tools := [_adb, _fmp, _fpb]):
            os.environ["PATH"] = os.path.dirname(_tls) + _env_symbol + os.environ.get("PATH", "")

        # 检查每个工具是否存在，如果缺失则显示错误信息并退出程序
        for _tls in _tools:
            if not shutil.which((_tls_name := os.path.basename(_tls))):
                raise FramixError(f"{const.DESC} missing files {_tls_name}")

        # 设置初始路径
        if not os.path.exists(
                _initial_source := os.path.join(_fx_feasible, const.F_SPECIALLY).format()
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

        freeze_support()  # 在 Windows 平台上启动多进程时确保冻结的可执行文件可以正确运行。

        """
        **命令行参数解析器解析命令行参数**
        
        注意:
            此代码块必须在 `__main__` 块下调用，否则可能会导致多进程模块无法正确加载。
        """
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
            # 如果命令行中包含配置参数，无论是否存在配置文件，都将覆盖配置文件，以命令行参数为第一优先级
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
            这样做的目的是避免在多进程环境中向子进程传递不可序列化的对象，因为这些对象在传递过程中可能会导致 `pickle.PicklingError` 错误。
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

        _missions = Missions(_wires, _level, _power, *_positions, **_keywords)  # 初始化主要任务对象

        Show.minor_logo()  # 显示应用程序标志
        Show.load_animation()  # 显示应用程序加载动画

        """
        **创建主事件循环**
        
        注意: 
            该事件循环对象 `_main_loop` 是不可序列化的，因此不能将其传递给子进程。
            在需要使用事件循环的类实例化或函数调用时，应当在子进程内创建新的事件循环。
        """
        _main_loop: "asyncio.AbstractEventLoop" = asyncio.get_event_loop()

        if _video_list := _lines.video:
            _main_loop.run_until_complete(
                arithmetic(_missions.video_file_task, _video_list)
            )

        elif _stack_list := _lines.stack:
            _main_loop.run_until_complete(
                arithmetic(_missions.video_data_task, _stack_list)
            )

        elif _train_list := _lines.train:
            _main_loop.run_until_complete(
                arithmetic(_missions.train_model, _train_list)
            )

        elif _build_list := _lines.build:
            _main_loop.run_until_complete(
                arithmetic(_missions.build_model, _build_list)
            )

        else:
            from engine.manage import (
                ScreenMonitor, SourceMonitor, Manage
            )
            from engine.medias import (
                Record, Player
            )

            _main_loop.run_until_complete(scheduling())

    except KeyboardInterrupt:
        Show.exit()
    except (OSError, RuntimeError, MemoryError, FramixError):
        Show.console.print_exception()
        Show.fail()
    else:
        Show.done()
    finally:
        sys.exit(Show.closure())
