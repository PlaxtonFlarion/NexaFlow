__all__ = []

import os
import sys
import shutil
from loguru import logger
from frameflow.skills.show import Show
from nexaflow import const

_platform = sys.platform.strip().lower()
_software = os.path.basename(os.path.abspath(sys.argv[0])).strip().lower()
_sys_symbol = os.sep
_env_symbol = os.path.pathsep

if _software == f"{const.NAME}.exe":
    _workable = os.path.dirname(os.path.abspath(sys.argv[0]))
    _feasible = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
elif _software == f"{const.NAME}.bin":
    _workable = os.path.dirname(sys.executable)
    _feasible = os.path.dirname(os.path.dirname(sys.executable))
elif _software == f"{const.NAME}":
    _workable = os.path.dirname(sys.executable)
    _feasible = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))))
elif _software == f"{const.NAME}.py":
    _workable = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _feasible = os.path.dirname(os.path.abspath(__file__))
else:
    logger.error(f"{const.ERR}Software compatible with {const.NAME}[/]")
    Show.simulation_progress(f"Exit after 5 seconds ...", 1, 0.05)
    sys.exit(Show.fail())

_turbo = os.path.join(_workable, "archivix", "tools")

if _platform == "win32":
    _adb = os.path.join(_turbo, "win", "platform-tools", "adb.exe")
    _fmp = os.path.join(_turbo, "win", "ffmpeg", "bin", "ffmpeg.exe")
    _fpb = os.path.join(_turbo, "win", "ffmpeg", "bin", "ffprobe.exe")
    _scc = os.path.join(_turbo, "win", "scrcpy", "scrcpy.exe")
elif _platform == "darwin":
    _adb = os.path.join(_turbo, "mac", "platform-tools", "adb")
    _fmp = os.path.join(_turbo, "mac", "ffmpeg", "bin", "ffmpeg")
    _fpb = os.path.join(_turbo, "mac", "ffmpeg", "bin", "ffprobe")
    _scc = os.path.join(_turbo, "mac", "scrcpy", "bin", "scrcpy")
else:
    logger.error(f"{const.ERR}{const.NAME} compatible with {const.ERR}Win & Mac[/]")
    Show.simulation_progress(f"Exit after 5 seconds ...", 1, 0.05)
    sys.exit(Show.fail())

for _tls in (_tools := [_adb, _fmp, _fpb, _scc]):
    os.environ["PATH"] = os.path.dirname(_tls) + _env_symbol + os.environ.get("PATH", "")

for _tls in _tools:
    if not shutil.which((_tls_name := os.path.basename(_tls))):
        logger.error(f"{const.ERR}{const.NAME} missing files {_tls_name}[/]")
        Show.simulation_progress(f"Exit after 5 seconds ...", 1, 0.05)
        sys.exit(Show.fail())

_atom_total_temp = os.path.join(_workable, "archivix", "pages", "template_atom_total.html")
_main_share_temp = os.path.join(_workable, "archivix", "pages", "template_main_share.html")
_main_total_temp = os.path.join(_workable, "archivix", "pages", "template_main_total.html")
_view_share_temp = os.path.join(_workable, "archivix", "pages", "template_view_share.html")
_view_total_temp = os.path.join(_workable, "archivix", "pages", "template_view_total.html")

for _tmp in (_temps := [_atom_total_temp, _main_share_temp, _main_total_temp, _view_share_temp, _view_total_temp]):
    if not os.path.isfile(_tmp):
        _tmp_name = os.path.basename(_tmp)
        logger.error(f"{const.ERR}{const.NAME} missing files {_tmp_name}[/]")
        Show.simulation_progress(f"Exit after 5 seconds ...", 1, 0.05)
        sys.exit(Show.fail())

_initial_source = os.path.join(_feasible, f"{const.NAME}.source")

_total_place = os.path.join(_feasible, f"{const.NAME}.report")
_model_place = os.path.join(_workable, "archivix", "molds", "Keras_Gray_W256_H256_00000")

if len(sys.argv) == 1:
    Show.help_document()
    sys.exit(Show.done())

_wires = sys.argv[1:]

try:
    import re
    import cv2
    import json
    import time
    import numpy
    import random
    import typing
    import signal
    import inspect
    import asyncio
    import tempfile
    import aiofiles
    import datetime
    from pathlib import Path
    from rich.prompt import Prompt
    from engine.manage import Manage
    from engine.switch import Switch
    from engine.terminal import Terminal
    from engine.flight import Find
    from engine.flight import Craft
    from engine.flight import Active
    from engine.flight import Review
    from engine.flight import FramixAnalysisError
    from engine.flight import FramixAnalyzerError
    from engine.flight import FramixReporterError
    from frameflow.argument import Wind
    from frameflow.skills.brexil import Option
    from frameflow.skills.brexil import Deploy
    from frameflow.skills.brexil import Script
    from frameflow.skills.drovix import Drovix
    from frameflow.skills.parser import Parser
    from nexaflow import toolbox
    from nexaflow.report import Report
    from nexaflow.video import VideoObject, VideoFrame
    from nexaflow.cutter.cutter import VideoCutter
    from nexaflow.hook import FrameSizeHook, FrameSaveHook
    from nexaflow.hook import PaintCropHook, PaintOmitHook
    from nexaflow.classifier.keras_classifier import KerasStruct
except (ImportError, ModuleNotFoundError):
    Show.console.print_exception()
    sys.exit(Show.fail())


class Missions(object):

    def __init__(self, level, power, lines, **kwargs):
        self.level = level
        self.power = power
        self.lines = lines
        self.atom_total_temp = kwargs["atom_total_temp"]
        self.main_share_temp = kwargs["main_share_temp"]
        self.main_total_temp = kwargs["main_total_temp"]
        self.view_share_temp = kwargs["view_share_temp"]
        self.view_total_temp = kwargs["view_total_temp"]
        self.initial_option = kwargs["initial_option"]
        self.initial_deploy = kwargs["initial_deploy"]
        self.initial_script = kwargs["initial_script"]
        self.total_place = kwargs["total_place"]
        self.model_place = kwargs["model_place"]
        self.adb = kwargs["adb"]
        self.fmp = kwargs["fmp"]
        self.fpb = kwargs["fpb"]
        self.scc = kwargs["scc"]

    # """Child Process"""
    def amazing(self, vision: str, *args, **kwargs):
        # Initial Loop
        loop = asyncio.get_event_loop()
        # Initial Alynex
        model_place = self.model_place if self.lines.keras else None
        alynex = Alynex(self.level, model_place, **kwargs)
        try:
            loop.run_until_complete(alynex.ask_model_load())
            loop.run_until_complete(alynex.ask_model_walk())
        except FramixAnalyzerError:
            pass
        loop_complete = loop.run_until_complete(
            alynex.ask_analyzer(vision, *args)
        )
        return loop_complete

    # """Child Process"""
    def bizarre(self, vision: str, *args, **kwargs):
        # Initial Loop
        loop = asyncio.get_event_loop()
        # Initial Alynex
        model_place = None
        alynex = Alynex(self.level, model_place, **kwargs)

        loop_complete = loop.run_until_complete(
            alynex.ask_exercise(vision, *args)
        )
        return loop_complete

    @staticmethod
    async def enforce(db: "Drovix", ks: typing.Optional["KerasStruct"], start: int, end: int, cost: float, *args):
        basic_columns = ["total_path", "title", "query_path", "query", "stage", "frame_path"]
        stage = json.dumps({"stage": {"start": start, "end": end, "cost": cost}})
        value = list(args[:4]) + [stage, args[4]]

        if ks:
            extra_columns = ["extra_path", "proto_path"]
            extra_data = args[5:7]
            column_list = basic_columns + extra_columns
            value.extend(extra_data)
        else:
            column_list = basic_columns

        await db.create("stocks", *column_list)
        await db.insert("stocks", column_list, tuple(value))

    async def als_track(
            self,
            deploy: "Deploy",
            clipix: "Clipix",
            task_list: list[list],
            main_loop: "asyncio.AbstractEventLoop"
    ) -> tuple[list, list]:
        """
        异步执行视频的处理追踪，包括内容提取和平衡视频长度等功能。

        此函数用于根据指定的部署配置，提取视频内容，并尝试将多个视频的长度调整为一致。
        主要处理包括解析视频信息、视频内容提取、时间平衡和删除临时文件。

        参数:
            deploy (Deploy): 配置信息对象，包含视频处理的起始、结束、限制时间和帧率等。
            clipix (Clipix): 视频处理工具对象，负责具体的视频内容提取和平衡操作。
            task_list (list[list]): 包含视频信息的任务列表，每个列表项包括视频模板和其他参数。
            main_loop (asyncio.AbstractEventLoop): 异步事件循环，用于执行非异步的任务。

        返回:
            tuple[list, list]: 返回处理后的原始视频列表和指示信息列表。

        注意:
            - 该函数为异步函数，需要在异步环境中运行。
            - 函数内部使用了多个异步gather来并行处理视频操作，提高效率。
            - 确保提供的每个视频都符合`Deploy`中定义的处理标准。
            - 异常处理：确保处理过程中捕获并妥善处理可能发生的任何异常，以避免程序中断。
        """

        # Video information
        start_ms = Parser.parse_mills(deploy.start)
        close_ms = Parser.parse_mills(deploy.close)
        limit_ms = Parser.parse_mills(deploy.limit)
        content_list = await asyncio.gather(
            *(clipix.vision_content(video_temp, start_ms, close_ms, limit_ms)
              for video_temp, *_ in task_list)
        )
        for (rlt, avg, dur, org, pnt), (video_temp, *_) in zip(content_list, task_list):
            vd_start, vd_close, vd_limit = pnt
            logger.debug(f"视频尺寸: {list(org)}")
            logger.debug(f"实际帧率: [{rlt}] 平均帧率: [{avg}] 转换帧率: [{deploy.frate}]")
            logger.debug(f"视频时长: [{dur:.6f}] [{Parser.parse_times(dur)}]")
            logger.debug(f"视频剪辑: start=[{vd_start}] close=[{vd_close}] limit=[{vd_limit}]")
            if self.level == "INFO":
                Show.content_pose(
                    rlt, avg, f"{dur:.6f}", org, vd_start, vd_close, vd_limit, video_temp, deploy.frate
                )
        *_, durations, originals, indicates = zip(*content_list)

        # Video Balance
        eliminate = []
        if self.lines.alike and len(task_list) > 1:
            logger.debug(tip := f"平衡时间: [{(standard := min(durations)):.6f}] [{Parser.parse_times(standard)}]")
            Show.show_panel(self.level, tip, Wind.STANDARD)
            video_dst_list = await asyncio.gather(
                *(clipix.vision_balance(duration, standard, video_src, deploy.frate)
                  for duration, (video_src, *_) in zip(durations, task_list))
            )
            for (video_idx, (video_dst, video_blc)), (video_src, *_) in zip(enumerate(video_dst_list), task_list):
                logger.debug(f"{video_blc}")
                Show.show_panel(self.level, video_blc, Wind.TAILOR)
                eliminate.append(main_loop.run_in_executor(None, os.remove, video_src))
                task_list[video_idx][0] = video_dst
        await asyncio.gather(*eliminate, return_exceptions=True)

        return originals, indicates

    async def als_speed(
            self,
            deploy: "Deploy",
            clipix: "Clipix",
            report: "Report",
            task_list: list[list],
            originals: list,
            indicates: list
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
        """

        Show.show_panel(self.level, Wind.SPEED_TEXT, Wind.SPEED)

        const_filter = [f"fps={deploy.frate}"] if deploy.color else [f"fps={deploy.frate}", "format=gray"]
        if deploy.shape:
            final_shape_list = await clipix.vision_improve(
                originals, deploy.shape
            )
            video_filter_list = [
                const_filter + [f"scale={w}:{h}"] for w, h, ratio in final_shape_list
            ]
        elif deploy.scale:
            scale = max(0.1, min(1.0, deploy.scale))
            video_filter_list = [
                const_filter + [f"scale=iw*{scale}:ih*{scale}"] for _ in task_list
            ]
        else:
            scale = const.COMPRESS
            video_filter_list = [
                const_filter + [f"scale=iw*{scale}:ih*{scale}"] for _ in task_list
            ]

        for flt, (video_temp, *_) in zip(video_filter_list, task_list):
            logger.debug(tip := f"视频过滤: {flt} {os.path.basename(video_temp)}")
            Show.show_panel(self.level, tip, Wind.FILTER)

        detach_result = await asyncio.gather(
            *(clipix.pixels(
                Switch.ask_video_detach, video_filter, video_temp, frame_path,
                start=vision_start, close=vision_close, limit=vision_limit
            ) for video_filter, (video_temp, *_, frame_path, _, _), (vision_start, vision_close, vision_limit)
                in zip(video_filter_list, task_list, indicates))
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
            Show.show_panel(self.level, "\n".join(message_list), Wind.METRIC)

        async def render_speed(todo_list: list[list]):
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
            logger.debug(f"Speeder: {json.dumps(result, ensure_ascii=False)}")
            await report.load(result)

            future_result = struct, start, end, cost
            todo_list_result = total_path, title, query_path, query, frame_path

            return future_result, todo_list_result

        # Speed Analyzer Result
        render_result = await asyncio.gather(
            *(render_speed(todo_list) for todo_list in task_list)
        )

        async with Drovix(os.path.join(report.reset_path, f"{const.NAME}_data.db")) as db:
            await asyncio.gather(
                *(self.enforce(db, *ftr, *tlr) for ftr, tlr in render_result)
            )

    async def als_keras(
            self,
            deploy: "Deploy",
            clipix: "Clipix",
            report: "Report",
            task_list: list[list],
            originals: list,
            indicates: list,
            main_loop: "asyncio.AbstractEventLoop",
            alynex: "Alynex"
    ) -> None:
        """
        异步执行视频的Keras模式分析或基本模式分析，包括视频过滤、尺寸调整和动态模板渲染等功能。

        此函数根据部署配置(deploy)调整视频帧率和尺寸，执行视频分析，并根据分析结果采用不同模式处理视频。
        如果启用了Keras模型，执行深度学习模型分析；否则执行基本分析。

        参数:
            deploy (Deploy): 配置信息对象，包含视频处理的帧率、颜色格式、尺寸等配置。
            clipix (Clipix): 视频处理工具对象，负责具体的视频内容调整和分析操作。
            report (Report): 报告处理对象，负责记录和展示处理结果。
            task_list (list[list]): 包含视频和其他相关参数的任务列表。
            originals (list): 原始视频列表，用于提取和处理视频内容。
            indicates (list): 指示信息列表，包含视频处理的具体指标和参数。
            main_loop (asyncio.AbstractEventLoop): 异步事件循环，用于执行非异步的任务。
            alynex (Alynex): 模型分析工具，决定使用Keras模型还是基础分析。

        注意:
            - 该函数为异步函数，需要在异步环境中运行。
            - 函数内部使用了多个异步gather来并行处理视频操作，提高效率。
            - 函数的执行路径依赖于`alynex.ks.model`的状态，确保Alynex实例正确初始化。
            - 异常处理：确保处理过程中捕获并妥善处理可能发生的任何异常，以避免程序中断。
        """

        Show.show_panel(
            self.level,
            Wind.KERAS_TEXT if alynex.ks.model else Wind.BASIC_TEXT,
            Wind.KERAS if alynex.ks.model else Wind.BASIC
        )

        video_target_list = [
            (os.path.join(
                os.path.dirname(video_temp), f"vision_fps{deploy.frate}_{random.randint(100, 999)}.mp4"
            ), [f"fps={deploy.frate}"]) for video_temp, *_ in task_list
        ]

        for (tar, flt), (video_temp, *_) in zip(video_target_list, task_list):
            logger.debug(tip := f"视频过滤: {flt} {os.path.basename(video_temp)}")
            Show.show_panel(self.level, tip, Wind.FILTER)

        change_result = await asyncio.gather(
            *(clipix.pixels(
                Switch.ask_video_change, video_filter, video_temp,
                target, start=vision_start, close=vision_close, limit=vision_limit
            ) for (target, video_filter), (video_temp, *_), (vision_start, vision_close, vision_limit)
                in zip(video_target_list, task_list, indicates))
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
            Show.show_panel(self.level, "\n".join(message_list), Wind.METRIC)
            eliminate.append(
                main_loop.run_in_executor(None, os.remove, video_temp)
            )
        await asyncio.gather(*eliminate, return_exceptions=True)

        if alynex.ks.model:
            deploy.view_deploy()

        # Ask Analyzer
        if len(task_list) == 1:
            task = [
                alynex.ask_analyzer(target, frame_path, extra_path, original)
                for (target, _), (*_, frame_path, extra_path, _), original
                in zip(video_target_list, task_list, originals)
            ]
            futures = await asyncio.gather(*task)

        else:
            this_level = self.level
            self.level = "ERROR"
            func = partial(self.amazing, **deploy.deploys)
            with ProcessPoolExecutor(self.power, None, Active.active, ("ERROR",)) as exe:
                task = [
                    main_loop.run_in_executor(exe, func, target, frame_path, extra_path, original)
                    for (target, _), (*_, frame_path, extra_path, _), original
                    in zip(video_target_list, task_list, originals)
                ]
                futures = await asyncio.gather(*task)
            self.level = this_level

        # Template
        if isinstance(atom_tmp := await Craft.achieve(self.atom_total_temp), Exception):
            logger.debug(tip := f"{atom_tmp}")
            return Show.show_panel(self.level, tip, Wind.KEEPER)

        async def render_keras(future: "Review", todo_list: list[list]):
            start, end, cost, scores, struct = future.material
            *_, total_path, title, query_path, query, frame_path, extra_path, proto_path = todo_list

            result = {
                "total": os.path.basename(total_path),
                "title": title,
                "query": query,
                "stage": {"start": start, "end": end, "cost": f"{cost:.5f}"},
                "frame": os.path.basename(frame_path)
            }

            stages_inform = ""
            if struct:
                logger.debug(f"模版引擎正在渲染 ...")
                stages_inform = await report.ask_draw(
                    scores, struct, proto_path, atom_tmp, deploy.boost
                )
                logger.debug(f"模版引擎渲染完毕 {os.path.basename(stages_inform)}")
                result["extra"] = os.path.basename(extra_path)
                result["proto"] = os.path.basename(stages_inform)
                result["style"] = "keras"
            else:
                result["style"] = "basic"

            logger.debug(f"Restore: {json.dumps(result, ensure_ascii=False)}")
            await report.load(result)

            future_result = struct, start, end, cost
            todo_list_result = total_path, title, query_path, query, frame_path, extra_path, proto_path

            return future_result, todo_list_result, os.path.basename(stages_inform)

        # Keras Analyzer Result
        render_result = await asyncio.gather(
            *(render_keras(future, todo_list) for future, todo_list in zip(futures, task_list) if future)
        )

        if alynex.ks.model:
            Show.show_panel(self.level, f"模版引擎正在渲染 ...", Wind.REPORTER)
            rendering_list = [f"模版引擎渲染完成 {rd}" for *_, rd in render_result]
            Show.show_panel(self.level, "\n".join(rendering_list), Wind.REPORTER)

        async with Drovix(os.path.join(report.reset_path, f"{const.NAME}_data.db")) as db:
            await asyncio.gather(
                *(self.enforce(db, *ftr, *tlr) for ftr, tlr, _ in render_result)
            )

    async def combine(self, report: "Report"):
        if len(report.range_list) == 0:
            logger.debug(tip := f"没有可以生成的报告")
            return Show.show_panel(self.level, tip, Wind.KEEPER)
        function = getattr(self, "combine_view" if self.lines.speed else "combine_main")
        return await function([os.path.dirname(report.total_path)])

    # 时空纽带分析系统
    async def combine_view(self, merge: list):
        views, total = await asyncio.gather(
            Craft.achieve(self.view_share_temp), Craft.achieve(self.view_total_temp),
            return_exceptions=True
        )

        logger.debug(tip := f"正在生成汇总报告 ...")
        Show.show_panel(self.level, tip, Wind.REPORTER)
        state_list = await asyncio.gather(
            *(Report.ask_create_total_report(m, self.lines.group, views, total) for m in merge)
        )

        efficient_state_list = []
        for state in state_list:
            if isinstance(state, Exception):
                logger.debug(tip := f"{state}")
                Show.show_panel(self.level, tip, Wind.KEEPER)
            logger.debug(tip := f"成功生成汇总报告 {os.path.relpath(state)}")
            efficient_state_list.append(tip)
        Show.show_panel(self.level, "\n".join(efficient_state_list), Wind.REPORTER)

    # 时序融合分析系统
    async def combine_main(self, merge: list):
        major, total = await asyncio.gather(
            Craft.achieve(self.main_share_temp), Craft.achieve(self.main_total_temp),
            return_exceptions=True
        )

        logger.debug(tip := f"正在生成汇总报告 ...")
        Show.show_panel(self.level, tip, Wind.REPORTER)
        state_list = await asyncio.gather(
            *(Report.ask_create_total_report(m, self.lines.group, major, total) for m in merge)
        )

        efficient_state_list = []
        for state in state_list:
            if isinstance(state, Exception):
                logger.debug(tip := f"{state}")
                Show.show_panel(self.level, tip, Wind.KEEPER)
            logger.debug(tip := f"成功生成汇总报告 {os.path.relpath(state)}")
            efficient_state_list.append(tip)
        Show.show_panel(self.level, "\n".join(efficient_state_list), Wind.REPORTER)

    # 视频解析探索
    async def video_file_task(self, video_file_list: list, *args):
        if len(video_file_list := [
            video_file for video_file in video_file_list if os.path.isfile(video_file)
        ]) == 0:
            logger.debug(tip := f"没有有效任务")
            return Show.show_panel(self.level, tip, Wind.KEEPER)

        # Receive Argument
        platform, deploy, main_loop = args

        clipix = Clipix(self.fmp, self.fpb)
        report = Report(self.total_place)
        report.title = f"{const.DESC}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

        # Profession
        task_list = []
        for video_file in video_file_list:
            report.query = f"{os.path.basename(video_file).split('.')[0]}_{time.strftime('%Y%m%d%H%M%S')}"
            new_video_path = os.path.join(report.video_path, os.path.basename(video_file))
            shutil.copy(video_file, new_video_path)
            task_list.append(
                [new_video_path, None, report.total_path, report.title, report.query_path,
                 report.query, report.frame_path, report.extra_path, report.proto_path]
            )

        # Information
        originals, indicates = await self.als_track(deploy, clipix, task_list, main_loop)

        # Pack Argument
        attack = deploy, clipix, report, task_list, originals, indicates

        if self.lines.speed:
            # Speed Analyzer
            await self.als_speed(*attack)
        else:
            # Initial Alynex
            model_place = self.model_place if self.lines.keras else None
            alynex = Alynex(self.level, model_place, **deploy.deploys)
            try:
                await alynex.ask_model_load()
                await alynex.ask_model_walk()
            except FramixAnalyzerError as e:
                logger.debug(e)
                Show.show_panel(self.level, e, Wind.KEEPER)

            charge = main_loop, alynex

            # Keras Analyzer
            await self.als_keras(*attack, *charge)

        # Create Report
        await self.combine(report)

    # 影像堆叠导航
    async def video_data_task(self, video_data_list: list, *args):

        async def load_entries():
            for video_data in video_data_list:
                finder_result = finder.accelerate(video_data)
                if isinstance(finder_result, Exception):
                    logger.debug(finder_result)
                    Show.show_panel(self.level, finder_result, Wind.KEEPER)
                    continue
                tree, collection_list = finder_result
                Show.console.print(tree)
                yield collection_list[0]

        # Receive Argument
        platform, deploy, main_loop = args

        finder = Find()
        clipix = Clipix(self.fmp, self.fpb)

        # Profession
        async for entries in load_entries():
            if entries:
                report = Report(self.total_place)
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

                    # Information
                    originals, indicates = await self.als_track(deploy, clipix, task_list, main_loop)

                    # Pack Argument
                    attack = deploy, clipix, report, task_list, originals, indicates

                    if self.lines.speed:
                        # Speed Analyzer
                        await self.als_speed(*attack)
                    else:
                        # Initial Alynex
                        model_place = self.model_place if self.lines.keras else None
                        alynex = Alynex(self.level, model_place, **deploy.deploys)
                        try:
                            await alynex.ask_model_load()
                            await alynex.ask_model_walk()
                        except FramixAnalyzerError as e:
                            logger.debug(e)
                            Show.show_panel(self.level, e, Wind.KEEPER)

                        charge = main_loop, alynex

                        # Keras Analyzer
                        await self.als_keras(*attack, *charge)

                # Create Report
                await self.combine(report)

    # 模型训练大师
    async def train_model(self, video_file_list: list, *args):
        if len(video_file_list := [
            video_file for video_file in video_file_list if os.path.isfile(video_file)
        ]) == 0:
            logger.debug(tip := f"没有有效任务")
            return Show.show_panel(self.level, tip, Wind.KEEPER)

        import uuid

        # Receive Argument
        platform, deploy, main_loop = args

        clipix = Clipix(self.fmp, self.fpb)
        report = Report(self.total_place)

        # Profession
        task_list = []
        for video_file in video_file_list:
            report.title = f"Model_{uuid.uuid4()}"
            new_video_path = os.path.join(report.video_path, os.path.basename(video_file))
            shutil.copy(video_file, new_video_path)
            task_list.append(
                [new_video_path, None, report.total_path, report.title, report.query_path,
                 report.query, report.frame_path, report.extra_path, report.proto_path]
            )

        # Information
        originals, indicates = await self.als_track(deploy, clipix, task_list, main_loop)

        video_target_list = [
            (os.path.join(
                report.query_path, f"tmp_fps{deploy.frate}_{random.randint(10000, 99999)}.mp4"
            ), [f"fps={deploy.frate}"]) for video_temp, *_ in task_list
        ]

        for (tar, flt), (video_temp, *_) in zip(video_target_list, task_list):
            logger.debug(tip := f"视频过滤: {flt} {os.path.basename(video_temp)}")
            Show.show_panel(self.level, tip, Wind.FILTER)

        change_result = await asyncio.gather(
            *(clipix.pixels(
                Switch.ask_video_change, video_filter, video_temp,
                target, start=vision_start, close=vision_close, limit=vision_limit
            ) for (target, video_filter), (video_temp, *_), (vision_start, vision_close, vision_limit)
                in zip(video_target_list, task_list, indicates))
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
            Show.show_panel(self.level, "\n".join(message_list), Wind.METRIC)
            eliminate.append(
                main_loop.run_in_executor(None, os.remove, video_temp)
            )
        await asyncio.gather(*eliminate, return_exceptions=True)

        # Initial Alynex
        model_place = None
        alynex = Alynex(self.level, model_place, **deploy.deploys)

        # Ask Analyzer
        if len(task_list) == 1:
            task = [
                alynex.ask_exercise(target, query_path, original)
                for (target, _), (_, _, _, _, query_path, *_), original
                in zip(video_target_list, task_list, originals)
            ]
            futures = await asyncio.gather(*task)

        else:
            this_level = self.level
            self.level = "ERROR"
            func = partial(self.bizarre, **deploy.deploys)
            with ProcessPoolExecutor(self.power, None, Active.active, ("ERROR",)) as exe:
                task = [
                    main_loop.run_in_executor(exe, func, target, query_path, original)
                    for (target, _), (_, _, _, _, query_path, *_), original
                    in zip(video_target_list, task_list, originals)
                ]
                futures = await asyncio.gather(*task)
            self.level = this_level

        pick_info_list = []
        for future in futures:
            logger.debug(tip := f"保存: {os.path.basename(future)}")
            pick_info_list.append(tip)
        Show.show_panel(self.level, "\n".join(pick_info_list), Wind.PROVIDER)

        await asyncio.gather(
            *(main_loop.run_in_executor(None, os.remove, target) for (target, _) in video_target_list)
        )

    # 模型编译大师
    async def build_model(self, video_data_list: list, *args):
        if len(video_data_list := [
            video_data for video_data in video_data_list if os.path.isdir(video_data)
        ]) == 0:
            logger.debug(tip := f"没有有效任务")
            return Show.show_panel(self.level, tip, Wind.KEEPER)

        # Receive Argument
        platform, deploy, main_loop = args

        task_list = []
        for video_data in video_data_list:
            real_path, file_list = "", []
            logger.debug(tip := f"搜索文件夹: {os.path.basename(video_data)}")
            Show.show_panel(self.level, tip, Wind.DESIGNER)
            for root, dirs, files in os.walk(video_data, topdown=False):
                for name in files:
                    file_list.append(os.path.join(root, name))
                for name in dirs:
                    if len(name) == 1 and re.search(r"0", name):
                        real_path = os.path.join(root, name)
                        logger.debug(tip := f"分类文件夹: {os.path.basename(os.path.dirname(real_path))}")
                        Show.show_panel(self.level, tip, Wind.DESIGNER)
                        break

            if real_path == "" or len(file_list) == 0:
                logger.debug(tip := f"分类不正确: {os.path.basename(video_data)}")
                Show.show_panel(self.level, tip, Wind.KEEPER)
                continue

            efficient_list = []
            image, image_color, image_aisle = None, "grayscale", 1
            for image_file in os.listdir(real_path):
                if not os.path.isfile(image_path := os.path.join(real_path, image_file)):
                    logger.debug(tip := f"存在不适用的文件: {os.path.basename(image_path)}")
                    Show.show_panel(self.level, tip, Wind.KEEPER)
                    break

                try:
                    image = cv2.imread(image_path)
                    if image.ndim == 3:
                        if numpy.array_equal(
                                image[:, :, 0], image[:, :, 1]
                        ) and numpy.array_equal(
                            image[:, :, 1], image[:, :, 2]
                        ):
                            logger.debug(tip := f"Image: {list(image.shape)} is grayscale image, stored in RGB format")
                            efficient_list.append(tip)
                        else:
                            logger.debug(tip := f"Image: {list(image.shape)} is color image")
                            efficient_list.append(tip)
                            image_color, image_aisle = "rgb", image.ndim
                    else:
                        logger.debug(tip := f"Image: {list(image.shape)} is grayscale image")
                        efficient_list.append(tip)
                except Exception as e:
                    logger.debug(e)
                    Show.show_panel(self.level, e, Wind.KEEPER)
                    image = None
                    break

            try:
                effective = image.shape
            except AttributeError as e:
                logger.debug(e)
                Show.show_panel(self.level, e, Wind.KEEPER)
                continue

            Show.show_panel(self.level, "\n".join(efficient_list), Wind.DESIGNER)

            image_shape = deploy.shape if deploy.shape else effective
            w, h = image_shape[:2]
            w, h = max(w, 10), max(h, 10)

            src_model_path = os.path.dirname(real_path)
            new_model_path = os.path.join(
                src_model_path, f"Create_Model_{time.strftime('%Y%m%d%H%M%S')}", f"{random.randint(100, 999)}"
            )

            name = f"Gray" if image_aisle == 1 else f"Hued"
            # new_model_name = f"Keras_{name}_W{w}_H{h}_{random.randint(10000, 99999)}.h5"
            new_model_name = f"Keras_{name}_W{w}_H{h}_{random.randint(10000, 99999)}"
            task_list.append(
                (image_color, image_shape, image_aisle, src_model_path, new_model_path, new_model_name)
            )

        if len(task_list) == 0:
            logger.debug(tip := f"缺少有效文件")
            return Show.show_panel(self.level, tip, Wind.KEEPER)

        model_place = None
        alynex = Alynex(self.level, model_place, **deploy.deploys)

        # Ask Analyzer
        if len(task_list) == 1:
            task = [
                main_loop.run_in_executor(None, alynex.ks.build, *compile_data)
                for compile_data in task_list
            ]
            await asyncio.gather(*task)

        else:
            this_level = self.level
            self.level = "ERROR"
            func = partial(alynex.ks.build)
            with ProcessPoolExecutor(self.power, None, Active.active, ("ERROR",)) as exe:
                task = [
                    main_loop.run_in_executor(exe, func, *compile_data)
                    for compile_data in task_list
                ]
                await asyncio.gather(*task)
            self.level = this_level

    # 线迹创造者
    async def painting(self, *args):

        import PIL.Image
        import PIL.ImageDraw
        import PIL.ImageFont

        async def paint_lines(sn):
            image_folder = "/sdcard/Pictures/Shots"
            image = f"{time.strftime('%Y%m%d%H%M%S')}_{random.randint(100, 999)}_" + "Shot.png"
            await Terminal.cmd_line(
                self.adb, "-s", sn, "wait-for-device"
            )
            await Terminal.cmd_line(
                self.adb, "-s", sn, "shell", "mkdir", "-p", image_folder
            )
            await Terminal.cmd_line(
                self.adb, "-s", sn, "shell", "screencap", "-p", f"{image_folder}/{image}"
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                image_save_path = os.path.join(temp_dir, image)
                await Terminal.cmd_line(
                    self.adb, "-s", sn, "pull", f"{image_folder}/{image}", image_save_path
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
                    image_scale = max_scale if deploy.shape else const.COMPRESS

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
                self.adb, "-s", sn, "shell", "rm", f"{image_folder}/{image}"
            )
            return resized

        platform, deploy, main_loop = args

        manage = Manage(self.adb)
        device_list = await manage.operate_device()
        tasks = [paint_lines(device.sn) for device in device_list]
        resized_result = await asyncio.gather(*tasks)

        while True:
            action = Prompt.ask(
                f"[bold]保存图片([bold #5FD700]Y[/]/[bold #FF87AF]N[/])?[/]",
                console=Show.console, default="Y"
            )
            if action.strip().upper() == "Y":
                report = Report(self.total_place)
                report.title = f"Hooks_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
                for device, resize_img in zip(device_list, resized_result):
                    img_save_path = os.path.join(
                        report.query_path, f"hook_{device.sn}_{random.randint(10000, 99999)}.png"
                    )
                    resize_img.save(img_save_path)
                    logger.debug(tip_ := f"保存图片: {os.path.relpath(img_save_path)}")
                    Show.show_panel(self.level, tip_, Wind.DRAWER)
                break
            elif action.strip().upper() == "N":
                break
            else:
                tip_ = f"没有该选项,请重新输入\n"
                Show.show_panel(self.level, tip_, Wind.KEEPER)

    # 循环节拍器
    async def analysis(self, *args):

        async def commence():

            async def wait_for_device(device):
                Show.mark(f"[bold #FAFAD2]Wait Device Online -> {device.tag} {device.sn}[/]")
                await Terminal.cmd_line(self.adb, "-s", device.sn, "wait-for-device")

            Show.mark(f"**<* {('独立' if self.lines.alone else '全局')}控制模式 *>**")

            await source.monitor()

            await asyncio.gather(
                *(wait_for_device(device) for device in device_list)
            )

            media_screen_w, media_screen_h = ScreenMonitor.screen_size()
            Show.mark(f"Media Screen W={media_screen_w} H={media_screen_h}")

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
                    window_x = 50  # 重置当前行的开始位置
                    if (new_y_height := window_y + max_y_height) + device_y > media_screen_h:
                        window_y += margin_y  # 如果新行加设备高度超出屏幕底部，则只增加一个 margin_y
                    else:
                        window_y = new_y_height  # 否则按计划设置新行的起始位置
                    max_y_height = 0  # 重置当前行的最大高度
                max_y_height = max(max_y_height, device_y)  # 更新当前行的最大高度

                location = window_x, window_y, device_x, device_y  # 位置确认

                window_x += device_x + margin_x  # 移动到下一个设备的起始位置

                await asyncio.sleep(0.5)  # 延时投屏，避免性能瓶颈

                report.query = os.path.join(format_folder, device.sn)

                video_temp, transports = await record.ask_start_record(
                    device, report.video_path, location=location
                )
                todo_list.append(
                    [video_temp, transports, report.total_path, report.title, report.query_path,
                     report.query, report.frame_path, report.extra_path, report.proto_path]
                )

            return todo_list

        async def analysis_tactics():
            if len(task_list) == 0:
                logger.debug(tip := f"没有有效任务")
                return Show.show_panel(self.level, tip, Wind.KEEPER)

            # Information
            originals, indicates = await self.als_track(deploy, clipix, task_list, main_loop)

            # Pack Argument
            attack = deploy, clipix, report, task_list, originals, indicates

            if self.lines.speed:
                # Speed Analyzer
                await self.als_speed(*attack)
            elif self.lines.basic or self.lines.keras:
                # Keras Analyzer
                await self.als_keras(*attack, *charge)
            else:
                logger.debug(tip := f"**<* 录制模式 *>**")
                Show.show_panel(self.level, tip, Wind.EXPLORER)

        async def anything_time():
            await asyncio.gather(
                *(record.check_timer(device, timer_mode) for device in device_list)
            )

        async def anything_over():
            effective_list = await asyncio.gather(
                *(record.ask_close_record(video_temp, transports, device)
                  for (video_temp, transports, *_), device in zip(task_list, device_list))
            )

            check_list = []
            for idx, (effective, video_name) in enumerate(effective_list):
                if "视频录制失败" in effective:
                    task = task_list.pop(idx)
                    logger.debug(tip := f"{effective}: {video_name} 移除: {os.path.basename(task[0])}")
                else:
                    logger.debug(tip := f"{effective}: {video_name}")
                check_list.append(f"{tip}")
            Show.show_panel(self.level, "\n".join(check_list), Wind.EXPLORER)

        async def call_commands(exec_func, exec_args, bean, live_devices):
            if not (callable(function := getattr(bean, exec_func, None))):
                logger.debug(tip := f"No callable {exec_func}")
                return Show.show_panel(self.level, tip, Wind.KEEPER)

            sn = getattr(bean, "sn", bean.__class__.__name__)
            try:
                logger.debug(tip := f"{sn} {function.__name__} {exec_args}")
                Show.show_panel(self.level, tip, Wind.EXPLORER)
                if inspect.iscoroutinefunction(function):
                    return await function(*exec_args)
                return await asyncio.to_thread(function, *exec_args)
            except asyncio.CancelledError:
                live_devices.pop(sn)
                logger.debug(tip := f"{sn} Call Commands Exit")
                Show.show_panel(self.level, tip, Wind.EXPLORER)
            except Exception as e:
                return e

        async def load_carry(carry):
            if len(parts := re.split(r",|;|!|\s", carry, 1)) == 2:
                loc, key = parts
                if isinstance(exec_dict := await load_fully(loc), Exception):
                    return exec_dict

                try:
                    if exec_dict.get(key, None):
                        return exec_dict
                except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                    return e

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

        async def pack_commands(resolve_list):
            exec_pairs_list = []
            for resolve in resolve_list:
                device_cmds_list = resolve.get("cmds", [])
                if all(isinstance(device_cmds, str) and device_cmds != "" for device_cmds in device_cmds_list):
                    device_cmds_list = list(dict.fromkeys(device_cmds_list))
                    device_args_list = resolve.get("args", [])
                    device_args_list = [
                        device_args if isinstance(device_args, list) else ([] if device_args == "" else [device_args])
                        for device_args in device_args_list
                    ]
                    device_args_list += [[]] * (len(device_cmds_list) - len(device_args_list))
                    exec_pairs_list.append(list(zip(device_cmds_list, device_args_list)))

            return exec_pairs_list

        async def exec_commands(exec_pairs_list, *change):

            async def substitute_star():
                substitute = iter(change)
                return [
                    "".join(next(substitute, "*") if c == "*" else c for c in i)
                    if isinstance(i, str) else (next(substitute, "*") if i == "*" else i) for i in exec_args
                ]

            live_devices = {device.sn: device for device in device_list}.copy()

            exec_tasks: dict[str, "asyncio.Task"] = {}
            stop_tasks: list["asyncio.Task"] = []

            for device in device_list:
                stop_tasks.append(
                    asyncio.create_task(record.check_event(device, exec_tasks), name="stop"))

            for exec_pairs in exec_pairs_list:
                if len(live_devices) == 0:
                    return Show.mark(f"[bold #F0FFF0 on #000000]All tasks canceled[/]")
                for exec_func, exec_args in exec_pairs:
                    exec_args = await substitute_star()
                    if exec_func == "audio_player":
                        await call_commands(exec_func, exec_args, player, live_devices)
                    else:
                        for device in live_devices.values():
                            exec_tasks[device.sn] = asyncio.create_task(
                                call_commands(exec_func, exec_args, device, live_devices))

                try:
                    exec_status_list = await asyncio.gather(*exec_tasks.values())
                except asyncio.CancelledError:
                    return Show.mark(f"[bold #F0FFF0 on #000000]All tasks canceled[/]")
                finally:
                    exec_tasks.clear()

                for status in exec_status_list:
                    if isinstance(status, Exception):
                        logger.debug(status)
                        Show.show_panel(self.level, status, Wind.KEEPER)

            for stop in stop_tasks:
                stop.cancel()

        # Initialization
        manage_ = Manage(self.adb)
        device_list = await manage_.operate_device()

        platform, deploy, main_loop = args

        clipix = Clipix(self.fmp, self.fpb)

        model_place = self.model_place if self.lines.keras else None
        alynex = Alynex(self.level, model_place, **deploy.deploys)
        try:
            await alynex.ask_model_load()
            await alynex.ask_model_walk()
        except FramixAnalyzerError as e_:
            logger.debug(e_)
            Show.show_panel(self.level, e_, Wind.KEEPER)

        charge = main_loop, alynex

        titles_ = {"speed": "Speed", "basic": "Basic", "keras": "Keras"}
        input_title_ = next((title for key, title in titles_.items() if getattr(self.lines, key)), "Video")

        record = Record(
            self.scc, platform, alone=self.lines.alone, whist=self.lines.whist, frate=deploy.frate
        )
        player = Player()
        source = SourceMonitor()

        # Flick Loop
        if self.lines.flick:
            report = Report(self.total_place)
            report.title = f"{input_title_}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
            timer_mode = 5
            while True:
                try:
                    await manage_.display_device()
                    start_tips_ = f"<<<按 Enter 开始 [bold #D7FF5F]{timer_mode}[/] 秒>>>"
                    if action_ := Prompt.ask(f"[bold #5FD7FF]{start_tips_}[/]", console=Show.console):
                        if (select_ := action_.strip().lower()) == "device":
                            device_list = await manage_.another_device()
                            continue
                        elif select_ == "cancel":
                            sys.exit(Show.exit())
                        elif "header" in select_:
                            if match_ := re.search(r"(?<=header\s).*", select_):
                                if hd_ := match_.group().strip():
                                    src_hd_, a_, b_ = f"{input_title_}_{time.strftime('%Y%m%d_%H%M%S')}", 10000, 99999
                                    Show.mark(f"{const.SUC}New title set successfully[/]")
                                    report.title = f"{src_hd_}_{hd_}" if hd_ else f"{src_hd_}_{random.randint(a_, b_)}"
                                    continue
                            raise FramixAnalysisError
                        elif select_ == "create":
                            await self.combine(report)
                            break
                        elif select_ == "deploy":
                            Show.mark(f"{const.WRN}请完全退出编辑器再继续操作[/]")
                            deploy.dump_deploy(self.initial_deploy)
                            first_ = ["Notepad"] if platform == "win32" else ["open", "-W", "-a", "TextEdit"]
                            first_.append(self.initial_deploy)
                            await Terminal.cmd_line(*first_)
                            deploy.load_deploy(self.initial_deploy)
                            deploy.view_deploy()
                            continue
                        elif select_.isdigit():
                            timer_value_, lower_bound_, upper_bound_ = int(select_), 5, 300
                            if timer_value_ > 300 or timer_value_ < 5:
                                bound_tips_ = f"{lower_bound_} <= [bold #FFD7AF]Time[/] <= {upper_bound_}"
                                Show.mark(f"[bold #FFFF87]{bound_tips_}[/]")
                            timer_mode = max(lower_bound_, min(upper_bound_, timer_value_))
                        else:
                            raise FramixAnalysisError
                except FramixAnalysisError:
                    Show.tips_document()
                    continue
                else:
                    task_list = await commence()
                    await anything_time()
                    await anything_over()
                    await analysis_tactics()
                    check_ = await record.flunk_event()
                    device_list = await manage_.operate_device() if check_ else device_list
                finally:
                    await record.clean_event()

        # Other Loop
        elif self.lines.carry or self.lines.fully:

            if self.lines.carry:
                load_script_data_ = await asyncio.gather(
                    *(load_carry(carry_) for carry_ in self.lines.carry), return_exceptions=True
                )
            elif self.lines.fully:
                load_script_data_ = await asyncio.gather(
                    *(load_fully(fully_) for fully_ in self.lines.fully), return_exceptions=True
                )
            else:
                return None

            for script_data_ in load_script_data_:
                if isinstance(script_data_, Exception):
                    raise FramixAnalysisError(script_data_)
            script_storage_ = [script_data_ for script_data_ in load_script_data_]

            await manage_.display_device()
            for script_dict_ in script_storage_:
                report = Report(self.total_place)
                for script_key_, script_value_ in script_dict_.items():
                    logger.debug(tip_ := f"Batch Exec: {script_key_}")
                    Show.show_panel(self.level, tip_, Wind.EXPLORER)

                    if (parser_ := script_value_.get("parser", {})) and type(parser_) is dict:
                        for parser_key_, parser_value_ in parser_.items():
                            setattr(deploy, parser_key_, parser_value_)
                            logger.debug(f"Parser Set <{parser_key_}> {parser_value_} -> {getattr(deploy, parser_key_)}")

                    header_ = header_ if type(
                        header_ := script_value_.get("header", [])
                    ) is list else ([header_] if type(header_) is str else [time.strftime("%Y%m%d%H%M%S")])

                    if change_ := script_value_.get("change", []):
                        change_ = change_ if type(change_) is list else (
                            [change_] if type(change_) is str else [str(change_)])

                    try:
                        looper_ = int(looper_) if (looper_ := script_value_.get("looper", None)) else 1
                    except ValueError as e_:
                        logger.debug(tip_ := f"重置循环次数 {(looper_ := 1)} {e_}")
                        Show.show_panel(self.level, tip_, Wind.EXPLORER)

                    if prefix_list_ := script_value_.get("prefix", []):
                        prefix_list_ = await pack_commands(prefix_list_)
                    if action_list_ := script_value_.get("action", []):
                        action_list_ = await pack_commands(action_list_)
                    if suffix_list_ := script_value_.get("suffix", []):
                        suffix_list_ = await pack_commands(suffix_list_)

                    for hd_ in header_:
                        report.title = f"{input_title_}_{script_key_}_{hd_}"
                        for _ in range(looper_):

                            # prefix
                            if prefix_list_:
                                await exec_commands(prefix_list_)

                            # start record
                            task_list = await commence()

                            # action
                            if action_list_:
                                change_list_ = [hd_ + c_ for c_ in change_] if change_ else [hd_]
                                await exec_commands(action_list_, *change_list_)

                            # close record
                            await anything_over()

                            check_ = await record.flunk_event()
                            device_list = await manage_.operate_device() if check_ else device_list
                            await record.clean_event()

                            # suffix
                            suffix_task_list_ = []
                            if suffix_list_:
                                suffix_task_list_.append(
                                    asyncio.create_task(exec_commands(suffix_list_), name="suffix"))

                            await analysis_tactics()
                            await asyncio.gather(*suffix_task_list_)

                await self.combine(report)

        else:
            return None


class Clipix(object):

    def __init__(self, fmp: str, fpb: str):
        self.fmp = fmp
        self.fpb = fpb

    async def vision_content(
            self,
            video_temp: str,
            start: typing.Optional[str],
            close: typing.Optional[str],
            limit: typing.Optional[str],
    ) -> tuple[str, str, float, tuple, tuple]:
        """
        异步获取特定视频文件的内容分析，包括实际和平均帧率、视频时长及其视觉处理点。

        此函数分析视频文件，提供关键视频流参数，如帧率和时长，并根据输入的起始、结束和限制时间点计算处理后的视觉时间点。

        参数:
            video_temp (str): 视频文件的路径。
            start (Optional[str]): 视频处理的起始时间点（如 "00:00:10" 表示从第10秒开始）。如果为None，则从视频开始处处理。
            close (Optional[str]): 视频处理的结束时间点。如果为None，则处理到视频结束。
            limit (Optional[str]): 处理视频的最大时长限制。如果为None，则没有时间限制。

        返回:
            tuple[str, str, float, tuple, tuple]: 返回一个包含以下元素的元组：
                - rlt (str): 实际帧率。
                - avg (str): 平均帧率。
                - duration (float): 视频总时长（秒）。
                - original (tuple): 原始视频分辨率和其他基础数据。
                - (vision_start, vision_close, vision_limit) (tuple): 处理后的起始、结束和限制时间点（格式化为字符串如"00:00:10"）。

        注意:
            - 确保视频文件路径正确且视频文件可访问。
            - 输入的时间格式应为字符串形式的标准时间表示（如"HH:MM:SS"），且应确保输入合法。
            - 返回的时间点格式化为易读的字符串，方便直接使用或显示。
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

        return rlt, avg, duration, original, (vision_start, vision_close, vision_limit)

    async def vision_balance(self, duration: float, standard: float, src: str, frate: float) -> tuple[str, str]:
        """
        异步调整视频时长以匹配指定的标准时长，通过裁剪视频的起始和结束时间。

        此函数计算原视频与标准时长的差值，基于这一差值调整视频的开始和结束时间点，以生成新的视频文件，
        保证其总时长接近标准时长。适用于需要统一视频播放长度的场景。

        参数:
            duration (float): 原视频的总时长（秒）。
            standard (float): 目标视频的标准时长（秒）。
            src (str): 原视频文件的路径。
            frate (float): 目标视频的帧率。

        返回:
            tuple[str, str]: 包含两个元素：
                - video_dst (str): 调整时长后生成的新视频文件的路径。
                - video_blc (str): 描述视频时长调整详情的字符串。

        注意:
            - 确保原视频文件路径正确且文件可访问。
            - 视频处理会生成新的文件，确保有足够的磁盘空间。
            - 此函数使用异步方式进行视频处理，确保在适当的异步环境中调用。
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
    async def vision_improve(originals: list[tuple[int, int]], shape: tuple) -> tuple:
        """
        异步调整一系列原始视频的分辨率到指定的目标形状。

        此方法接收一系列视频的原始分辨率和一个目标分辨率形状，调整每个视频的分辨率以匹配这个目标形状。
        主要用于视频前处理，确保所有视频具有统一的分辨率。

        参数:
            originals (list[tuple[int, int]]): 包含每个视频的原始分辨率的列表，每个元素是一个包含宽度和高度的元组。
            shape (tuple): 目标视频分辨率形状，为一个包含目标宽度和高度的元组。

        返回:
            tuple: 包含每个视频调整后的新分辨率信息的元组。

        注意:
            - 此函数是异步的，需要在适当的异步环境中运行。
            - 确保所有的输入参数都是准确和有效的。
        """
        final_shape_list = await asyncio.gather(
            *(Switch.ask_magic_frame(original, shape) for original in originals)
        )
        return final_shape_list

    async def pixels(self, function: "typing.Callable", video_filter: list, src: str, dst: str, **kwargs) -> tuple[str]:
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

        注意:
            - 该方法依赖于提供的`function`能够异步执行并返回处理结果。
            - 确保源视频和目标视频路径正确，且文件系统有足够权限读写文件。
        """
        return await function(self.fmp, video_filter, src, dst, **kwargs)


class Alynex(object):

    __ks: typing.Optional["KerasStruct"] = KerasStruct()

    def __init__(self, level: str, model_place: typing.Optional[str] = None, **kwargs):
        self.level = level
        self.model_place = model_place

        self.boost = kwargs.get("boost", const.BOOST)
        self.color = kwargs.get("color", const.COLOR)

        self.shape = kwargs.get("shape", const.SHAPE)
        self.scale = kwargs.get("scale", const.SCALE)
        _ = kwargs.get("start", const.START)
        _ = kwargs.get("close", const.CLOSE)
        _ = kwargs.get("limit", const.LIMIT)
        self.begin = kwargs.get("begin", const.BEGIN)
        self.final = kwargs.get("final", const.FINAL)

        _ = kwargs.get("frate", const.FRATE)
        self.thres = kwargs.get("thres", const.THRES)
        self.shift = kwargs.get("shift", const.SHIFT)
        self.block = kwargs.get("block", const.BLOCK)
        self.crops = kwargs.get("crops", const.CROPS)
        self.omits = kwargs.get("omits", const.OMITS)

    @property
    def ks(self) -> typing.Optional["KerasStruct"]:
        return self.__ks

    @ks.setter
    def ks(self, value):
        self.__ks = value

    async def ask_model_load(self):
        if self.model_place:
            try:
                assert self.ks
                self.ks.load_model(self.model_place)
            except (OSError, ValueError, AssertionError) as e:
                self.ks.model = None
                raise FramixAnalyzerError(e)

    async def ask_model_walk(self):
        if self.ks.model:
            try:
                channel = self.ks.model.input_shape[-1]
                if self.color:
                    assert channel == 3, f"彩色模式需要匹配彩色模型 Model Color Channel={channel}"
                else:
                    assert channel == 1, f"灰度模式需要匹配灰度模型 Model Color Channel={channel}"
            except AssertionError as e:
                self.ks.model = None
                raise FramixAnalyzerError(e)

    @staticmethod
    async def ask_frame_grid(vision: str):
        target_screen = None
        if os.path.isfile(vision):
            screen = cv2.VideoCapture(vision)
            if screen.isOpened():
                target_screen = vision
            screen.release()
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

    async def ask_frame_flip(self, shape: tuple, scale: float, original: tuple):
        if shape:
            w, h, ratio = await Switch.ask_magic_frame(original, shape)
            shape = w, h
            logger.debug(f"{(tip := f'调整宽高比: {w} x {h}')}")
            Show.show_panel(self.level, tip, Wind.LOADER)
        elif scale:
            scale = max(0.1, min(1.0, scale))
        else:
            scale = 0.4

        return shape, scale

    async def ask_exercise(self, vision: str, *args) -> typing.Optional[str]:
        if (target_vision := await self.ask_frame_grid(vision)) is None:
            logger.debug(tip := f"视频文件损坏: {os.path.basename(vision)}")
            return Show.show_panel(self.level, tip, Wind.KEEPER)

        query_path, original, *_ = args

        shape, scale = await self.ask_frame_flip(self.shape, self.scale, original)

        load_start_time = time.time()
        video = VideoObject(target_vision)
        logger.debug(f"{(task_name := '视频帧长度: ' f'{video.frame_count}')}")
        logger.debug(f"{(task_info := '视频帧尺寸: ' f'{video.frame_size}')}")
        logger.debug(f"{(task_desc := '加载视频帧: ' f'{video.name}')}")
        Show.show_panel(self.level, f"{task_name}\n{task_info}\n{task_desc}", Wind.LOADER)
        video.load_frames(
            scale=scale, shape=shape, color=self.color
        )
        logger.debug(f"{(task_name := '视频帧加载完成: ' f'{video.frame_details(video.frames_data)}')}")
        logger.debug(f"{(task_info := '视频帧加载耗时: ' f'{time.time() - load_start_time:.2f} 秒')}")
        Show.show_panel(self.level, f"{task_name}\n{task_info}", Wind.LOADER)

        cut_start_time = time.time()
        cutter = VideoCutter()
        logger.debug(f"{(cut_name := '视频帧长度: ' f'{video.frame_count}')}")
        logger.debug(f"{(cut_part := '视频帧片段: ' f'{video.frame_count - 1}')}")
        logger.debug(f"{(cut_info := '视频帧尺寸: ' f'{video.frame_size}')}")
        logger.debug(f"{(cut_desc := '压缩视频帧: ' f'{video.name}')}")
        Show.show_panel(self.level, f"{cut_name}\n{cut_part}\n{cut_info}\n{cut_desc}", Wind.CUTTER)
        cut_range = cutter.cut(
            video=video, block=self.block
        )
        logger.debug(f"{(cut_name := '视频帧压缩完成: ' f'{video.name}')}")
        logger.debug(f"{(cut_info := '视频帧压缩耗时: ' f'{time.time() - cut_start_time:.2f} 秒')}")
        Show.show_panel(self.level, f"{cut_name}\n{cut_info}", Wind.CUTTER)

        stable, unstable = cut_range.get_range(
            threshold=self.thres, offset=self.shift
        )

        return cut_range.pick_and_save(
            stable, 20, query_path,
            meaningful_name=True,
            compress_rate=None,
            target_size=None,
            not_grey=True
        )

    async def ask_analyzer(self, vision: str, *args) -> typing.Optional["Review"]:

        async def frame_forge(frame):
            try:
                picture = f"{frame.frame_id}_{format(round(frame.timestamp, 5), '.5f')}.png"
                _, codec = cv2.imencode(".png", frame.data)
                async with aiofiles.open(os.path.join(frame_path, picture), "wb") as f:
                    await f.write(codec.tobytes())
            except Exception as e:
                return e
            return {"id": frame.frame_id, "picture": os.path.join(os.path.basename(frame_path), picture)}

        async def frame_flick():
            begin_stage_index, begin_frame_index = self.begin
            final_stage_index, final_frame_index = self.final
            logger.debug(
                f"{(extract := f'取关键帧: begin={list(self.begin)} final={list(self.final)}')}"
            )
            Show.show_panel(self.level, extract, Wind.FASTER)

            try:
                logger.debug(f"{(stage_name := f'阶段划分: {struct.get_ordered_stage_set()}')}")
                Show.show_panel(self.level, stage_name, Wind.FASTER)
                unstable_stage_range = struct.get_not_stable_stage_range()
                begin_frame = unstable_stage_range[begin_stage_index][begin_frame_index]
                final_frame = unstable_stage_range[final_stage_index][final_frame_index]
            except (AssertionError, IndexError) as e:
                logger.debug(e)
                Show.show_panel(self.level, e, Wind.KEEPER)
                begin_frame = struct.get_important_frame_list()[0]
                final_frame = struct.get_important_frame_list()[-1]

            if final_frame.frame_id <= begin_frame.frame_id:
                logger.debug(tip := f"{final_frame} <= {begin_frame}")
                Show.show_panel(self.level, tip, Wind.KEEPER)
                begin_frame, end_frame = struct.data[0], struct.data[-1]

            time_cost = final_frame.timestamp - begin_frame.timestamp

            begin_id, begin_ts = begin_frame.frame_id, begin_frame.timestamp
            final_id, final_ts = final_frame.frame_id, final_frame.timestamp
            begin_fr, final_fr = f"{begin_id} - {begin_ts:.5f}", f"{final_id} - {final_ts:.5f}"
            logger.debug(f"开始帧:[{begin_fr}] 结束帧:[{final_fr}] 总耗时:[{(stage_cs := f'{time_cost:.5f}')}]")
            if self.level == "INFO":
                Show.assort_frame(begin_fr, final_fr, stage_cs)
            return begin_frame.frame_id, final_frame.frame_id, time_cost

        async def frame_hold():
            if struct is None:
                return [i for i in video.frames_data]

            frames_list = []
            important_frames = struct.get_important_frame_list()
            if self.boost:
                pbar = toolbox.show_progress(total=struct.get_length(), color=50)
                frames_list.append(previous := important_frames[0])
                pbar.update(1)
                for current in important_frames[1:]:
                    frames_list.append(current)
                    pbar.update(1)
                    frames_diff = current.frame_id - previous.frame_id
                    if not previous.is_stable() and not current.is_stable() and frames_diff > 1:
                        for specially in struct.data[previous.frame_id: current.frame_id - 1]:
                            frames_list.append(specially)
                            pbar.update(1)
                    previous = current
                pbar.close()
            else:
                for current in toolbox.show_progress(items=struct.data, color=50):
                    frames_list.append(current)

            return frames_list

        async def frame_flow():
            cut_start_time = time.time()
            cutter = VideoCutter()

            just_hook_list, area_hook_list = [], []

            size_hook = FrameSizeHook(1.0, None, True)
            cutter.add_hook(size_hook)
            logger.debug(
                f"{(cut_size := f'视频帧处理: {size_hook.__class__.__name__} {[1.0, None, True]}')}"
            )
            just_hook_list.append(cut_size)

            if len(crop_list := self.crops) > 0 and sum([j for i in crop_list for j in i.values()]) > 0:
                for crop in crop_list:
                    x, y, x_size, y_size = crop.values()
                    crop_hook = PaintCropHook((y_size, x_size), (y, x))
                    cutter.add_hook(crop_hook)
                    logger.debug(
                        f"{(cut_crop := f'视频帧处理: {crop_hook.__class__.__name__} {x, y, x_size, y_size}')}"
                    )
                    area_hook_list.append(cut_crop)

            if len(omit_list := self.omits) > 0 and sum([j for i in omit_list for j in i.values()]) > 0:
                for omit in omit_list:
                    x, y, x_size, y_size = omit.values()
                    omit_hook = PaintOmitHook((y_size, x_size), (y, x))
                    cutter.add_hook(omit_hook)
                    logger.debug(
                        f"{(cut_omit := f'视频帧处理: {omit_hook.__class__.__name__} {x, y, x_size, y_size}')}"
                    )
                    area_hook_list.append(cut_omit)

            save_hook = FrameSaveHook(extra_path)
            cutter.add_hook(save_hook)
            logger.debug(
                f"{(cut_save := f'视频帧处理: {save_hook.__class__.__name__} {[os.path.basename(extra_path)]}')}"
            )
            just_hook_list.append(cut_save)

            if len(just_hook_list) > 0:
                Show.show_panel(self.level, "\n".join(just_hook_list), Wind.CUTTER)
            if len(area_hook_list) > 0:
                Show.show_panel(self.level, "\n".join(area_hook_list), Wind.CUTTER)

            logger.debug(f"{(cut_name := '视频帧长度: ' f'{video.frame_count}')}")
            logger.debug(f"{(cut_part := '视频帧片段: ' f'{video.frame_count - 1}')}")
            logger.debug(f"{(cut_info := '视频帧尺寸: ' f'{video.frame_size}')}")
            logger.debug(f"{(cut_desc := '压缩视频帧: ' f'{video.name}')}")
            Show.show_panel(self.level, f"{cut_name}\n{cut_part}\n{cut_info}\n{cut_desc}", Wind.CUTTER)
            cut_range = cutter.cut(
                video=video, block=self.block
            )
            logger.debug(f"{(cut_name := '视频帧压缩完成: ' f'{video.name}')}")
            logger.debug(f"{(cut_info := '视频帧压缩耗时: ' f'{time.time() - cut_start_time:.2f} 秒')}")
            Show.show_panel(self.level, f"{cut_name}\n{cut_info}", Wind.CUTTER)

            stable, unstable = cut_range.get_range(threshold=self.thres, offset=self.shift)

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
                toolbox.draw_line(os.path.join(extra_path, draw))

            try:
                struct_data = self.ks.classify(
                    video=video, valid_range=stable, keep_data=True
                )
            except AssertionError as e:
                logger.debug(e)
                return Show.show_panel(self.level, e, Wind.KEEPER)
            return struct_data

        async def analytics_basic():
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
                        logger.error(f"{const.ERR}{r}[/]")
                    else:
                        scores[r["id"]] = r["picture"]

            begin_frame, final_frame = frames[0], frames[-1]
            time_cost = final_frame.timestamp - begin_frame.timestamp
            return begin_frame.frame_id, final_frame.frame_id, time_cost, scores, None

        async def analytics_keras():
            forge_tasks = [
                [frame_forge(frame) for frame in chunk] for chunk in
                [frames[i:i + 100] for i in range(0, len(frames), 100)]
            ]
            flick_result, *forge_result = await asyncio.gather(
                frame_flick(), *(asyncio.gather(*forge) for forge in forge_tasks)
            )

            scores = {}
            for result in forge_result:
                for r in result:
                    if isinstance(r, Exception):
                        logger.error(f"{const.ERR}{r}[/]")
                    else:
                        scores[r["id"]] = r["picture"]

            begin_frame_id, final_frame_id, time_cost = flick_result
            return begin_frame_id, final_frame_id, time_cost, scores, struct

        if (target_vision_ := await self.ask_frame_grid(vision)) is None:
            logger.debug(tip_ := f"视频文件损坏: {os.path.basename(vision)}")
            return Show.show_panel(self.level, tip_, Wind.KEEPER)

        frame_path, extra_path, original, *_ = args

        shape_, scale_ = await self.ask_frame_flip(self.shape, self.scale, original)

        start_time_ = time.time()
        video = VideoObject(target_vision_)
        logger.debug(f"{(task_name_ := '视频帧长度: ' f'{video.frame_count}')}")
        logger.debug(f"{(task_info_ := '视频帧尺寸: ' f'{video.frame_size}')}")
        logger.debug(f"{(task_desc_ := '加载视频帧: ' f'{video.name}')}")
        Show.show_panel(self.level, f"{task_name_}\n{task_info_}\n{task_desc_}", Wind.LOADER)
        video.load_frames(
            scale=scale_, shape=shape_, color=self.color
        )
        logger.debug(f"{(task_name := '视频帧加载完成: ' f'{video.frame_details(video.frames_data)}')}")
        logger.debug(f"{(task_info := '视频帧加载耗时: ' f'{time.time() - start_time_:.2f} 秒')}")
        Show.show_panel(self.level, f"{task_name}\n{task_info}", Wind.LOADER)

        struct = await frame_flow() if self.ks.model else None
        frames = await frame_hold()

        if struct:
            return Review(*(await analytics_keras()))
        return Review(*(await analytics_basic()))


async def arithmetic(function: "typing.Callable", parameters: list[str], *args) -> None:
    try:
        parameters = [(await Craft.revise_path(param)) for param in parameters]
        await function(parameters, *args)
    except (FramixAnalysisError, FramixAnalyzerError, FramixReporterError):
        Show.console.print_exception()
        sys.exit(Show.fail())


async def scheduling() -> None:
    try:
        # --flick --carry --fully
        if _lines.flick or _lines.carry or _lines.fully:
            await _missions.analysis(
                _platform, _deploy, _main_loop
            )
        # --paint
        elif _lines.paint:
            await _missions.painting(
                _platform, _deploy, _main_loop
            )
        # --union
        elif _lines.union:
            await _missions.combine_view(_lines.union)
        # --merge
        elif _lines.merge:
            await _missions.combine_main(_lines.merge)
        else:
            Show.help_document()
    except (FramixAnalysisError, FramixAnalyzerError, FramixReporterError):
        Show.console.print_exception()
        sys.exit(Show.fail())


if __name__ == '__main__':
    _lines = Parser.parse_cmd()

    Active.active(_level := "DEBUG" if _lines.debug else "INFO")

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

    _initial_option = os.path.join(_initial_source, "option.json")
    _initial_deploy = os.path.join(_initial_source, "deploy.json")
    _initial_script = os.path.join(_initial_source, "script.json")
    logger.debug(f"配置文件路径: {_initial_option}")
    logger.debug(f"部署文件路径: {_initial_deploy}")
    logger.debug(f"脚本文件路径: {_initial_script}")

    _option = Option(_initial_option)
    _total_place = _option.total_place or _total_place
    _model_place = _option.model_place or _model_place
    logger.debug(f"报告文件路径: {_total_place}")
    logger.debug(f"模型文件路径: {_model_place}")

    logger.debug(f"处理器核心数: {(_power := os.cpu_count())}")

    _main_loop: "asyncio.AbstractEventLoop" = asyncio.get_event_loop()

    _deploy = Deploy(_initial_deploy)
    for _attr, _attribute in _deploy.deploys.items():
        if any(_line.startswith(f"--{_attr}") for _line in _wires):
            setattr(_deploy, _attr, getattr(_lines, _attr))
            logger.debug(f"Initialize Set <{_attr}> {_attribute} -> {getattr(_deploy, _attr)}")

    _missions = Missions(
        _level,
        _power,
        _lines,
        atom_total_temp=_atom_total_temp,
        main_share_temp=_main_share_temp,
        main_total_temp=_main_total_temp,
        view_share_temp=_view_share_temp,
        view_total_temp=_view_total_temp,
        initial_option=_initial_option,
        initial_deploy=_initial_deploy,
        initial_script=_initial_script,
        total_place=_total_place,
        model_place=_model_place,
        adb=_adb,
        fmp=_fmp,
        fpb=_fpb,
        scc=_scc,
    )

    from functools import partial
    from concurrent.futures import ProcessPoolExecutor

    Show.load_animation()

    _args = _platform, _deploy, _main_loop

    # Main Process
    try:
        # --video
        if _video_list := _lines.video:
            _main_loop.run_until_complete(
                arithmetic(_missions.video_file_task, _video_list, *_args)
            )
        # --stack
        elif _stack_list := _lines.stack:
            _main_loop.run_until_complete(
                arithmetic(_missions.video_data_task, _stack_list, *_args)
            )
        # --train
        elif _train_list := _lines.train:
            _main_loop.run_until_complete(
                arithmetic(_missions.train_model, _train_list, *_args)
            )
        # --build
        elif _build_list := _lines.build:
            _main_loop.run_until_complete(
                arithmetic(_missions.build_model, _build_list, *_args)
            )
        else:
            from engine.manage import ScreenMonitor
            from engine.manage import SourceMonitor
            from engine.medias import Record
            from engine.medias import Player

            _main_loop.run_until_complete(scheduling())

    except KeyboardInterrupt:
        sys.exit(Show.exit())
    except (OSError, RuntimeError, MemoryError, TypeError, ValueError, AttributeError):
        Show.console.print_exception()
        sys.exit(Show.fail())
    else:
        sys.exit(Show.done())
