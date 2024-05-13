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

_lines = sys.argv[1:]

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
    from frameflow.skills.config import Option
    from frameflow.skills.config import Deploy
    from frameflow.skills.config import Script
    from frameflow.skills.insert import Insert
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

    def __init__(self, *args, **kwargs):
        self.flick, self.carry, self.fully, self.speed, self.basic, self.keras, *_ = args
        *_, self.alone, self.whist, self.alike, self.group = args
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

    @staticmethod
    def enforce(reset_path: str, kc: typing.Optional["KerasStruct"], start: int, end: int, cost: float, *args):
        with Insert(os.path.join(reset_path, f"{const.NAME}_data.db")) as database:
            if kc:
                total_path, title, query_path, query, frame_path, extra_path, proto_path, *_ = args
                column_list = [
                    "total_path", "title", "query_path", "query", "stage", "frame_path", "extra_path", "proto_path"
                ]
                database.create("stocks", *column_list)
                stage = {"stage": {"start": start, "end": end, "cost": cost}}
                database.insert(
                    "stocks", column_list,
                    (total_path, title, query_path, query,
                     json.dumps(stage), frame_path, extra_path, proto_path)
                )
            else:
                total_path, title, query_path, query, frame_path, *_ = args
                column_list = [
                    "total_path", "title", "query_path", "query", "stage", "frame_path"
                ]
                database.create('stocks', *column_list)
                stage = {"stage": {"start": start, "end": end, "cost": cost}}
                database.insert(
                    "stocks", column_list,
                    (total_path, title, query_path, query, json.dumps(stage), frame_path)
                )

    # """Child Process"""
    def amazing(self, vision: str, *args, **kwargs):
        attack = self.total_place, self.model_place
        alynex = Alynex(*attack)
        if alynex.kc:
            try:
                channel = alynex.kc.model.input_shape[-1]
                if kwargs["color"]:
                    assert channel == 3, f"彩色模式需要匹配彩色模型 Model Color Channel={channel}"
                else:
                    assert channel == 1, f"灰度模式需要匹配灰度模型 Model Color Channel={channel}"
            except AssertionError as e:
                logger.error(f"{const.ERR}{e}[/]")
                alynex.kc = None

        loop = asyncio.get_event_loop()
        loop_complete = loop.run_until_complete(
            alynex.ask_analyzer(vision, *args, **kwargs)
        )
        return loop_complete

    # """Child Process"""
    async def video_file_task(self, video_file_list: list, *args):
        video_file_list = [
            video_file for video_file in video_file_list if os.path.isfile(video_file)
        ]

        if len(video_file_list) == 0:
            return logger.warning(f"{const.WRN}没有有效任务[/]")

        cmd_lines, platform, deploy, level, power, main_loop, *_ = args
        clipix = Clipix(self.fmp, self.fpb)

        reporter = Report(self.total_place)
        reporter.title = f"{const.DESC}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

        task_list = []
        for video_file in video_file_list:
            reporter.query = f"{os.path.basename(video_file).split('.')[0]}_{time.strftime('%Y%m%d%H%M%S')}"
            new_video_path = os.path.join(reporter.video_path, os.path.basename(video_file))
            shutil.copy(video_file, new_video_path)
            task_list.append(
                [new_video_path, None, reporter.total_path, reporter.title, reporter.query_path,
                 reporter.query, reporter.frame_path, reporter.extra_path, reporter.proto_path]
            )

        start_ms = Parser.parse_mills(deploy.start)
        close_ms = Parser.parse_mills(deploy.close)
        limit_ms = Parser.parse_mills(deploy.limit)
        content_list = await asyncio.gather(
            *(clipix.vision_content(video_temp, start_ms, close_ms, limit_ms, deploy.frate)
              for video_temp, *_ in task_list)
        )
        duration_list, original_list, final_point_list = zip(*content_list)

        if self.alike and len(task_list) > 1:
            logger.info(f"视频均衡: [{(standard := min(duration_list))}] [{Parser.parse_times(standard)}]")
            await asyncio.gather(
                *(clipix.vision_balance(duration, standard, video_idx, video_src, deploy.frate, task_list)
                  for duration, (video_idx, (video_src, *_)) in zip(
                    duration_list, enumerate(task_list)) if duration != standard)
            )

        if self.speed:
            logger.info(f"△ △ △ 快速模式 △ △ △")

            const_filter = [f"fps={deploy.frate}"] if deploy.color else [f"fps={deploy.frate}", "format=gray"]
            if deploy.shape:
                final_shape_list = await clipix.vision_improve(
                    original_list, deploy.shape
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
                logger.info(f"{os.path.basename(video_temp)} 视频过滤: {flt}")

            await asyncio.gather(
                *(clipix.pixels(
                    Switch.ask_video_detach, video_filter, video_temp, frame_path,
                    start=vision_start, close=vision_close, limit=vision_limit
                ) for video_filter, (video_temp, *_, frame_path, _, _), (vision_start, vision_close, vision_limit)
                    in zip(video_filter_list, task_list, final_point_list))
            )

            start, end, cost, scores, struct = 0, 0, 0, None, None
            for *_, total_path, title, query_path, query, frame_path, _, _ in task_list:
                result = {
                    "total": os.path.basename(total_path),
                    "title": title,
                    "query": query,
                    "stage": {"start": start, "end": end, "cost": cost},
                    "frame": os.path.basename(frame_path),
                    "style": "speed"
                }
                logger.debug(f"Speeder: {json.dumps(result, ensure_ascii=False)}")
                await reporter.load(result)

                self.enforce(
                    reporter.reset_path, struct, start, end, cost,
                    total_path, title, query_path, query, frame_path
                )

            logger.info(f"正在生成汇总报告 ...")
            try:
                total_html = await reporter.ask_create_total_report(
                    os.path.dirname(reporter.total_path),
                    self.group,
                    await Craft.achieve(self.view_share_temp),
                    await Craft.achieve(self.view_total_temp)
                )
            except FramixReporterError:
                return Show.console.print_exception()

            logger.info(f"成功生成汇总报告 {os.path.relpath(total_html)}")
            return reporter.total_path

        elif self.keras and not self.basic:
            attack = self.total_place, self.model_place
        else:
            attack = self.total_place, None

        alynex = Alynex(*attack)
        if alynex.kc:
            try:
                channel = alynex.kc.model.input_shape[-1]
                if deploy.color:
                    assert channel == 3, f"彩色模式需要匹配彩色模型 Model Color Channel={channel}"
                else:
                    assert channel == 1, f"灰度模式需要匹配灰度模型 Model Color Channel={channel}"
            except AssertionError as e:
                logger.error(f"{const.ERR}{e}[/]")
                alynex.kc = None

        logger.info(f"△ △ △ {'智能模式' if alynex.kc else '基础模式'} △ △ △")

        video_target_list = [
            (os.path.join(
                os.path.dirname(video_temp), f"vision_fps{deploy.frate}_{random.randint(100, 999)}.mp4"
            ), [f"fps={deploy.frate}"]) for video_temp, *_ in task_list
        ]

        await asyncio.gather(
            *(clipix.pixels(
                Switch.ask_video_change, video_filter, video_temp,
                target, start=vision_start, close=vision_close, limit=vision_limit
            ) for (target, video_filter), (video_temp, *_), (vision_start, vision_close, vision_limit)
                in zip(video_target_list, task_list, final_point_list))
        )

        for (target, _), (video_temp, *_) in zip(video_target_list, task_list):
            logger.info(f"视频转换完成: {os.path.basename(target)}")
            os.remove(video_temp)
            logger.info(f"移除旧的视频: {os.path.basename(video_temp)}")

        if len(task_list) == 1:
            task = [
                alynex.ask_analyzer(target, frame_path, extra_path, original, **deploy.deploys)
                for (target, _), (*_, frame_path, extra_path, _), original
                in zip(video_target_list, task_list, original_list)
            ]
            futures = await asyncio.gather(*task)

        else:
            func = partial(self.amazing, **deploy.deploys)
            with ProcessPoolExecutor(power, None, Active.active, ("ERROR",)) as exe:
                task = [
                    main_loop.run_in_executor(exe, func, target, frame_path, extra_path, original)
                    for (target, _), (*_, frame_path, extra_path, _), original
                    in zip(video_target_list, task_list, original_list)
                ]
                futures = await asyncio.gather(*task)

        for future, todo in zip(futures, task_list):
            if future is None:
                continue

            start, end, cost, scores, struct = future.material
            *_, total_path, title, query_path, query, frame_path, extra_path, proto_path = todo

            result = {
                "total": os.path.basename(total_path),
                "title": title,
                "query": query,
                "stage": {"start": start, "end": end, "cost": f"{cost:.5f}"},
                "frame": os.path.basename(frame_path)
            }

            if struct:
                if isinstance(
                        tmp := await Craft.achieve(self.atom_total_temp), Exception):
                    return logger.error(f"{const.ERR}{tmp}[/]")
                logger.info(f"模版引擎正在渲染 ...")
                original_inform = await reporter.ask_draw(
                    scores, struct, proto_path, tmp, deploy.boost
                )
                logger.info(f"模版引擎渲染完毕 {os.path.relpath(original_inform)}")
                result["extra"] = os.path.basename(extra_path)
                result["proto"] = os.path.basename(original_inform)
                result["style"] = "keras"
            else:
                result["style"] = "basic"

            logger.debug(f"Restore: {json.dumps(result, ensure_ascii=False)}")
            await reporter.load(result)

            self.enforce(
                reporter.reset_path, struct, start, end, cost,
                total_path, title, query_path, query, frame_path, extra_path, proto_path
            )

        logger.info(f"正在生成汇总报告 ...")
        try:
            total_html = await reporter.ask_create_total_report(
                os.path.dirname(reporter.total_path),
                self.group,
                await Craft.achieve(self.main_share_temp),
                await Craft.achieve(self.main_total_temp)
            )
        except FramixReporterError:
            return Show.console.print_exception()

        logger.info(f"成功生成汇总报告 {os.path.relpath(total_html)}")
        return reporter.total_path

    # """Child Process"""
    async def video_data_task(self, video_data_list: list, *args):

        async def load_entries():
            for video_data in video_data_list:
                find_result = find.accelerate(video_data)
                if isinstance(find_result, Exception):
                    logger.error(f"{const.ERR}{find_result}")
                    continue
                root_tree, collection_list = find_result
                Show.console.print(root_tree)
                yield collection_list[0]

        find = Find()

        cmd_lines, platform, deploy, level, power, main_loop, *_ = args
        clipix = Clipix(self.fmp, self.fpb)

        reporter = Report(self.total_place)

        async for entries in load_entries():
            for entry in entries:
                reporter.title = entry.title
                task_list = []
                for video in entry.sheet:
                    reporter.query = video["query"]
                    shutil.copy(path := video["video"], reporter.video_path)
                    new_video_path = os.path.join(reporter.video_path, os.path.basename(path))
                    task_list.append(
                        [new_video_path, None, reporter.total_path, reporter.title, reporter.query_path,
                         reporter.query, reporter.frame_path, reporter.extra_path, reporter.proto_path]
                    )

                start_ms = Parser.parse_mills(deploy.start)
                close_ms = Parser.parse_mills(deploy.close)
                limit_ms = Parser.parse_mills(deploy.limit)
                content_list = await asyncio.gather(
                    *(clipix.vision_content(video_temp, start_ms, close_ms, limit_ms, deploy.frate)
                      for video_temp, *_ in task_list)
                )
                duration_list, original_list, final_point_list = zip(*content_list)

                if self.alike and len(task_list) > 1:
                    logger.info(f"视频均衡: [{(standard := min(duration_list))}] [{Parser.parse_times(standard)}]")
                    await asyncio.gather(
                        *(clipix.vision_balance(duration, standard, video_idx, video_src, deploy.frate, task_list)
                          for duration, (video_idx, (video_src, *_)) in zip(
                            duration_list, enumerate(task_list)) if duration != standard)
                    )

                if self.speed:
                    logger.info(f"△ △ △ 快速模式 △ △ △")

                    const_filter = [f"fps={deploy.frate}"] if deploy.color else [f"fps={deploy.frate}", "format=gray"]
                    if deploy.shape:
                        final_shape_list = await clipix.vision_improve(
                            original_list, deploy.shape
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
                        logger.info(f"{os.path.basename(video_temp)} 视频过滤: {flt}")

                    await asyncio.gather(
                        *(clipix.pixels(
                            Switch.ask_video_detach, video_filter, video_temp, frame_path,
                            start=vision_start, close=vision_close, limit=vision_limit
                        ) for video_filter, (video_temp, *_, frame_path, _, _), (vision_start, vision_close, vision_limit)
                            in zip(video_filter_list, task_list, final_point_list))
                    )

                    start, end, cost, scores, struct = 0, 0, 0, None, None
                    for *_, total_path, title, query_path, query, frame_path, _, _ in task_list:
                        result = {
                            "total": os.path.basename(total_path),
                            "title": title,
                            "query": query,
                            "stage": {"start": start, "end": end, "cost": cost},
                            "frame": os.path.basename(frame_path),
                            "style": "speed"
                        }
                        logger.debug(f"Speeder: {json.dumps(result, ensure_ascii=False)}")
                        await reporter.load(result)

                        self.enforce(
                            reporter.reset_path, struct, start, end, cost,
                            total_path, title, query_path, query, frame_path
                        )

                    logger.info(f"正在生成汇总报告 ...")
                    try:
                        total_html = await reporter.ask_create_total_report(
                            os.path.dirname(reporter.total_path),
                            self.group,
                            await Craft.achieve(self.view_share_temp),
                            await Craft.achieve(self.view_total_temp)
                        )
                    except FramixReporterError:
                        Show.console.print_exception()
                        continue

                    logger.info(f"成功生成汇总报告 {os.path.relpath(total_html)}")
                    continue

                elif self.keras and not self.basic:
                    attack = self.total_place, self.model_place
                else:
                    attack = self.total_place, None

                alynex = Alynex(*attack)
                if alynex.kc:
                    try:
                        channel = alynex.kc.model.input_shape[-1]
                        if deploy.color:
                            assert channel == 3, f"彩色模式需要匹配彩色模型 Model Color Channel={channel}"
                        else:
                            assert channel == 1, f"灰度模式需要匹配灰度模型 Model Color Channel={channel}"
                    except AssertionError as e:
                        logger.error(f"{const.ERR}{e}[/]")
                        alynex.kc = None

                logger.info(f"△ △ △ {'智能模式' if alynex.kc else '基础模式'} △ △ △")

                video_target_list = [
                    (os.path.join(
                        os.path.dirname(video_temp), f"vision_fps{deploy.frate}_{random.randint(100, 999)}.mp4"
                    ), [f"fps={deploy.frate}"]) for video_temp, *_ in task_list
                ]

                await asyncio.gather(
                    *(clipix.pixels(
                        Switch.ask_video_change, video_filter, video_temp,
                        target, start=vision_start, close=vision_close, limit=vision_limit
                    ) for (target, video_filter), (video_temp, *_), (vision_start, vision_close, vision_limit)
                        in zip(video_target_list, task_list, final_point_list))
                )

                for (target, _), (video_temp, *_) in zip(video_target_list, task_list):
                    logger.info(f"视频转换完成: {os.path.basename(target)}")
                    os.remove(video_temp)
                    logger.info(f"移除旧的视频: {os.path.basename(video_temp)}")

                if len(task_list) == 1:
                    task = [
                        alynex.ask_analyzer(target, frame_path, extra_path, original, **deploy.deploys)
                        for (target, _), (*_, frame_path, extra_path, _), original
                        in zip(video_target_list, task_list, original_list)
                    ]
                    futures = await asyncio.gather(*task)

                else:
                    func = partial(self.amazing, **deploy.deploys)
                    with ProcessPoolExecutor(power, None, Active.active, ("ERROR",)) as exe:
                        task = [
                            main_loop.run_in_executor(exe, func, target, frame_path, extra_path, original)
                            for (target, _), (*_, frame_path, extra_path, _), original
                            in zip(video_target_list, task_list, original_list)
                        ]
                        futures = await asyncio.gather(*task)

                for future, todo in zip(futures, task_list):
                    if future is None:
                        continue

                    start, end, cost, scores, struct = future.material
                    *_, total_path, title, query_path, query, frame_path, extra_path, proto_path = todo

                    result = {
                        "total": os.path.basename(total_path),
                        "title": title,
                        "query": query,
                        "stage": {"start": start, "end": end, "cost": f"{cost:.5f}"},
                        "frame": os.path.basename(frame_path)
                    }

                    if struct:
                        if isinstance(
                                tmp := await Craft.achieve(self.atom_total_temp), Exception):
                            return logger.error(f"{const.ERR}{tmp}[/]")
                        logger.info(f"模版引擎正在渲染 ...")
                        original_inform = await reporter.ask_draw(
                            scores, struct, proto_path, tmp, deploy.boost
                        )
                        logger.info(f"模版引擎渲染完毕 {os.path.relpath(original_inform)}")
                        result["extra"] = os.path.basename(extra_path)
                        result["proto"] = os.path.basename(original_inform)
                        result["style"] = "keras"
                    else:
                        result["style"] = "basic"

                    logger.debug(f"Restore: {json.dumps(result, ensure_ascii=False)}")
                    await reporter.load(result)

                    self.enforce(
                        reporter.reset_path, struct, start, end, cost,
                        total_path, title, query_path, query, frame_path, extra_path, proto_path
                    )

                logger.info(f"正在生成汇总报告 ...")
                try:
                    total_html = await reporter.ask_create_total_report(
                        os.path.dirname(reporter.total_path),
                        self.group,
                        await Craft.achieve(self.main_share_temp),
                        await Craft.achieve(self.main_total_temp)
                    )
                except FramixReporterError:
                    Show.console.print_exception()
                    continue

                logger.info(f"成功生成汇总报告 {os.path.relpath(total_html)}")
                continue

    # """Child Process"""
    def train_model(self, video_file: str, deploy: "Deploy"):
        if not os.path.isfile(video_file):
            return logger.error(f"{const.ERR}视频文件丢失 {video_file}[/]")
        logger.info(f"视频文件 {video_file}")

        screen = cv2.VideoCapture(video_file)
        if not screen.isOpened():
            return logger.error(f"{const.ERR}视频文件损坏 {video_file}[/]")
        screen.release()
        logger.info(f"播放正常 {video_file}")

        reporter = Report(self.total_place)
        reporter.title = f"Model_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}"
        if not os.path.exists(reporter.query_path):
            os.makedirs(reporter.query_path, exist_ok=True)

        loop = asyncio.get_event_loop()

        video_streams = loop.run_until_complete(
            Switch.ask_video_stream(self.fpb, video_file)
        )

        rlt_frame_rate = video_streams["rlt_frame_rate"]
        avg_frame_rate = video_streams["avg_frame_rate"]
        duration = video_streams["duration"]
        original = video_streams["original"]
        logger.info(f"视频时长: [{duration}] [{Parser.parse_times(duration)}]")
        logger.info(f"视频尺寸: {list(original)}")
        logger.info(f"实际帧率: [{rlt_frame_rate}] 平均帧率: [{avg_frame_rate}] 转换帧率: [{deploy.frate}]")

        vision_start, vision_close, vision_limit = loop.run_until_complete(Switch.ask_magic_point(
            Parser.parse_mills(deploy.start),
            Parser.parse_mills(deploy.close),
            Parser.parse_mills(deploy.limit),
            duration
        ))
        vision_start: str = Parser.parse_times(vision_start)
        vision_close: str = Parser.parse_times(vision_close)
        vision_limit: str = Parser.parse_times(vision_limit)
        logger.info(f"视频剪辑: start=[{vision_start}] close=[{vision_close}] limit=[{vision_limit}]")

        video_temp_file = os.path.join(
            reporter.query_path, f"tmp_fps{deploy.frate}.mp4"
        )

        loop.run_until_complete(
            Switch.ask_video_change(
                self.fmp, [f"fps={deploy.frate}"], video_file, video_temp_file,
                start=vision_start, close=vision_close, limit=vision_limit
            )
        )

        video = VideoObject(video_temp_file)
        logger.info(f"视频帧长度: {video.frame_count} 分辨率: {video.frame_size}")
        logger.info(f"加载到内存: {video.name}")
        video_load_time = time.time()
        video.load_frames(
            scale=None, shape=None, color=True
        )
        logger.info(f"视频帧已加载: {video.frame_details(video.frames_data)}")
        logger.info(f"视频加载耗时: {time.time() - video_load_time:.2f} 秒")

        cutter = VideoCutter()

        logger.info(f"压缩视频: {video.name}")
        logger.info(f"视频帧数: {video.frame_count} 帧片段数: {video.frame_count - 1} 帧分辨率: {video.frame_size}")
        cut_start_time = time.time()
        cut_range = cutter.cut(video=video, block=deploy.block)
        logger.info(f"压缩完成: {video.name}")
        logger.info(f"压缩耗时: {time.time() - cut_start_time:.2f} 秒")

        stable, unstable = cut_range.get_range(
            threshold=deploy.thres, offset=deploy.shift
        )

        if deploy.shape:
            w, h, ratio = loop.run_until_complete(
                Switch.ask_magic_frame(original, deploy.shape)
            )
            target_shape = w, h
            target_scale = deploy.scale
            logger.info(f"调整宽高比: {w} x {h}")
        elif deploy.scale:
            target_shape = deploy.shape
            target_scale = max(0.1, min(1.0, deploy.scale))
        else:
            target_shape = deploy.shape
            target_scale = const.COMPRESS

        cut_range.pick_and_save(
            range_list=stable,
            frame_count=20,
            to_dir=reporter.query_path,
            meaningful_name=True,
            not_grey=deploy.color,
            compress_rate=target_scale,
            target_size=target_shape
        )

        os.remove(video_temp_file)

    # """Child Process"""
    @staticmethod
    def build_model(video_data: str, deploy: "Deploy"):
        if not os.path.isdir(video_data):
            return logger.error(f"{const.ERR}编译模型需要一个已经分类的文件夹[/]")

        real_path, file_list = "", []
        logger.debug(f"搜索文件夹: {video_data}")
        for root, dirs, files in os.walk(video_data, topdown=False):
            for name in files:
                file_list.append(os.path.join(root, name))
            for name in dirs:
                if len(name) == 1 and re.search(r"0", name):
                    real_path = os.path.join(root, name)
                    logger.debug(f"分类文件夹: {real_path}")
                    break

        if not real_path or len(file_list) == 0:
            return logger.error(f"{const.ERR}文件夹未正确分类[/]")

        image, image_color, image_aisle = None, "grayscale", 1
        for image_file in os.listdir(real_path):
            image_path = os.path.join(real_path, image_file)
            if not os.path.isfile(image_path):
                return logger.error(f"{const.ERR}存在无效的图像文件[/]")
            image = cv2.imread(image_path)
            logger.info(f"图像分辨率: {image.shape}")
            if image.ndim == 3:
                if numpy.array_equal(image[:, :, 0], image[:, :, 1]) and numpy.array_equal(image[:, :, 1], image[:, :, 2]):
                    logger.info(f"The image is grayscale image, stored in RGB format")
                else:
                    logger.info(f"The image is color image")
                    image_color, image_aisle = "rgb", image.ndim
            else:
                logger.info(f"The image is grayscale image")
            break

        src_model_path = os.path.dirname(real_path)
        new_model_path = os.path.join(
            src_model_path, f"Create_Model_{time.strftime('%Y%m%d%H%M%S')}", f"{random.randint(100, 999)}"
        )

        image_shape = deploy.shape if deploy.shape else image.shape
        w, h, *_ = image_shape

        name = f"Gray" if image_aisle == 1 else f"Hued"
        # new_model_name = f"Keras_{name}_W{w}_H{h}_{random.randint(10000, 99999)}.h5"
        new_model_name = f"Keras_{name}_W{w}_H{h}_{random.randint(10000, 99999)}"

        kc = KerasStruct()
        kc.build(image_color, image_shape, image_aisle, src_model_path, new_model_path, new_model_name)

    async def combines_view(self, merge: list):
        views, total = await asyncio.gather(
            Craft.achieve(self.view_share_temp), Craft.achieve(self.view_total_temp),
            return_exceptions=True
        )
        logger.info(f"正在生成汇总报告 ...")
        state_list = await asyncio.gather(
            *(Report.ask_create_total_report(m, self.group, views, total) for m in merge)
        )
        for state in state_list:
            if isinstance(state, Exception):
                logger.error(f"{const.ERR}{state}[/]")
            logger.info(f"成功生成汇总报告 {os.path.relpath(state)}")

    async def combines_main(self, merge: list):
        major, total = await asyncio.gather(
            Craft.achieve(self.main_share_temp), Craft.achieve(self.main_total_temp),
            return_exceptions=True
        )
        logger.info(f"正在生成汇总报告 ...")
        state_list = await asyncio.gather(
            *(Report.ask_create_total_report(m, self.group, major, total) for m in merge)
        )
        for state in state_list:
            if isinstance(state, Exception):
                logger.error(f"{const.ERR}{state}[/]")
            logger.info(f"成功生成汇总报告 {os.path.relpath(state)}")

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
                    f"原始尺寸: {(original_w, original_h)} 调整尺寸: {(new_w, new_h)} 缩放比例: {int(image_scale * 100)}%"
                )

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

        cmd_lines, platform, deploy, level, power, main_loop, *_ = args

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
                reporter = Report(self.total_place)
                reporter.title = f"Hooks_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
                for device, resize_img in zip(device_list, resized_result):
                    img_save_path = os.path.join(
                        reporter.query_path, f"hook_{device.sn}_{random.randint(10000, 99999)}.png"
                    )
                    resize_img.save(img_save_path)
                    logger.info(f"保存图片: {os.path.relpath(img_save_path)}")
                break
            elif action.strip().upper() == "N":
                break
            else:
                logger.warning(f"{const.WRN}没有该选项,请重新输入[/]\n")

    async def analysis(self, *args):

        async def combines():
            if len(reporter.range_list) == 0:
                return logger.warning(f"{const.WRN}没有可以生成的报告[/]")
            report = getattr(self, "combines_view" if self.speed else "combines_main")
            return await report([os.path.dirname(reporter.total_path)])

        async def commence():

            async def wait_for_device(device):
                logger.info(f"[bold #FAFAD2]Wait Device Online -> {device.tag} {device.sn}[/]")
                await Terminal.cmd_line(self.adb, "-s", device.sn, "wait-for-device")

            logger.info(f"△ △ △ {('独立' if self.alone else '全局')}控制模式 △ △ △")

            await source_monitor.monitor()

            await asyncio.gather(
                *(wait_for_device(device) for device in device_list)
            )

            media_screen_w, media_screen_h = ScreenMonitor.screen_size()
            logger.info(f"Media Screen W={media_screen_w} H={media_screen_h}")

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

                reporter.query = os.path.join(format_folder, device.sn)

                video_temp, transports = await record.ask_start_record(
                    device, reporter.video_path, location=location
                )
                todo_list.append(
                    [video_temp, transports, reporter.total_path, reporter.title, reporter.query_path,
                     reporter.query, reporter.frame_path, reporter.extra_path, reporter.proto_path]
                )

            return todo_list

        async def analysis_tactics():
            if len(task_list) == 0:
                task_list.clear()
                return logger.warning(f"{const.WRN}没有有效任务[/]")

            start_ms = Parser.parse_mills(deploy.start)
            close_ms = Parser.parse_mills(deploy.close)
            limit_ms = Parser.parse_mills(deploy.limit)
            content_list = await asyncio.gather(
                *(clipix.vision_content(video_temp, start_ms, close_ms, limit_ms, deploy.frate)
                  for video_temp, *_ in task_list)
            )
            duration_list, original_list, final_point_list = zip(*content_list)

            if self.alike and len(task_list) > 1:
                logger.info(f"视频均衡: [{(standard := min(duration_list))}] [{Parser.parse_times(standard)}]")
                await asyncio.gather(
                    *(clipix.vision_balance(duration, standard, video_idx, video_src, deploy.frate, task_list)
                      for duration, (video_idx, (video_src, *_)) in zip(
                        duration_list, enumerate(task_list)) if duration != standard)
                )

            if self.speed:
                logger.info(f"△ △ △ 快速模式 △ △ △")

                const_filter = [f"fps={deploy.frate}"] if deploy.color else [f"fps={deploy.frate}", "format=gray"]
                if deploy.shape:
                    final_shape_list = await clipix.vision_improve(
                        original_list, deploy.shape
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
                    logger.info(f"视频过滤: {flt}")

                await asyncio.gather(
                    *(clipix.pixels(
                        Switch.ask_video_detach, video_filter, video_temp, frame_path,
                        start=vision_start, close=vision_close, limit=vision_limit
                    ) for video_filter, (video_temp, *_, frame_path, _, _), (vision_start, vision_close, vision_limit)
                        in zip(video_filter_list, task_list, final_point_list))
                )

                start, end, cost, scores, struct = 0, 0, 0, None, None
                for *_, total_path, title, query_path, query, frame_path, _, _ in task_list:
                    result = {
                        "total": os.path.basename(total_path),
                        "title": title,
                        "query": query,
                        "stage": {"start": start, "end": end, "cost": cost},
                        "frame": os.path.basename(frame_path),
                        "style": "speed"
                    }
                    logger.debug(f"Speeder: {json.dumps(result, ensure_ascii=False)}")
                    await reporter.load(result)

                    self.enforce(
                        reporter.reset_path, struct, start, end, cost,
                        total_path, title, query_path, query, frame_path
                    )

            elif self.basic or self.keras:
                logger.info(f"△ △ △ {'智能模式' if alynex.kc else '基础模式'} △ △ △")

                video_target_list = [
                    (os.path.join(
                        os.path.dirname(video_temp), f"vision_fps{deploy.frate}_{random.randint(100, 999)}.mp4"
                    ), [f"fps={deploy.frate}"]) for video_temp, *_ in task_list
                ]

                await asyncio.gather(
                    *(clipix.pixels(
                        Switch.ask_video_change, video_filter, video_temp,
                        target, start=vision_start, close=vision_close, limit=vision_limit
                    ) for (target, video_filter), (video_temp, *_), (vision_start, vision_close, vision_limit)
                        in zip(video_target_list, task_list, final_point_list))
                )

                for (target, _), (video_temp, *_) in zip(video_target_list, task_list):
                    logger.info(f"视频转换完成: {os.path.basename(target)}")
                    os.remove(video_temp)
                    logger.info(f"移除旧的视频: {os.path.basename(video_temp)}")

                if len(task_list) == 1:
                    task = [
                        alynex.ask_analyzer(target, frame_path, extra_path, original, **deploy.deploys)
                        for (target, _), (*_, frame_path, extra_path, _), original
                        in zip(video_target_list, task_list, original_list)
                    ]
                    futures = await asyncio.gather(*task)

                else:
                    func = partial(self.amazing, **deploy.deploys)
                    with ProcessPoolExecutor(power, None, Active.active, ("ERROR",)) as exe:
                        task = [
                            main_loop.run_in_executor(exe, func, target, frame_path, extra_path, original)
                            for (target, _), (*_, frame_path, extra_path, _), original
                            in zip(video_target_list, task_list, original_list)
                        ]
                        futures = await asyncio.gather(*task)

                for future, todo in zip(futures, task_list):
                    if future is None:
                        continue

                    start, end, cost, scores, struct = future.material
                    *_, total_path, title, query_path, query, frame_path, extra_path, proto_path = todo

                    result = {
                        "total": os.path.basename(total_path),
                        "title": title,
                        "query": query,
                        "stage": {"start": start, "end": end, "cost": f"{cost:.5f}"},
                        "frame": os.path.basename(frame_path)
                    }

                    if struct:
                        if isinstance(
                                tmp := await Craft.achieve(self.atom_total_temp), Exception):
                            return logger.error(f"{const.ERR}{tmp}[/]")
                        logger.info(f"模版引擎正在渲染 ...")
                        original_inform = await reporter.ask_draw(
                            scores, struct, proto_path, tmp, deploy.boost
                        )
                        logger.info(f"模版引擎渲染完毕 {os.path.relpath(original_inform)}")
                        result["extra"] = os.path.basename(extra_path)
                        result["proto"] = os.path.basename(original_inform)
                        result["style"] = "keras"
                    else:
                        result["style"] = "basic"

                    logger.debug(f"Restore: {json.dumps(result, ensure_ascii=False)}")
                    await reporter.load(result)

                    self.enforce(
                        reporter.reset_path, struct, start, end, cost,
                        total_path, title, query_path, query, frame_path, extra_path, proto_path
                    )

            else:
                return logger.info(f"△ △ △ 录制模式 △ △ △")

        async def anything_time():
            await asyncio.gather(
                *(record.check_timer(device, timer_mode) for device in device_list)
            )

        async def anything_over():
            effective_list = await asyncio.gather(
                *(record.ask_close_record(video_temp, transports, device)
                  for (video_temp, transports, *_), device in zip(task_list, device_list))
            )
            for idx, (effective, video_name) in enumerate(effective_list):
                if "视频录制失败" in effective:
                    task = task_list.pop(idx)
                    logger.warning(f"{const.WRN}{effective}: {video_name} 移除: {os.path.basename(task[0])}")
                else:
                    logger.success(f"{const.SUC}{effective}: {video_name}")

        async def call_commands(exec_func, exec_args, bean, live_devices):
            if not (callable(function := getattr(bean, exec_func, None))):
                return logger.error(f"{const.ERR}No callable {exec_func}[/]")

            sn = getattr(bean, 'sn', bean.__class__.__name__)
            try:
                logger.info(f"{sn} {function.__name__} {exec_args}")
                if inspect.iscoroutinefunction(function):
                    return await function(*exec_args)
                return await asyncio.to_thread(function, *exec_args)
            except asyncio.CancelledError:
                live_devices.pop(sn)
                logger.info(f"[bold #CD853F]{sn} Call Commands Exit[/]")
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
                    return logger.info(f"[bold #F0FFF0 on #000000]All tasks canceled[/]")
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
                    return logger.info(f"[bold #F0FFF0 on #000000]All tasks canceled[/]")
                finally:
                    exec_tasks.clear()

                for status in exec_status_list:
                    if isinstance(status, Exception):
                        logger.error(f"{const.ERR}{status}[/]")

            for stop in stop_tasks:
                stop.cancel()

        # Initialization ===============================================================================================
        cmd_lines, platform, deploy, level, power, main_loop, *_ = args
        clipix = Clipix(self.fmp, self.fpb)

        manage_ = Manage(self.adb)
        device_list = await manage_.operate_device()

        titles_ = {"speed": "Speed", "basic": "Basic", "keras": "Keras"}
        input_title_ = next((title for key, title in titles_.items() if getattr(self, key)), "Video")
        reporter = Report(self.total_place)

        if self.keras and not self.speed and not self.basic:
            attack_ = self.total_place, self.model_place
        else:
            attack_ = self.total_place, None

        alynex = Alynex(*attack_)
        if alynex.kc:
            try:
                channel_ = alynex.kc.model.input_shape[-1]
                if deploy.color:
                    assert channel_ == 3, f"彩色模式需要匹配彩色模型 Model Color Channel={channel_}"
                else:
                    assert channel_ == 1, f"灰度模式需要匹配灰度模型 Model Color Channel={channel_}"
            except AssertionError as e_:
                logger.error(f"{const.ERR}{e_}[/]")
                alynex.kc = None

        record = Record(
            self.scc, platform, alone=self.alone, whist=self.whist, frate=deploy.frate
        )
        player = Player()
        source_monitor = SourceMonitor()
        # Initialization ===============================================================================================

        # Flick Loop ===================================================================================================
        if self.flick:
            const_title_ = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
            reporter.title = f"{input_title_}_{const_title_}"
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
                                    logger.success(f"{const.SUC}New title set successfully[/]")
                                    reporter.title = f"{src_hd_}_{hd_}" if hd_ else f"{src_hd_}_{random.randint(a_, b_)}"
                                    continue
                            raise FramixAnalysisError(f"Set Error")
                        elif select_ in ["invent", "create"]:
                            await combines()
                            break
                        elif select_ == "deploy":
                            logger.warning(f"{const.WRN}请完全退出编辑器再继续操作[/]")
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
                                logger.info(f"[bold #FFFF87]{bound_tips_}[/]")
                            timer_mode = max(lower_bound_, min(upper_bound_, timer_value_))
                        else:
                            raise FramixAnalysisError(f"Set Error")
                except FramixAnalysisError as e_:
                    logger.error(f"{const.ERR}{e_}[/]")
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
        # Flick Loop ===================================================================================================

        # Other Loop ===================================================================================================
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
                return None

            for script_data_ in load_script_data_:
                if isinstance(script_data_, Exception):
                    raise FramixAnalysisError(script_data_)
            script_storage_ = [script_data_ for script_data_ in load_script_data_]

            await manage_.display_device()
            for script_dict_ in script_storage_:
                for script_key_, script_value_ in script_dict_.items():
                    logger.info(f"Batch Exec: {script_key_}")

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
                        logger.error(f"{const.ERR}{e_}[/]")
                        logger.error(f"{const.ERR}重置循环次数[/] {(looper_ := 1)}")

                    if prefix_list_ := script_value_.get("prefix", []):
                        prefix_list_ = await pack_commands(prefix_list_)
                    if action_list_ := script_value_.get("action", []):
                        action_list_ = await pack_commands(action_list_)
                    if suffix_list_ := script_value_.get("suffix", []):
                        suffix_list_ = await pack_commands(suffix_list_)

                    for hd_ in header_:
                        reporter.title = f"{input_title_}_{script_key_}_{hd_}"
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

            return await combines()
        # Other Loop ===================================================================================================

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
            frate: int
    ) -> tuple[float, tuple[int, int], tuple[typing.Optional[str], typing.Optional[str], typing.Optional[str]]]:

        video_streams = await Switch.ask_video_stream(self.fpb, video_temp)

        rlt_frame_rate = video_streams["rlt_frame_rate"]
        avg_frame_rate = video_streams["avg_frame_rate"]
        duration = video_streams["duration"]
        original = video_streams["original"]
        logger.info(f"视频时长: [{duration}] [{Parser.parse_times(duration)}]")
        logger.info(f"视频尺寸: {list(original)}")
        logger.info(f"实际帧率: [{rlt_frame_rate}] 平均帧率: [{avg_frame_rate}] 转换帧率: [{frate}]")

        vision_start, vision_close, vision_limit = await Switch.ask_magic_point(
            Parser.parse_mills(start),
            Parser.parse_mills(close),
            Parser.parse_mills(limit),
            duration
        )
        vision_start: str = Parser.parse_times(vision_start)
        vision_close: str = Parser.parse_times(vision_close)
        vision_limit: str = Parser.parse_times(vision_limit)
        logger.info(f"视频剪辑: start=[{vision_start}] close=[{vision_close}] limit=[{vision_limit}]")

        return duration, original, (vision_start, vision_close, vision_limit)

    async def vision_balance(
            self, duration: float, standard: float, idx: int, src: str, frate: float, task_list: list[list]
    ):

        start_time_point = duration - standard
        end_time_point = duration
        start_time_str = str(datetime.timedelta(seconds=start_time_point))
        end_time_str = str(datetime.timedelta(seconds=end_time_point))

        logger.info(f"{os.path.basename(src)} [{duration}] [{start_time_str} - {end_time_str}]")
        video_dst = os.path.join(
            os.path.dirname(src), f"tailor_fps{frate}_{random.randint(100, 999)}.mp4"
        )

        await Switch.ask_video_tailor(
            self.fmp, src, video_dst, start=start_time_str, limit=end_time_str
        )

        try:
            os.remove(src)
        except FileNotFoundError:
            pass
        logger.info(f"Balance complete {os.path.basename(src)}")
        task_list[idx][0] = video_dst

    @staticmethod
    async def vision_improve(
            original_list: list[tuple[int, int]], shape: tuple
    ):

        final_shape_list = await asyncio.gather(
            *(Switch.ask_magic_frame(original, shape) for original in original_list)
        )
        return final_shape_list

    async def pixels(
            self, function: "typing.Callable", video_filter: list, src: str, dst: str, **kwargs
    ):

        return await function(self.fmp, video_filter, src, dst, **kwargs)


class Alynex(object):

    __kc: typing.Optional["KerasStruct"] = None

    def __init__(
            self,
            total_place: typing.Optional[str] = None,
            model_place: typing.Optional[str] = None,
    ):

        self.total_place = total_place
        self.model_place = model_place

        if self.total_place and self.model_place:
            try:
                self.kc = KerasStruct()
                self.kc.load_model(model_place)
            except Exception as e:
                logger.error(f"{const.ERR}Keras sequence model load fail[/]")
                logger.error(f"{e}")
                self.kc = None

    @property
    def kc(self) -> typing.Optional["KerasStruct"]:
        return self.__kc

    @kc.setter
    def kc(self, value):
        self.__kc = value

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def ask_analyzer(self, vision: str, *args, **kwargs) -> typing.Optional["Review"]:

        frame_path, extra_path, original, *_ = args

        boost = kwargs.get("boost", const.BOOST)
        color = kwargs.get("color", const.COLOR)

        shape = kwargs.get("shape", const.SHAPE)
        scale = kwargs.get("scale", const.SCALE)
        _ = kwargs.get("start", const.START)
        _ = kwargs.get("close", const.CLOSE)
        _ = kwargs.get("limit", const.LIMIT)

        begin = kwargs.get("begin", const.BEGIN)
        final = kwargs.get("final", const.FINAL)

        crops = kwargs.get("crops", const.CROPS)
        omits = kwargs.get("omits", const.OMITS)

        _ = kwargs.get("frate", const.FRATE)
        thres = kwargs.get("thres", const.THRES)
        shift = kwargs.get("shift", const.SHIFT)
        block = kwargs.get("block", const.BLOCK)

        async def frame_check():
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
            begin_stage_index, begin_frame_index = begin
            final_stage_index, final_frame_index = final
            logger.info(
                f"Extract frames begin={list(begin)} final={list(final)}"
            )

            try:
                logger.info(f"阶段划分: {struct.get_ordered_stage_set()}")
                unstable_stage_range = struct.get_not_stable_stage_range()
                begin_frame = unstable_stage_range[begin_stage_index][begin_frame_index]
                final_frame = unstable_stage_range[final_stage_index][final_frame_index]
            except (AssertionError, IndexError) as e:
                logger.error(f"{const.ERR}{e}[/]")
                logger.warning(f"{const.WRN}Analyzer Neural Engine is recalculating ...[/]")
                begin_frame = struct.get_important_frame_list()[0]
                final_frame = struct.get_important_frame_list()[-1]

            if final_frame.frame_id <= begin_frame.frame_id:
                logger.warning(f"{const.WRN}{final_frame} <= {begin_frame}[/]")
                logger.warning(f"{const.WRN}Analyzer Neural Engine is recalculating ...[/]")
                begin_frame, end_frame = struct.data[0], struct.data[-1]

            time_cost = final_frame.timestamp - begin_frame.timestamp
            logger.info(
                f"分类结果: [开始帧: {begin_frame.timestamp:.5f}] [结束帧: {final_frame.timestamp:.5f}] [总耗时: {time_cost:.5f}]"
            )
            return begin_frame.frame_id, final_frame.frame_id, time_cost

        async def frame_flip():
            if shape:
                w, h, ratio = await Switch.ask_magic_frame(original, shape)
                target_shape = w, h
                target_scale = scale
                logger.info(f"调整宽高比: {w} x {h}")
            elif scale:
                target_shape = shape
                target_scale = max(0.1, min(1.0, scale))
            else:
                target_shape = shape
                target_scale = 0.4

            return target_shape, target_scale

        async def frame_hold():
            if struct is None:
                return [i for i in video.frames_data]

            logger.info(f"{'△ △ △ 获取关键帧 △ △ △' if boost else '△ △ △ 获取全部帧 △ △ △'}")
            frames_list = []
            important_frames = struct.get_important_frame_list()
            if boost:
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

            cutter = VideoCutter()

            size_hook = FrameSizeHook(1.0, None, True)
            cutter.add_hook(size_hook)
            logger.info(
                f"视频帧处理: {size_hook.__class__.__name__} "
                f"{[size_hook.compress_rate, size_hook.target_size, size_hook.not_grey]}"
            )

            if len(crop_list := crops) > 0 and sum([j for i in crop_list for j in i.values()]) > 0:
                for crop in crop_list:
                    x, y, x_size, y_size = crop.values()
                    crop_hook = PaintCropHook((y_size, x_size), (y, x))
                    cutter.add_hook(crop_hook)
                    logger.info(
                        f"视频帧处理: {crop_hook.__class__.__name__} "
                        f"{x, y, x_size, y_size}"
                    )

            if len(omit_list := omits) > 0 and sum([j for i in omit_list for j in i.values()]) > 0:
                for omit in omit_list:
                    x, y, x_size, y_size = omit.values()
                    omit_hook = PaintOmitHook((y_size, x_size), (y, x))
                    cutter.add_hook(omit_hook)
                    logger.info(
                        f"视频帧处理: {omit_hook.__class__.__name__} "
                        f"{x, y, x_size, y_size}"
                    )

            save_hook = FrameSaveHook(extra_path)
            cutter.add_hook(save_hook)
            logger.info(
                f"视频帧处理: {save_hook.__class__.__name__} "
                f"{[os.path.basename(extra_path)]}"
            )

            logger.info(f"压缩视频: {video.name}")
            logger.info(f"视频帧数: {video.frame_count} 片段数: {video.frame_count - 1} 分辨率: {video.frame_size}")
            cut_start_time = time.time()
            cut_range = cutter.cut(video=video, block=block)
            logger.info(f"压缩完成: {video.name}")
            logger.info(f"压缩耗时: {time.time() - cut_start_time:.2f} 秒")

            stable, unstable = cut_range.get_range(threshold=thres, offset=shift)

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

            struct_start_time = time.time()
            try:
                struct_data = self.kc.classify(
                    video=video, valid_range=stable, keep_data=True
                )
            except AssertionError as e:
                return logger.error(f"{const.ERR}{e}[/]")

            logger.info(f"分类耗时: {time.time() - struct_start_time:.2f} 秒")
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

        # Start
        if (target_record := await frame_check()) is None:
            return logger.warning(f"{const.WRN}视频文件损坏: {os.path.basename(vision)}[/]")
        logger.info(f"开始加载视频: {os.path.basename(target_record)}")

        shape, scale = await frame_flip()
        video_load_time = time.time()
        video = VideoObject(target_record)
        logger.info(f"视频帧长度: {video.frame_count} 分辨率: {video.frame_size}")
        logger.info(f"加载到内存: {video.name}")
        video.load_frames(
            scale=scale, shape=shape, color=color
        )
        logger.info(f"视频帧已加载: {video.frame_details(video.frames_data)}")
        logger.info(f"视频加载耗时: {time.time() - video_load_time:.2f} 秒")

        struct = await frame_flow() if self.kc else None
        frames = await frame_hold()

        if struct:
            return Review(*(await analytics_keras()))
        return Review(*(await analytics_basic()))


async def distribute(function: "typing.Callable", parameters: list[str]) -> None:

    proc = members if (members := len(parameters)) <= _power else _power
    rank = "ERROR" if members > 1 else _level

    parameters = [(await Craft.revise_path(param), _deploy) for param in parameters]

    with Pool(proc, Active.active, (rank,)) as pool:
        async_task = pool.starmap_async(function, parameters)
        async_task.get()

    sys.exit(Show.done())


async def arithmetic(function: "typing.Callable", parameters: list[str]) -> None:
    try:
        await function(
            [(await Craft.revise_path(param)) for param in parameters],
            _cmd_lines, _platform, _deploy, _level, _power, _main_loop
        )
    except (FramixAnalysisError, FramixAnalyzerError, FramixReporterError):
        Show.console.print_exception()
        sys.exit(Show.fail())


async def scheduling() -> None:
    try:
        # --flick --carry --fully
        if _cmd_lines.flick or _cmd_lines.carry or _cmd_lines.fully:
            await _missions.analysis(
                _cmd_lines, _platform, _deploy, _level, _power, _main_loop
            )
        # --paint
        elif _cmd_lines.paint:
            await _missions.painting(
                _cmd_lines, _platform, _deploy, _level, _power, _main_loop
            )
        # --union
        elif _cmd_lines.union:
            await _missions.combines_view(_cmd_lines.union)
        # --merge
        elif _cmd_lines.merge:
            await _missions.combines_main(_cmd_lines.merge)
        else:
            Show.help_document()
    except (FramixAnalysisError, FramixAnalyzerError, FramixReporterError):
        Show.console.print_exception()
        sys.exit(Show.fail())


if __name__ == '__main__':
    _cmd_lines = Parser.parse_cmd()

    Active.active(_level := "DEBUG" if _cmd_lines.debug else "INFO")

    logger.debug(f"操作系统: {_platform}")
    logger.debug(f"应用名称: {_software}")
    logger.debug(f"系统路径: {_sys_symbol}")
    logger.debug(f"环境变量: {_env_symbol}")
    logger.debug(f"工具路径: {_turbo}")
    logger.debug(f"命令参数: {_lines}")
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

    _flick = _cmd_lines.flick
    _carry = _cmd_lines.carry
    _fully = _cmd_lines.fully

    _speed = _cmd_lines.speed
    _basic = _cmd_lines.basic
    _keras = _cmd_lines.keras

    _alone = _cmd_lines.alone
    _whist = _cmd_lines.whist

    _alike = _cmd_lines.alike

    _group = _cmd_lines.group

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
        if any(_line.startswith(f"--{_attr}") for _line in _lines):
            setattr(_deploy, _attr, getattr(_cmd_lines, _attr))
            logger.debug(f"Initialize Set <{_attr}> {_attribute} -> {getattr(_deploy, _attr)}")

    _missions = Missions(
        _flick, _carry, _fully, _speed, _basic, _keras, _alone, _whist, _alike, _group,
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
    from multiprocessing import Pool
    from concurrent.futures import ProcessPoolExecutor

    Show.load_animation()

    # Main Process =====================================================================================================
    try:
        # --video
        if _video_list := _cmd_lines.video:
            _main_loop.run_until_complete(
                arithmetic(_missions.video_file_task, _video_list)
            )
        # --stack
        elif _stack_list := _cmd_lines.stack:
            _main_loop.run_until_complete(
                arithmetic(_missions.video_data_task, _stack_list)
            )
        # --train
        elif _train_list := _cmd_lines.train:
            _main_loop.run_until_complete(
                distribute(_missions.train_model, _train_list)
            )
        # --build
        elif _build_list := _cmd_lines.build:
            _main_loop.run_until_complete(
                distribute(_missions.build_model, _build_list)
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
    # Main Process =====================================================================================================
