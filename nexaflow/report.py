#
#   ____                       _
#  |  _ \ ___ _ __   ___  _ __| |_
#  | |_) / _ \ '_ \ / _ \| '__| __|
#  |  _ <  __/ |_) | (_) | |  | |_
#  |_| \_\___| .__/ \___/|_|   \__|
#            |_|
#

import os
import re
import json
import time
import shutil
import typing
import random
import asyncio
import aiofiles
from pathlib import Path
from loguru import logger
from jinja2 import Template
from collections import defaultdict
from engine.tinker import FramixError
from nexaflow import (
    toolbox, const
)
from nexaflow.classifier.base import ClassifierResult


class Report(object):
    """
    用于管理和生成分析报告的核心类。

    该类封装了从分析结果数据结构到最终 HTML 报告文件的整个流程，支持单个分析报告的构建、
    多报告的合并、以及生成带有图像与统计信息的总汇总报告。其功能主要包括：

    - 初始化报告输出路径结构与日志路径；
    - 管理报告标题与查询信息；
    - 加载报告片段信息；
    - 生成单个分析报告、创建汇总报告；
    - 绘制分析图表并生成 HTML 文件。

    该类依赖 aiofiles 进行异步文件操作，Jinja2 用于渲染 HTML 模板，并通过日志记录操作流程。
    报告内容来源于 `ClassifierResult` 结构，帧图像来源于保存路径或 Base64 格式处理。
    """

    def __init__(self, total_path: str):
        """
        初始化 Report 实例并创建目录结构与日志文件。

        Parameters
        ----------
        total_path : str
            指定的根目录路径，用于存放分析报告的总目录。

        Notes
        -----
        - 创建总报告目录路径（包含时间戳与进程号）与恢复目录；
        - 初始化 query_path、video_path、frame_path、extra_path 等路径为空；
        - 设置 `range_list` 和 `total_list` 用于存储分析数据；
        - 设置日志输出路径至恢复目录中的日志文件，并使用 logger 记录后续分析事件；
        - 所有生成路径均确保目录存在（os.makedirs + exist_ok=True）。

        Workflow
        --------
        1. 生成带时间戳的唯一输出目录；
        2. 初始化输出路径（视频、帧图、附加图）为空；
        3. 创建恢复路径和日志记录文件；
        4. 初始化报告内容缓存结构。
        """
        self.__title: str = ""
        self.__query: str = ""

        self.query_path: str = ""
        self.video_path: str = ""
        self.frame_path: str = ""
        self.extra_path: str = ""

        self.range_list: list = []
        self.total_list: list = []

        _tms, _pid = time.strftime("%Y%m%d%H%M%S"), os.getpid()

        self.total_path = os.path.join(
            total_path, f"{const.R_TOTAL_TAG}_{_tms}_{_pid}", const.R_COLLECTION
        )
        os.makedirs(self.total_path, exist_ok=True)

        self.reset_path = os.path.join(
            os.path.dirname(self.total_path), const.R_RECOVERY
        )
        os.makedirs(self.reset_path, exist_ok=True)

        log_papers = os.path.join(
            self.reset_path, const.R_LOG_FILE
        )
        logger.add(log_papers, level=const.NOTE_LEVEL, format=const.WRITE_FORMAT)

    @property
    def proto_path(self) -> str:
        return os.path.join(self.query_path, self.query)

    @property
    def title(self) -> str:
        return self.__title

    @title.setter
    def title(self, title: str):
        self.__title = title
        self.query_path = os.path.join(self.total_path, self.title)
        os.makedirs(self.query_path, exist_ok=True)

    @title.deleter
    def title(self):
        del self.__title

    @property
    def query(self):
        return self.__query

    @query.setter
    def query(self, query: str):
        self.__query = query
        self.video_path = os.path.join(self.query_path, self.query, "video")
        self.frame_path = os.path.join(self.query_path, self.query, "frame")
        self.extra_path = os.path.join(self.query_path, self.query, "extra")
        os.makedirs(self.video_path, exist_ok=True)
        os.makedirs(self.frame_path, exist_ok=True)
        os.makedirs(self.extra_path, exist_ok=True)

    @query.deleter
    def query(self):
        del self.__query

    async def load(self, inform: dict) -> None:
        """
        将分析信息追加到 `range_list` 列表中。

        Parameters
        ----------
        inform : dict
            单个分析阶段的数据结构，包含 stage、frame、extra 等信息。

        Notes
        -----
        该方法主要用于单报告内容的记录，可被多次调用以汇总阶段信息。
        """
        if inform:
            self.range_list.append(inform)

    @staticmethod
    async def ask_merge_report(
            merge_list: typing.Union[list, tuple], template_file: str
    ) -> str:
        """
        合并多个分析报告，生成汇总 HTML 文件。

        该方法从多个已完成的分析报告中提取恢复日志内容，并将相关目录合并至一个新目录中，最终根据模板生成一个汇总 HTML 报告。

        Parameters
        ----------
        merge_list : list or tuple
            含有多个报告输出目录路径的列表或元组，每个路径都应包含 recovery 日志。

        template_file : str
            用于渲染汇总报告的 Jinja2 模板文件路径。

        Returns
        -------
        str
            最终生成的汇总 HTML 报告的文件路径。

        Raises
        ------
        FramixError
            - 如果合并列表为空。
            - 如果日志提取失败或模板渲染出错。
            - 如果复制目录失败或日志内容无效。

        Notes
        -----
        - 每个目录应包含 `R_RECOVERY/R_LOG_FILE` 文件，且日志中包含 `Recovery:` 开头的 JSON 结构。
        - 合并操作会创建新目录 `R_UNION_TAG_<timestamp>` 用于收集所有内容。
        - 报告合并时会忽略 `.log`、`.db` 文件以及总汇文件，避免污染新目录。
        - 生成的 HTML 会统一写入新合并目录的上一级，文件名为 `R_UNION_FILE`。

        Workflow
        --------
        1. 依次提取每个日志文件中以 `Recovery:` 开头的 JSON 内容；
        2. 合并所有目录内容（排除日志和数据库文件）到新合并目录；
        3. 使用传入的模板渲染 HTML 内容；
        4. 将渲染结果写入最终的汇总 HTML 文件；
        5. 返回该 HTML 文件的完整路径。
        """

        async def assemble(file):
            async with aiofiles.open(file, mode="r", encoding=const.CHARSET) as recovery_file:
                return re.findall(r"(?<=Recovery: ).*}", await recovery_file.read())

        log_file = const.R_RECOVERY, const.R_LOG_FILE
        if not (log_file_list := [os.path.join(merge, *log_file) for merge in merge_list]):
            raise FramixError(f"没有可以合并的报告 ...")

        merge_name = f"{const.R_UNION_TAG}_{(merge_time := time.strftime('%Y%m%d%H%M%S'))}", const.R_COLLECTION
        if not os.path.exists(
            merge_path := os.path.join(os.path.dirname(merge_list[0]), *merge_name).format()
        ):
            os.makedirs(merge_path, exist_ok=True)

        merge_log_list = await asyncio.gather(
            *(assemble(log) for log in log_file_list), return_exceptions=True
        )

        ignore = const.R_TOTAL_FILE, ".log", ".db"
        for m in merge_list:
            if isinstance(m, Exception):
                raise FramixError(m)
            shutil.copytree(
                os.path.join(m, const.R_COLLECTION).format(),
                merge_path,
                ignore=shutil.ignore_patterns(*ignore),
                dirs_exist_ok=True
            )

        if not (total_list := [json.loads(i) for logs in merge_log_list for i in logs if i]):
            raise FramixError(f"没有可以合并的报告 ...")

        async with aiofiles.open(template_file, "r", encoding=const.CHARSET) as f:
            template_open = await f.read()

        html = Template(template_open).render(
            head=const.R_TOTAL_HEAD, report_time=merge_time, total_list=total_list
        )

        report_html = os.path.join(os.path.dirname(merge_path), const.R_UNION_FILE)
        async with aiofiles.open(report_html, "w", encoding=const.CHARSET) as f:
            await f.write(html)

        return report_html

    @staticmethod
    async def ask_create_report(
            total: "Path", title: str, serial: str, parts_list: list, style_loc: str
    ) -> typing.Optional[dict]:
        """
        生成单个分析报告的 HTML 文件。

        该方法负责将某个分析任务的图像、原型、阶段数据等汇总并生成 HTML 格式的可视化报告，用于回溯和比对。

        Parameters
        ----------
        total : Path
            报告的根路径。

        title : str
            当前报告的标题名，将作为文件夹名称创建子目录。

        serial : str
            用于标识报告组的编号，如果为空则自动生成随机编号。

        parts_list : list
            包含多组图像路径、阶段信息、原型等数据的列表，每一项为一个字典。

        style_loc : str
            Jinja2 模板路径，用于渲染最终的 HTML 报告页面。

        Returns
        -------
        dict or None
            如果成功生成报告，返回包含汇总信息的字典（如耗时、跳转链接等），否则返回 None。

        Raises
        ------
        ZeroDivisionError
            当 parts_list 中所有耗时为 0 时，平均耗时计算除零异常。

        Notes
        -----
        - 每个 parts_list 字典需包含字段 `query`、`frame`、`extra`、`proto` 和 `stage`；
        - 图像名称需可提取帧 ID 和时间戳信息，否则跳过；
        - 报告将保存为 HTML 文件，位于 `total/title/title_serial.html`；
        - 报告的元数据会以 JSON 格式记录在日志中。

        Workflow
        --------
        1. 遍历 `parts_list`，提取每组图像路径、原型文件和阶段标记；
        2. 组织所有图片和元数据，生成渲染上下文；
        3. 使用 Jinja2 渲染 HTML 模板；
        4. 保存生成的 HTML 文件至对应目录；
        5. 构造并返回该报告的描述信息（耗时、路径、编号等）。
        """

        async def acquire(query: str, frame: typing.Optional[str]) -> typing.Optional[list[dict]]:
            """
            解析指定目录中的图像文件，提取帧 ID 与时间戳信息。

            根据图像文件名提取帧编号与时间戳，构造统一格式的字典结构，
            并按帧编号进行排序，用于报告中图像内容的展示。
            """
            if not frame:
                return None

            frame_list = []
            for image in os.listdir(os.path.join(total, title, query, frame)):
                image_src = os.path.join(query, frame, image)

                # Note
                # frame_00322.png
                # 79_1.30000.png
                # 219(3_6333333333333333).png
                if fit_id := re.search(r"(?<=frame_)\d+", image):
                    image_idx = fit_id.group()
                elif fit_id := re.search(r"^(?!frame)(?!.*\().*?(\d+)(?=_)", image):
                    image_idx = fit_id.group()
                else:
                    image_idx = image.split("(")[0]

                image_source = {
                    "src": image_src, "frames_id": int(image_idx)
                }

                if fit_time := re.search(r"(?<=_)\d+\.\d+(?=\.)", image):
                    image_source["timestamp"] = fit_time.group()

                frame_list.append(image_source)

            frame_list.sort(key=lambda x: x["frames_id"])
            return frame_list

        async def transform(inform_part: dict) -> list[dict]:
            """
            转换报告数据字典，整合图像与阶段信息。

            解析 inform_part 中的 frame、extra、stage、proto 字段，
            并通过 acquire 加载图像数据，构造统一结构以供模板渲染使用。
            """
            query = inform_part.get("query", "")
            stage = inform_part.get("stage", {})
            frame = inform_part.get("frame", "")
            extra = inform_part.get("extra", "")
            proto = inform_part.get("proto", "")

            image_list, extra_list = await asyncio.gather(
                *(acquire(query, i) for i in [frame, extra])
            )

            return [
                {
                    "query": query,
                    "stage": stage,
                    "image_list": image_list,
                    "extra_list": extra_list,
                    "proto": os.path.join(query, proto) if proto else None
                }
            ]

        if not parts_list:
            return None

        transform_result = await asyncio.gather(
            *(transform(parts) for parts in parts_list)
        )
        images_list = [i for result in transform_result for i in result]

        html_temp = Template(style_loc).render(
            name=const.NAME, title=title, images_list=images_list
        )
        team = serial if serial else random.randint(10000, 99999)
        html_path = Path(os.path.join(total, title, f"{title}_{team}.html"))
        async with aiofiles.open(html_path, "w", encoding=const.CHARSET) as range_file:
            await range_file.write(html_temp)

        cost_list = [cost["stage"]["cost"] for cost in images_list]
        try:
            avg = sum(map(float, cost_list)) / len(cost_list)
        except ZeroDivisionError:
            avg = 0.00000

        href = os.path.join(total.name, title, html_path.name)
        single = {
            "case": title, "team": team, "cost_list": cost_list, "avg": f"{avg:.5f}", "href": href
        }
        logger.debug("Recovery: " + json.dumps(single, ensure_ascii=False))

        return single

    @staticmethod
    async def ask_create_total_report(
            file_name: str, group: bool, style_loc: str, total_loc: str
    ) -> typing.Union[typing.Any, str]:
        """
        汇总多个单项分析报告，生成最终总报告 HTML。

        该方法负责从多个单项报告的日志文件中提取数据，合并处理后生成完整的汇总页面。支持分组模式以生成分项目归类的展示结构。

        Parameters
        ----------
        file_name : str
            指向汇总目标目录的路径，包含所有子报告及其日志文件。

        group : bool
            是否按 query 名称进行分组处理，如果为 True 则将同一 query 下的多个分析任务归为一组。

        style_loc : str
            单项报告模板路径，用于渲染每个分析小项的 HTML 页面。

        total_loc : str
            总报告模板路径，用于渲染最终的 HTML 汇总文件。

        Returns
        -------
        str
            返回最终生成的总报告 HTML 文件路径。

        Raises
        ------
        FramixError
            - 当日志文件不存在或无法读取；
            - 当未能提取出任何有效数据或报告内容为空；
            - 当报告渲染过程中发生异常。

        Notes
        -----
        - 此方法仅处理符合格式的 Recovery 日志；
        - 会遍历日志中记录的所有分项任务，自动调用 `ask_create_report` 生成子报告；
        - 合并结果最终会调用 Jinja2 模板生成汇总页面并写入至文件系统。

        Workflow
        --------
        1. 读取 `file_name` 目录下日志文件中记录的任务；
        2. 从日志中提取 Speeder / Restore 信息；
        3. 若开启分组，按 `(total, title, query)` 对数据进行归类；
        4. 针对每个分析项目，调用 `ask_create_report` 生成 HTML 报告；
        5. 合并每个子报告的摘要信息，组织为总表结构；
        6. 使用 Jinja2 渲染总报告模板；
        7. 写入最终总报告文件至指定位置并返回路径。
        """
        try:
            log_file = const.R_RECOVERY, const.R_LOG_FILE
            async with aiofiles.open(os.path.join(file_name, *log_file), "r", encoding=const.CHARSET) as f:
                open_file = await f.read()
        except FileNotFoundError as e:
            raise FramixError(e)

        if match_speeder_list := re.findall(r"(?<=Speeder: ).*}", open_file):
            match_list = match_speeder_list
        elif match_restore_list := re.findall(r"(?<=Restore: ).*}", open_file):
            match_list = match_restore_list
        else:
            raise FramixError(f"没有符合条件的数据 ...")

        parts_list: list[dict] = [
            json.loads(file) for file in match_list if file
        ]
        parts_list: list[dict] = [
            {(p.pop("total"), p.pop("title"), Path(p["query"]).name if group else ""): p} for p in parts_list
        ]

        async def format_packed() -> dict:
            """
            将报告信息按 (total, title, serial) 进行分组打包。

            遍历 parts_list 中的数据，根据三元组键聚合各字段内容，
            生成格式统一的嵌套字典结构，用于后续创建单份报告。
            """
            parts_dict = defaultdict(lambda: defaultdict(list))

            for parts in parts_list:
                for key, value in parts.items():
                    for k, v in value.items():
                        parts_dict[key][k].append(v)
            normal_dict = {k: dict(v) for k, v in parts_dict.items()}

            return {
                k: [dict(zip(v.keys(), entry)) for entry in zip(*v.values())] for k, v in normal_dict.items()
            }

        async def format_merged() -> list[dict]:
            """
            合并创建结果中的字段内容，生成汇总报告的格式化结构。

            根据 case 与 team 组合键对报告字段聚合，构建统一结构的阶段信息，
            用于绘制总报告中每个实验条目的合并图表与详细数据。
            """
            parts_dict, segment = defaultdict(lambda: defaultdict(list)), "<@@@>"

            for rp in create_total_result:
                for key, value in rp.items():
                    if key not in ["case", "team"]:
                        parts_dict[f"{rp['case']}{segment}{rp['team']}"][key].append(value)

            return [
                {
                    "case": keys.split(f"{segment}")[0],
                    "team": keys.split(f"{segment}")[1],
                    **{k: v for k, v in attrs.items()}
                } for keys, attrs in parts_dict.items()
            ]

        packed_dict = await format_packed()
        for detail in packed_dict.values():
            logger.debug(f"{detail}")

        try:
            create_result = await asyncio.gather(
                *(Report.ask_create_report(
                    Path(os.path.join(file_name, total)), title, sn, result_dict, style_loc)
                    for (total, title, sn), result_dict in packed_dict.items())
            )
        except Exception as e:
            raise FramixError(e)

        create_total_result = [create for create in create_result if create]
        merged_list = await format_merged()

        for i in merged_list:
            i["merge_list"] = list(zip(i.pop("href"), i.pop("avg"), i.pop("cost_list")))
            for j in i["merge_list"]:
                logger.debug(f"{j}")

        if not (total_list := [single for single in merged_list if single]):
            raise FramixError(f"没有可以汇总的报告 ...")

        total_html_temp = Template(total_loc).render(
            head=const.R_TOTAL_HEAD, report_time=time.strftime('%Y.%m.%d %H:%M:%S'), total_list=total_list
        )
        total_html = os.path.join(file_name, const.R_TOTAL_FILE)
        async with aiofiles.open(total_html, "w", encoding=const.CHARSET) as f:
            await f.write(total_html_temp)

        return total_html

    @staticmethod
    async def ask_draw(
            scores: dict, struct_result: "ClassifierResult", proto_path: str, template_file: str, boost: bool,
    ) -> typing.Union[typing.Any, str]:
        """
        绘制阶段图像摘要报告，生成 HTML 格式的图像展示页面。

        本方法根据结构分类结果和图像评分信息，构造出稳定与不稳定阶段的缩略图数据，并填充 HTML 模板，生成完整可视化报告页面。

        Parameters
        ----------
        scores : dict
            图像评分结果字典，键为 frame_id，值为对应帧图像的 base64 编码或图像路径。

        struct_result : ClassifierResult
            分类器分析的结构化结果，包含所有阶段及关键帧信息。

        proto_path : str
            HTML 报告输出路径，支持目录或具体文件名。

        template_file : str
            Jinja2 模板路径，用于渲染 HTML 报告。

        boost : bool
            是否启用图像增强绘制模式，启用后会将稳定阶段中间帧以 base64 嵌入，其他阶段使用已保存图像路径。

        Returns
        -------
        str
            返回 HTML 报告文件的保存路径。

        Raises
        ------
        AssertionError
            若分类器输出结构不完整或为空阶段集合时抛出。

        FramixError
            若写入 HTML 报告文件失败或路径非法时抛出。

        Notes
        -----
        - 在 `boost=True` 时，稳定阶段的中间帧将被嵌入为 base64 编码；
        - 不稳定阶段或未知阶段以已保存图像的路径填充；
        - 会自动统计每个阶段的时长及 ID 范围作为标题；
        - HTML 中将包含图像列表、成本信息、时间戳、工具元信息等字段。

        Workflow
        --------
        1. 从 `struct_result` 中获取所有阶段片段；
        2. 遍历每个阶段片段并提取中间帧及全部帧图像；
        3. 构建每个阶段的标题、帧区间、耗时等信息；
        4. 使用 Jinja2 模板渲染 HTML 页面；
        5. 将报告保存至 `proto_path` 指定路径；
        6. 返回生成 HTML 报告文件路径。
        """
        label_stable = "稳定阶段"
        label_unstable = "不稳定阶段"
        label_unspecific = "不明阶段"

        thumbnail_list = []
        extra_dict = {}

        try:
            stage_range = struct_result.get_stage_range()
        except AssertionError:
            stage_range = [struct_result.data]

        if boost:
            for cur_index, _ in enumerate(stage_range):
                each_range = stage_range[cur_index]
                middle = each_range[len(each_range) // 2]

                image_list = []
                if middle.is_stable():
                    label = label_stable
                    image = "data:image/png;base64," + toolbox.np2b64str(middle.get_data())
                    frame = {
                        "frame_id": middle.frame_id, "timestamp": f"{middle.timestamp:.5f}", "image": image
                    }
                    image_list.append(frame)
                else:
                    if middle.stage == const.UNKNOWN_STAGE_FLAG:
                        label = label_unspecific
                    else:
                        label = label_unstable

                    if cur_index + 1 < len(stage_range):
                        new_each = [*each_range, stage_range[cur_index + 1][0]]
                    else:
                        new_each = each_range

                    for i in new_each:
                        frame = {
                            "frame_id": i.frame_id, "timestamp": f"{i.timestamp:.5f}", "image": scores[i.frame_id]
                        }
                        image_list.append(frame)

                first, last = each_range[0], each_range[-1]
                title = (f"{label} "
                         f"区间: {first.frame_id}({first.timestamp:.5f}) - {last.frame_id}({last.timestamp:.5f}) "
                         f"耗时: {last.timestamp - first.timestamp:.5f} "
                         f"分类: {first.stage}")
                thumbnail_list.append({title: image_list})

        else:
            for cur_index, _ in enumerate(stage_range):
                each_range = stage_range[cur_index]
                middle = each_range[len(each_range) // 2]

                image_list = []
                if middle.is_stable():
                    label = label_stable
                elif middle.stage == const.UNKNOWN_STAGE_FLAG:
                    label = label_unspecific
                else:
                    label = label_unstable

                if cur_index + 1 < len(stage_range):
                    new_each = [*each_range, stage_range[cur_index + 1][0]]
                else:
                    new_each = each_range

                for i in new_each:
                    frame = {
                        "frame_id": i.frame_id, "timestamp": f"{i.timestamp:.5f}", "image": scores[i.frame_id]
                    }
                    image_list.append(frame)

                first, last = each_range[0], each_range[-1]
                title = (f"{label} "
                         f"区间: {first.frame_id}({first.timestamp:.5f}) - {last.frame_id}({last.timestamp:.5f}) "
                         f"耗时: {last.timestamp - first.timestamp:.5f} "
                         f"分类: {first.stage}")
                thumbnail_list.append({title: image_list})

        cost_dict = struct_result.calc_changing_cost()
        timestamp = toolbox.get_timestamp_str()

        extra_dict["视频路径"] = struct_result.video_path
        extra_dict["总计帧数"] = str(struct_result.get_length())
        extra_dict["每帧间隔"] = str(struct_result.get_offset())

        html_template = Template(template_file).render(
            thumbnail_list=thumbnail_list,
            extras=extra_dict,
            background_color=const.BACKGROUND_COLOR,
            cost_dict=cost_dict,
            timestamp=timestamp,
            name=const.NAME,
            version_code=const.VERSION
        )

        if os.path.isdir(proto_path):
            report_path = os.path.join(proto_path, f"{timestamp}.html")
        else:
            report_path = proto_path

        async with aiofiles.open(report_path, "w", encoding=const.CHARSET) as f:
            await f.write(html_template)

        return report_path


if __name__ == '__main__':
    pass
