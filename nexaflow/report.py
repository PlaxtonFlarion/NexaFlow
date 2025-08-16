#   ____                       _
#  |  _ \ ___ _ __   ___  _ __| |_
#  | |_) / _ \ '_ \ / _ \| '__| __|
#  |  _ <  __/ |_) | (_) | |  | |_
#  |_| \_\___| .__/ \___/|_|   \__|
#            |_|
#
# ==== Notes: License ====
# Copyright (c) 2024  Framix :: 画帧秀
# This file is licensed under the Framix :: 画帧秀 License. See the LICENSE.md file for more details.

import os
import re
import json
import time
import typing
import random
import string
import asyncio
import aiofiles
import aiosqlite
from pathlib import Path
from loguru import logger
from jinja2 import Template
from collections import defaultdict
from engine.tinker import FramixError
from nexacore.cubicle import DB
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

    def __init__(self, total_path: str, label: typing.Optional[str] = None):
        """
        初始化实例并构建报告的目录结构与必要属性。

        Parameters
        ----------
        total_path : str
            分析报告的根目录路径，用于存放所有生成的报告及相关文件。

        label : str, optional
            报告的标签标识，如果未提供则自动生成时间戳与进程号的唯一标识。

        Notes
        -----
        该方法会初始化多个路径属性（查询目录、视频目录、帧目录、附加目录）并
        准备存储范围数据和汇总数据的列表结构。同时会生成用于区分不同报告的唯一标签。
        """
        self.__title: str = ""
        self.__query: str = ""

        self.query_path: str = ""
        self.video_path: str = ""
        self.frame_path: str = ""
        self.extra_path: str = ""

        self.range_list: list = []
        self.total_list: list = []

        tag: str = f"{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}"
        tender: str = const.R_TOTAL_TAG + "_" + (label or tag)

        self.total_path: str = os.path.join(total_path, tender, const.R_COLLECTION)
        if not (total_path := Path(self.total_path)).exists():
            total_path.mkdir(parents=True, exist_ok=True)

        self.reset_path: str = os.path.join(os.path.dirname(self.total_path), const.R_RECOVERY)
        if not (reset_path := Path(self.reset_path)).exists():
            reset_path.mkdir(parents=True, exist_ok=True)

        log_papers: str = os.path.join(self.reset_path, const.R_LOG_FILE)
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
    async def write_html_file(html: typing.Union[str, "Path"], html_template: str) -> None:
        """
        将渲染后的 HTML 模板异步写入指定路径文件中。

        Parameters
        ----------
        html : Union[str, Path]
            输出 HTML 文件的路径，若文件已存在将被覆盖。

        html_template : str
            渲染后的 HTML 字符串内容。
        """
        async with aiofiles.open(html, "w", encoding=const.CHARSET) as f:
            await f.write(html_template)

    @staticmethod
    async def ask_create_report(
        total: "Path",
        title: str,
        serial: str,
        parts_list: list,
        style_loc: str
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
                    "src": image_src, "frame_id": int(image_idx)
                }

                if fit_time := re.search(r"(?<=_)\d+\.\d+(?=\.)", image):
                    image_source["timestamp"] = fit_time.group()

                frame_list.append(image_source)

            frame_list.sort(key=lambda x: x["frame_id"])
            return frame_list

        async def transform(inform_part: dict) -> dict[str, typing.Any]:
            """
            转换报告数据字典，整合图像与阶段信息。
            """
            query = inform_part.get("query", "")
            stage = inform_part.get("stage", {})
            frame = inform_part.get("frame", "")
            extra = inform_part.get("extra", "")
            proto = inform_part.get("proto", "")

            image_list = await acquire(query, frame)

            return {
                "query": query,
                "stage": stage,
                "image_list": image_list,
                "extra": os.path.join(query, extra) if extra else None,
                "proto": os.path.join(query, proto) if proto else None,
            }

        # Notes: Start from here
        if not parts_list:
            return None

        transform_result = await asyncio.gather(
            *(transform(parts) for parts in parts_list)
        )
        images_list = [result for result in transform_result]

        html_temp = Template(style_loc).render(
            app_desc=const.DESC, title=title, images_list=images_list
        )

        team = serial if serial else random.randint(10000, 99999)
        html = total / title / f"{title}_{team}.html"

        write_task = asyncio.create_task(
            Report.write_html_file(html, html_temp)
        )

        cost_list = [cost["stage"]["cost"] for cost in images_list]
        try:
            avg = sum(map(float, cost_list)) / len(cost_list)
        except ZeroDivisionError:
            avg = 0.00000

        href = os.path.join(total.name, title, html.name)
        single = {
            "case": title, "team": team, "cost_list": cost_list, "avg": f"{avg:.5f}", "href": href
        }
        logger.debug("Recovery: " + json.dumps(single, ensure_ascii=False))

        await write_task

        return single

    @staticmethod
    async def ask_create_total_report(
        file_name: str,
        group: bool,
        style_loc: str,
        total_loc: str
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
        Union[Any, str]
            返回值可能是任意类型（`Any`）或字符串（`str`）：
            - **Any**：当函数的具体逻辑允许返回非字符串的其他类型数据时（例如对象、数值、字典等）。
            - **str**：当函数执行成功并返回文件路径、消息或其他字符串结果时。

        Raises
        ------
        FramixError
            - 当日志文件不存在或无法读取；
            - 当未能提取出任何有效数据或报告内容为空；
            - 当报告渲染过程中发生异常。
        """
        try:
            async with DB(Path(file_name) / const.R_RECOVERY / const.DB_FILES_NAME) as db:
                match_list = [nest[0] for nest in await db.demand()]
        except aiosqlite.Error as e:
            raise FramixError(e)

        if not (parts_list := [json.loads(file) for file in match_list if file]):
            raise FramixError(f"没有符合条件的数据 ...")

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

        async def format_merged(create_total_result: list[dict]) -> list[dict]:
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
                    Path(file_name) / total, title, sn, result_dict, style_loc
                ) for (total, title, sn), result_dict in packed_dict.items())
            )
        except Exception as e:
            raise FramixError(e)

        merged_list = await format_merged([c for c in create_result if c])

        for i in merged_list:
            i["merge_list"] = list(zip(i.pop("href"), i.pop("avg"), i.pop("cost_list")))
            for j in i["merge_list"]:
                logger.debug(f"{j}")

        if not (total_list := [s for s in merged_list if s]):
            raise FramixError(f"没有可以汇总的报告 ...")

        html_temp = Template(total_loc).render(
            head=const.R_TOTAL_HEAD,
            app_desc=const.DESC,
            report_time=time.strftime('%Y.%m.%d %H:%M:%S'),
            total_list=total_list
        )

        salt: typing.Callable[
            [], str
        ] = lambda: "".join(random.choices(
            string.ascii_uppercase + string.digits, k=5)
        )
        html = os.path.join(file_name, f"{const.R_TOTAL_NAME}_{salt()}.html")

        await Report.write_html_file(html, html_temp)

        return html

    @staticmethod
    async def ask_line(
        extra_path: str,
        proto_path: str,
        template_file: str,
        rp_timestamp: str
    ) -> typing.Union[typing.Any, str]:
        """
        生成包含帧图列表的 HTML 报告页面，用于比对额外帧结构内容。

        Parameters
        ----------
        extra_path : str
            包含额外帧图像文件的目录路径，文件命名应含帧索引前缀。

        proto_path : str
            输出报告的路径或目录；
            若为目录，将根据时间戳自动命名 HTML 报告文件。

        template_file : str
            Jinja2 模板文件路径，作为 HTML 渲染基础。

        rp_timestamp : str
            报告命名用的时间戳，用于构造最终文件名。

        Returns
        -------
        Union[Any, str]
            返回 HTML 文件的输出路径；
            若发生异常或特殊逻辑处理，也可能返回任意类型结果。
        """
        frame_list = []
        for image in os.listdir(extra_path):
            image_src = os.path.join(os.path.basename(extra_path), image)
            image_idx = image.split("(")[0]

            image_source = {
                "src": image_src, "frame_id": int(image_idx)
            }
            frame_list.append(image_source)

        frame_list.sort(key=lambda x: x["frame_id"])

        html_template = Template(template_file).render(
            app_desc=const.DESC,
            resp={"extra_list": frame_list}
        )

        if os.path.isdir(proto_path):
            report_path = os.path.join(proto_path, f"{rp_timestamp}_line.html")
        else:
            report_path = proto_path

        await Report.write_html_file(report_path, html_template)

        return report_path

    @staticmethod
    async def ask_draw(
        scores: dict,
        struct_result: "ClassifierResult",
        proto_path: str,
        template_file: str,
        boost: bool,
        rp_timestamp: str
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

        rp_timestamp : str
            报告命名用的时间戳，用于构造最终文件名。

        Returns
        -------
        Union[Any, str]
            返回值可能是任意类型（`Any`）或字符串（`str`）：
            - **Any**：当函数的具体逻辑允许返回非字符串的其他类型数据时（例如对象、数值、字典等）。
            - **str**：当函数执行成功并返回文件路径、消息或其他字符串结果时。

        Raises
        ------
        AssertionError
            若分类器输出结构不完整或为空阶段集合时抛出。

        FramixError
            若写入 HTML 报告文件失败或路径非法时抛出。
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

        extra_dict["视频路径"] = struct_result.video_path
        extra_dict["总计帧数"] = str(struct_result.get_length())
        extra_dict["每帧间隔"] = str(struct_result.get_offset())

        html_template = Template(template_file).render(
            thumbnail_list=thumbnail_list,
            extras=extra_dict,
            background_color=const.BACKGROUND_COLOR,
            cost_dict=cost_dict,
            timestamp=rp_timestamp,
            app_desc=const.DESC,
            version_code=const.VERSION
        )

        if os.path.isdir(proto_path):
            report_path = os.path.join(proto_path, f"{rp_timestamp}_atom.html")
        else:
            report_path = proto_path

        await Report.write_html_file(report_path, html_template)

        return report_path


if __name__ == '__main__':
    pass
