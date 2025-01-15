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
from engine.tinker import FramixReporterError
from nexaflow import toolbox, const
from nexaflow.classifier.base import ClassifierResult


class Report(object):

    def __init__(self, total_path: str):
        """
        初始化类时设置默认值。

        参数:
            total_path (str): 存储文件的根路径，用于初始化后续的文件夹结构。

        私有属性:
            __title (str): 用于存储标题的私有属性，默认值为空字符串。
            __query (str): 用于存储查询字符串的私有属性，默认值为空字符串。

        公共属性:
            query_path (str): 存储查询路径的属性，默认值为空字符串。
            video_path (str): 存储视频路径的属性，默认值为空字符串。
            frame_path (str): 存储帧路径的属性，默认值为空字符串。
            extra_path (str): 存储额外资源路径的属性，默认值为空字符串。
            range_list (list): 存储范围列表的属性，默认值为空列表。
            total_list (list): 存储总列表的属性，默认值为空列表。

        内部变量:
            _tms (str): 当前时间戳，用于生成唯一的路径名称。
            _pid (int): 当前进程的ID，用于生成唯一的路径名称。

        路径属性:
            total_path (str): 总路径，基于传入的total_path，时间戳和进程ID生成唯一目录，并在其中创建总集合文件夹。
            reset_path (str): 重置路径，基于total_path的父目录生成恢复文件夹。

        日志文件:
            log_papers (str): 日志文件的路径，保存在重置路径中，记录操作日志。
            logger (Logger): 配置日志记录器，记录操作日志，日志级别为NOTE_LEVEL，格式为WRITE_FORMAT。
        """

        self.__title = ""
        self.__query = ""

        self.query_path = ""
        self.video_path = ""
        self.frame_path = ""
        self.extra_path = ""

        self.range_list = []
        self.total_list = []

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
        if inform:
            self.range_list.append(inform)

    @staticmethod
    async def ask_merge_report(
            merge_list: typing.Union[list, tuple],
            template_file: str
    ) -> str:
        """
        静态方法: `ask_merge_report`

        功能:
            合并多个报告文件，生成最终的合并报告。此方法从多个日志文件中提取数据，将其合并到指定的模板文件中，并生成最终的HTML报告。

        参数:
            merge_list (typing.Union[list, tuple]): 需要合并的报告目录列表或元组。
            template_file (str): 用于生成最终报告的HTML模板文件路径。

        内部函数:
            assemble(file):
                异步读取指定文件的内容，并提取出包含“Recovery”关键词的数据。

        异常处理:
            FramixReporterError:
                当无法找到可以合并的报告时，抛出此异常。

        操作流程:
            1. 生成合并路径，并确保目标目录存在。
            2. 使用assemble函数从各个报告目录的日志文件中提取有效数据。
            3. 复制各个报告目录下的所有数据到合并路径中，忽略指定的文件类型。
            4. 使用模板文件生成最终的HTML报告，并保存到指定路径。

        返回值:
            report_html (str): 返回最终生成的HTML报告文件路径。

        异常:
            如果在操作过程中遇到任何异常或错误，将抛出 `FramixReporterError` 以便处理。
        """

        async def assemble(file):
            async with aiofiles.open(file, mode="r", encoding=const.CHARSET) as recovery_file:
                return re.findall(r"(?<=Recovery: ).*}", await recovery_file.read())

        log_file = const.R_RECOVERY, const.R_LOG_FILE
        if not (log_file_list := [os.path.join(merge, *log_file) for merge in merge_list]):
            raise FramixReporterError(f"没有可以合并的报告 ...")

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
                raise FramixReporterError(m)
            shutil.copytree(
                os.path.join(m, const.R_COLLECTION).format(),
                merge_path,
                ignore=shutil.ignore_patterns(*ignore),
                dirs_exist_ok=True
            )

        if not (total_list := [json.loads(i) for logs in merge_log_list for i in logs if i]):
            raise FramixReporterError(f"没有可以合并的报告 ...")

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
            total: "Path",
            title: str,
            serial: str,
            parts_list: list,
            style_loc: str
    ) -> typing.Optional[dict]:
        """
        静态方法: `ask_create_report`

        功能:
            生成视频分析的HTML报告，并返回包含分析结果的字典。该方法根据输入的分析数据，生成相应的图像列表和统计信息，最终输出一个HTML报告文件。

        参数:
            total (Path): 存放报告的根目录路径。
            title (str): 报告的标题名称。
            serial (str): 报告的序列号，用于唯一标识报告文件。如果未提供，将生成一个随机序列号。
            parts_list (list): 包含多个字典的列表，每个字典存储一个视频分析的详细信息。
            style_loc (str): HTML模板文件的路径，用于生成最终报告。

        内部函数:
            views_frame(query, frame):
                根据`views`样式从指定路径加载并返回图像列表，按照帧编号排序。

            major_frame(query, frame):
                根据`major`样式从指定路径加载并返回图像列表，按照帧编号排序，并包含时间戳。

            extra_frame(query, frame):
                从指定路径加载`extra`样式的图像列表，并按照图像的索引排序。

            transform(inform_part):
                根据输入的分析部分信息`inform_part`，调用相应的图像加载函数并构建分析数据字典。

        操作流程:
            1. 根据分析样式，调用相应的图像加载函数生成图像列表。
            2. 将图像列表及其他相关数据渲染到HTML模板中，并保存为HTML文件。
            3. 计算图像处理的平均耗时，生成包含报告信息的字典并返回。

        返回值:
            single (dict): 包含报告信息的字典，键包括`case`(报告标题)、`cost_list`(耗时列表)、`avg`(平均耗时)和`href`(HTML报告的路径)。

        异常:
            ZeroDivisionError:
                在计算平均耗时时，如果耗时列表为空，处理零除异常，返回平均耗时为`0.00000`。
        """

        async def views_frame(query, frame):
            frame_list = []
            for image in os.listdir(os.path.join(total, title, query, frame)):
                image_src = os.path.join(query, frame, image)
                image_idx = int(re.search(r"(?<=frame_)\d+", image).group())
                frame_list.append(
                    {
                        "src": image_src,
                        "frames_id": image_idx,
                    }
                )
            frame_list.sort(key=lambda x: x["frames_id"])
            return frame_list

        async def major_frame(query, frame):
            frame_list = []
            for image in os.listdir(os.path.join(total, title, query, frame)):
                image_src = os.path.join(query, frame, image)
                image_idx = re.search(r"\d+(?=_)", image).group()
                timestamp = float(re.search(r"(?<=_).+(?=\.)", image).group())
                frame_list.append(
                    {
                        "src": image_src,
                        "frames_id": image_idx,
                        "timestamp": f"{timestamp:.5f}"
                    }
                )
            frame_list.sort(key=lambda x: int(x["frames_id"]))
            return frame_list

        async def extra_frame(query, frame):
            frame_list = []
            for image in os.listdir(os.path.join(total, title, query, frame)):
                image_src = os.path.join(query, frame, image)
                image_idx = image.split("(")[0]
                frame_list.append(
                    {
                        "src": image_src,
                        "idx": image_idx
                    }
                )
            frame_list.sort(key=lambda x: int(x["idx"].split("(")[0]))
            return frame_list

        async def transform(inform_part):
            inform_list = []
            style = inform_part.get("style", "")
            query = inform_part.get("query", "")
            stage = inform_part.get("stage", {})
            frame = inform_part.get("frame", "")
            extra = inform_part.get("extra", "")
            proto = inform_part.get("proto", "")
            inform_dict: dict = {"query": query, "stage": stage}

            if style == "speed":
                inform_dict["image_list"] = await views_frame(query, frame)
            elif style == "basic":
                inform_dict["image_list"] = await major_frame(query, frame)
            elif style == "keras":
                image_list, extra_list = await asyncio.gather(
                    major_frame(query, frame), extra_frame(query, extra)
                )
                inform_dict["image_list"] = image_list
                inform_dict["extra_list"] = extra_list
                inform_dict["proto"] = os.path.join(query, proto)

            inform_list.append(inform_dict)
            return inform_list

        if len(parts_list) == 0:
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
            file_name: str,
            group: bool,
            style_loc: str,
            total_loc: str
    ) -> str:
        """
        静态方法: `ask_create_total_report`

        功能:
            生成一个总报告，汇总多个分析报告的结果。该方法从指定文件中提取分析数据，按照一定的规则对数据进行分组和整理，最终生成一个包含所有分析结果的总报告HTML文件。

        参数:
            file_name (str): 存放日志文件的根目录路径。
            group (bool): 是否将结果按照某个标准分组。
            style_loc (str): HTML模板文件的路径，用于生成最终报告。
            total_loc (str): 总报告模板文件的路径。

        操作流程:
            1. 尝试打开并读取指定文件中的日志数据。
            2. 从日志中提取符合条件的分析数据，并将其转换为字典列表。
            3. 根据`group`参数，按一定规则对数据进行分组和整理。
            4. 调用`Report.ask_create_report`生成单个报告文件，并收集生成的报告信息。
            5. 整理生成的报告信息，按照一定格式将数据合并。
            6. 渲染总报告模板，并生成总报告HTML文件。

        返回值:
            total_html (str): 生成的总报告HTML文件的路径。

        异常:
            FileNotFoundError:
                如果无法找到指定的日志文件，将抛出此异常。
            FramixReporterError:
                如果在日志文件中无法找到符合条件的数据，或者在生成报告时发生错误，将抛出此异常。
        """

        try:
            log_file = const.R_RECOVERY, const.R_LOG_FILE
            async with aiofiles.open(os.path.join(file_name, *log_file), "r", encoding=const.CHARSET) as f:
                open_file = await f.read()
        except FileNotFoundError as e:
            raise FramixReporterError(e)

        if match_speeder_list := re.findall(r"(?<=Speeder: ).*}", open_file):
            match_list = match_speeder_list
        elif match_restore_list := re.findall(r"(?<=Restore: ).*}", open_file):
            match_list = match_restore_list
        else:
            raise FramixReporterError(f"没有符合条件的数据 ...")

        parts_list: list[dict] = [
            json.loads(file) for file in match_list if file
        ]
        parts_list: list[dict] = [
            {(p.pop("total"), p.pop("title"), Path(p["query"]).name if group else ""): p} for p in parts_list
        ]

        async def format_packed():
            parts_dict = defaultdict(lambda: defaultdict(list))
            for parts in parts_list:
                for key, value in parts.items():
                    for k, v in value.items():
                        parts_dict[key][k].append(v)
            normal_dict = {k: dict(v) for k, v in parts_dict.items()}

            return {
                k: [dict(zip(v.keys(), entry)) for entry in zip(*v.values())] for k, v in normal_dict.items()
            }

        async def format_merged():
            parts_dict = defaultdict(lambda: defaultdict(list))
            for rp in create_total_result:
                for key, value in rp.items():
                    if key not in ["case", "team"]:
                        parts_dict[f"{rp['case']}-{rp['team']}"][key].append(value)

            return [
                {"case": keys.split("-")[0], "team": keys.split("-")[1], **{k: v for k, v in attrs.items()}}
                for keys, attrs in parts_dict.items()
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
            raise FramixReporterError(e)

        create_total_result = [create for create in create_result if create]
        merged_list = await format_merged()

        for i in merged_list:
            i["merge_list"] = list(zip(i.pop("href"), i.pop("avg"), i.pop("cost_list")))
            for j in i["merge_list"]:
                logger.debug(f"{j}")

        if not (total_list := [single for single in merged_list if single]):
            raise FramixReporterError(f"没有可以汇总的报告 ...")

        total_html_temp = Template(total_loc).render(
            head=const.R_TOTAL_HEAD, report_time=time.strftime('%Y.%m.%d %H:%M:%S'), total_list=total_list
        )
        total_html = os.path.join(file_name, const.R_TOTAL_FILE)
        async with aiofiles.open(total_html, "w", encoding=const.CHARSET) as f:
            await f.write(total_html_temp)

        return total_html

    @staticmethod
    async def ask_draw(
            scores: dict,
            struct_result: "ClassifierResult",
            proto_path: str,
            template_file: str,
            boost: bool,
    ) -> str:
        """
        静态方法: `ask_draw`

        功能:
            根据分类器的结果生成一份包含分析数据和缩略图的HTML报告。
            报告内容包括各个阶段的帧图像、时间戳、分类标签等信息，最后将生成的报告保存为HTML文件。

        参数:
            scores (dict): 每帧对应的图像数据，以帧ID为键，图像数据为值。
            struct_result (ClassifierResult): 分类器的结果，包含帧的分析数据和阶段信息。
            proto_path (str): 保存生成的报告的路径，如果是目录则在目录下生成HTML文件，否则将路径作为文件名。
            template_file (str): HTML模板文件的路径，用于渲染最终的报告。
            boost (bool): 如果为True，报告中将只包含中间帧的信息；如果为False，报告将包含所有阶段的所有帧。

        操作流程:
            1. 根据分类器的结果获取每个阶段的帧范围。
            2. 根据`boost`参数决定如何选择帧，并生成各个阶段的图像列表和相关信息。
            3. 计算视频的分析成本数据，并生成附加信息如视频路径、帧总数、每帧间隔等。
            4. 使用指定的HTML模板渲染报告内容。
            5. 将渲染后的HTML报告保存到指定路径。

        返回值:
            report_path (str): 生成的HTML报告文件的路径。

        异常:
            无直接异常处理，但`aiofiles`的文件操作可能引发I/O异常。
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
