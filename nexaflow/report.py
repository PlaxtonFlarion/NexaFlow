import os
import re
import ast
import json
import time
import shutil
import typing
import random
import asyncio
import aiofiles
import threading
from pathlib import Path
from loguru import logger
from jinja2 import Template
from collections import defaultdict
from nexaflow import toolbox, const


class Report(object):

    __lock: threading.Lock = threading.Lock()
    __initialized: bool = False
    __instance = None
    __init_var = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            with cls.__lock:
                if cls.__instance is None:
                    cls.__instance = super(Report, cls).__new__(cls)
                    cls.__init_var = (args, kwargs)
        return cls.__instance

    def __init__(self, total_path: str):
        if not self.__initialized:
            self.__initialized = True

            self.__title = ""
            self.__query = ""
            self.query_path = ""
            self.video_path = ""
            self.frame_path = ""
            self.extra_path = ""

            self.range_list = []
            self.total_list = []

            self.total_path = os.path.join(
                total_path, f"Nexa_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}", "Nexa_Collection"
            )
            os.makedirs(self.total_path, exist_ok=True)

            self.reset_path = os.path.join(os.path.dirname(self.total_path), "Nexa_Recovery")
            os.makedirs(self.reset_path, exist_ok=True)
            log_papers = os.path.join(self.reset_path, "nexaflow.log")
            logger.add(log_papers, format=const.LOG_FORMAT, level="DEBUG")

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
    def reset_report(file_name: str, template_file: str) -> None:
        with open(os.path.join(file_name, "Nexa_Recovery", "nexaflow.log"), "r", encoding="utf-8") as f:
            log_restore = re.findall(r"(?<=Recovery: ).*}", f.read())

        total_list = [json.loads(file) for file in log_restore]

        html = Template(template_file).render(
            report_time=time.strftime('%Y.%m.%d %H:%M:%S'),
            total_list=total_list
        )
        report_html = os.path.join(file_name, "NexaFlow.html")
        with open(report_html, "w", encoding=const.CHARSET) as f:
            f.write(html)
            logger.info(f"生成汇总报告: {report_html}")

    @staticmethod
    def merge_report(merge_list: list[str], merge_loc: str) -> None:
        merge_time = time.strftime("%Y%m%d%H%M%S")
        merge_path = os.path.join(
            os.path.dirname(os.path.dirname(merge_list[0])),
            f"Union_Report_{merge_time}", "Nexa_Collection"
        )
        os.makedirs(merge_path, exist_ok=True)

        pattern = "Recovery"

        log_list = []
        for merge in merge_list:
            logs = os.path.join(os.path.dirname(merge), "Nexa_Recovery", "nexaflow.log")
            with open(logs, "r", encoding=const.CHARSET) as f:
                log_list.extend(re.findall(fr"(?<={pattern}: ).*}}", f.read()))

            shutil.copytree(
                merge, merge_path, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("NexaFlow.html", "nexaflow.log")
            )

        if not (total_list := [json.loads(file.replace("'", '"')) for file in log_list if file]):
            return logger.warning(f"没有可以合并的报告 ...")

        html = Template(merge_loc).render(
            report_time=merge_time, total_list=total_list
        )

        report_html = os.path.join(os.path.dirname(merge_path), "NexaFlow.html")
        with open(report_html, "w", encoding=const.CHARSET) as f:
            f.write(html)
            logger.info(f"合并汇总报告: {report_html}")

    @staticmethod
    async def ask_create_report(total: "Path", title: str, serial: str, parts_list: list, style_loc: str):

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
            inform_dict: dict[str | int | list | bytes] = {"query": query, "stage": stage}

            if style == "quick":
                inform_dict["image_list"] = await views_frame(query, frame)
            elif style == "basic":
                inform_dict["image_list"] = await major_frame(query, frame)
            else:
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
        teams = serial if serial else random.randint(10000, 99999)
        html_path = Path(os.path.join(total, title, f"{title}_{teams}.html"))
        async with aiofiles.open(html_path, "w", encoding=const.CHARSET) as range_file:
            await range_file.write(html_temp)

        cost_list = [cost["stage"]["cost"] for cost in images_list]
        try:
            avg = sum(map(float, cost_list)) / len(cost_list)
        except ZeroDivisionError:
            avg = 0.00000

        href = os.path.join(total.name, title, html_path.name)
        single = {
            "case": title, "cost_list": cost_list, "avg": f"{avg:.5f}", "href": href
        }
        logger.debug("Recovery: " + json.dumps(single, ensure_ascii=False))
        return single

    @staticmethod
    async def ask_create_total_report(file_name: str, file_term: str, style_loc: str, total_loc: str, group: bool):
        try:
            file_path = os.path.join(file_name, "Nexa_Recovery", "nexaflow.log")
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                open_file = await f.read()
        except FileNotFoundError as e:
            return e

        pattern = "Restore" if file_term == "main" else "Quicker"

        if len(match_list := re.findall(fr"(?<={pattern}: ).*}}", open_file)) == 0:
            return logger.warning(f"没有符合条件的数据 ...")

        parts_list = [json.loads(file) for file in match_list if file]
        parts_dict = {ast.literal_eval(k): [v] for parts in parts_list for k, v in parts.items()}

        create_result = await asyncio.gather(
            *(Report.ask_create_report(
                Path(os.path.join(file_name, a)), b, c, v, style_loc)
                for (a, b, c), v in parts_dict.items())
        )
        total_result = [create for create in create_result if create]

        if group:
            merged = defaultdict(lambda: defaultdict(list))
            for case in total_result:
                for k, v in case.items():
                    if k != "case":
                        merged[case["case"]][k].append(v)
            total_list = [
                {"case": case, **{k: v for k, v in attrs.items()}} for case, attrs in merged.items()
            ]
            for item in total_list:
                item["merge_list"] = list(zip(item.pop("href"), item.pop("avg"), item.pop("cost_list")))
            logger.debug("=" * 30)
            for item in total_list:
                logger.debug(item["case"])
                for detail in item["merge_list"]:
                    logger.debug(detail)
            logger.debug("=" * 30)
        else:
            total_list = total_result

        if len(total_list := [single for single in total_list if single]) == 0:
            return logger.warning(f"没有可以汇总的报告 ...")

        total_html_temp = Template(total_loc).render(
            report_time=time.strftime('%Y.%m.%d %H:%M:%S'),
            total_list=total_list
        )
        total_html = os.path.join(file_name, "NexaFlow.html")
        async with aiofiles.open(total_html, "w", encoding=const.CHARSET) as f:
            await f.write(total_html_temp)
            logger.info(f"生成汇总报告: {total_html}")

    @staticmethod
    async def ask_draw(
        classifier_result,
        proto_path: str,
        template_file: str,
        scale: typing.Optional[float] = None,
        shape: typing.Optional[tuple[int, int]] = None,
        boost: bool = False,
    ) -> str:

        label_stable = "稳定阶段"
        label_unstable = "不稳定阶段"
        label_unspecific = "不明阶段"

        thumbnail_list = []
        extra_dict = {}

        try:
            stage_range = classifier_result.get_stage_range()
        except AssertionError:
            stage_range = [classifier_result.data]

        image_list = []
        if boost:
            for cur_index, _ in enumerate(stage_range):
                each_range = stage_range[cur_index]
                middle = each_range[len(each_range) // 2]
                if middle.is_stable():
                    label = label_stable
                    image = toolbox.compress_frame(
                        middle.get_data(), compress_rate=scale, target_size=shape
                    )
                    frame = {
                        "frame_id": middle.frame_id, "timestamp": f"{middle.timestamp:.5f}", "image": toolbox.np2b64str(image)
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
                        image = toolbox.compress_frame(
                            i.get_data(), compress_rate=scale, target_size=shape
                        )
                        frame = {
                            "frame_id": i.frame_id, "timestamp": f"{i.timestamp:.5f}", "image": toolbox.np2b64str(image)
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
                    image = toolbox.compress_frame(
                        i.get_data(), compress_rate=scale, target_size=shape
                    )
                    frame = {
                        "frame_id": i.frame_id, "timestamp": f"{i.timestamp:.5f}", "image": toolbox.np2b64str(image)
                    }
                    image_list.append(frame)

                first, last = each_range[0], each_range[-1]
                title = (f"{label} "
                         f"区间: {first.frame_id}({first.timestamp:.5f}) - {last.frame_id}({last.timestamp:.5f}) "
                         f"耗时: {last.timestamp - first.timestamp:.5f} "
                         f"分类: {first.stage}")
                thumbnail_list.append({title: image_list})

        cost_dict = classifier_result.calc_changing_cost()
        timestamp = toolbox.get_timestamp_str()

        extra_dict["视频路径"] = classifier_result.video_path
        extra_dict["总计帧数"] = str(classifier_result.get_length())
        extra_dict["每帧间隔"] = str(classifier_result.get_offset())

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
        logger.info(f"生成阶段报告: {os.path.basename(report_path)}")

        return report_path


if __name__ == '__main__':
    pass
