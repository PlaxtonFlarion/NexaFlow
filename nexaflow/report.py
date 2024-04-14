import os
import re
import json
import time
import shutil
import random
import asyncio
import aiofiles
import threading
from pathlib import Path
from loguru import logger
from jinja2 import Template
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Union
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

            self.clock: Any = lambda: time.strftime("%Y%m%d%H%M%S")

            self.__title = ""
            self.__query = ""
            self.query_path = ""
            self.video_path = ""
            self.frame_path = ""
            self.extra_path = ""

            self.range_list = []
            self.total_list = []

            self.total_path = os.path.join(total_path, f"Nexa_{self.clock()}_{os.getpid()}", "Nexa_Collection")
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
        logger.info(f"✪✪✪✪✪✪✪✪✪✪ {self.title} ✪✪✪✪✪✪✪✪✪✪")

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
        logger.info(f"Start -> {self.query}")

    @query.deleter
    def query(self):
        del self.__query

    @staticmethod
    def template(template_path: str) -> str:
        with open(template_path, encoding=const.CHARSET) as t:
            template_file = t.read()
        return template_file

    def load(self, inform: Optional[dict[str, Union[str | dict]]]) -> None:
        if inform:
            self.range_list.append(inform)
        logger.info(f"End -> {inform.get('query', '......')}")

    def create_report(self, template_file: str) -> None:

        def start_create(result):
            handler_list = []
            query = result.get("query", "TimeCost")
            stage = result.get("stage", {"start": 1, "end": 2, "cost": "0.00000"})
            frame = result.get("frame", "")
            extra = result.get("extra", "")
            proto = result.get("proto", "")

            image_list = []
            for image in os.listdir(frame):
                image_src = os.path.join(query, "frame", image)
                image_ids = re.search(r"\d+(?=_)", image).group()
                timestamp = float(re.search(r"(?<=_).+(?=\.)", image).group())
                image_list.append(
                    {
                        "src": image_src,
                        "frames_id": image_ids,
                        "timestamp": f"{timestamp:.5f}"
                    }
                )
            image_list.sort(key=lambda x: int(x["frames_id"]))

            extra_list = []
            for ex in os.listdir(extra):
                extra_src = os.path.join(query, "extra", ex)
                extra_idx = ex.split("(")[0]
                extra_list.append(
                    {
                        "src": extra_src,
                        "idx": extra_idx
                    }
                )
            extra_list.sort(key=lambda x: int(x["idx"].split("(")[0]))

            handler_list.append(
                {
                    "query": query,
                    "stage": stage,
                    "image_list": image_list,
                    "extra_list": extra_list,
                    "proto": os.path.join(query, os.path.basename(proto)),
                    "should_display": True if proto else False
                }
            )

            return handler_list

        if len(self.range_list) == 0:
            logger.info("没有可以聚合的报告 ...")
            return logger.info(f"✪✪✪✪✪✪✪✪✪✪ {self.title} ✪✪✪✪✪✪✪✪✪✪")

        if len(self.range_list) == 1:
            images_list = start_create(self.range_list[0])
        else:
            with ThreadPoolExecutor() as executor:
                future = executor.map(start_create, self.range_list)
            images_list = [i for f in future for i in f]

        html = Template(self.template(template_file)).render(
            name=const.NAME,
            title=self.title,
            images_list=images_list
        )
        report_html = os.path.join(self.query_path, f"{self.title}.html")
        with open(report_html, "w", encoding=const.CHARSET) as f:
            f.write(html)
            logger.info(f"生成聚合报告: {os.path.basename(report_html)}")

        cost_list = [cost['stage']['cost'] for cost in images_list]
        try:
            avg = sum(map(float, cost_list)) / len(cost_list)
        except ZeroDivisionError:
            avg = 0.00000
            logger.warning("未获取到平均值 ...")

        href_path = os.path.join(
            os.path.basename(self.total_path),
            self.title,
            os.path.basename(report_html)
        )
        single = {
            "case": self.title, "cost_list": cost_list, "avg": avg, "href": href_path
        }
        logger.debug("Recovery: " + json.dumps(single, ensure_ascii=False))
        self.total_list.append(single)
        self.range_list.clear()
        return logger.info(f"✪✪✪✪✪✪✪✪✪✪ {self.title} ✪✪✪✪✪✪✪✪✪✪")

    def create_total_report(self, template_file: str) -> None:
        if len(self.total_list) == 0:
            return logger.info("没有可以汇总的报告 ...")

        html = Template(self.template(template_file)).render(
            report_time=time.strftime('%Y.%m.%d %H:%M:%S'),
            total_list=self.total_list
        )
        report_html = os.path.join(os.path.dirname(self.total_path), "NexaFlow.html")
        with open(report_html, "w", encoding=const.CHARSET) as f:
            f.write(html)
            logger.info(f"生成汇总报告: {report_html}")
        self.total_list.clear()

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
    def merge_report(merge_list: list[str], merge_loc: str, quick: bool = False) -> None:
        merge_time = time.strftime("%Y%m%d%H%M%S")
        merge_path = os.path.join(
            os.path.dirname(os.path.dirname(merge_list[0])),
            f"Union_Report_{merge_time}", "Nexa_Collection"
        )
        os.makedirs(merge_path, exist_ok=True)

        pattern = "View" if quick else "Recovery"

        log_list = []
        for merge in merge_list:
            logs = os.path.join(os.path.dirname(merge), "Nexa_Recovery", "nexaflow.log")
            with open(logs, "r", encoding=const.CHARSET) as f:
                log_list.extend(re.findall(fr"(?<={pattern}: ).*}}", f.read()))

            shutil.copytree(
                merge, merge_path, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("NexaFlow.html", "nexaflow.log")
            )

        if not (total_list := [json.loads(file) for file in log_list]):
            return logger.warning(f"没有可以合并的报告 ...")

        html = Template(merge_loc).render(
            report_time=merge_time, total_list=total_list
        )

        report_html = os.path.join(os.path.dirname(merge_path), "NexaFlow.html")
        with open(report_html, "w", encoding=const.CHARSET) as f:
            f.write(html)
            logger.info(f"合并汇总报告: {report_html}")

    @staticmethod
    async def ask_invent_report(total_path: Path, title: str, serial: str, parts_list: list, views_loc: str):

        async def deal_with_inform(result):
            handler_list = []
            query = result.get("query", "query")
            stage = result.get("stage", {"start": 0, "end": 0, "cost": 0})
            frame = result.get("frame", "frame")

            async def handler_frame():
                handler_image_list = []
                for image in os.listdir(os.path.join(total_path, title, query, frame)):
                    image_src = os.path.join(query, os.path.basename(frame), image)
                    image_ids = int(re.search(r"(?<=frame_)\d+", image).group())
                    handler_image_list.append(
                        {
                            "src": image_src,
                            "frames_id": image_ids,
                        }
                    )
                handler_image_list.sort(key=lambda x: x["frames_id"])
                return handler_image_list

            data = {"query": query, "stage": stage}

            image_list = await handler_frame()

            data["image_list"] = image_list

            handler_list.append(data)
            return handler_list

        async def handler_start():
            single = {}
            if len(parts_list) > 0:
                tasks = [deal_with_inform(result) for result in parts_list]
                results = await asyncio.gather(*tasks)
                images_list = [ele for res in results for ele in res]

                range_html_temp = Template(views_loc).render(
                    title=f"{const.DESC}",
                    report_time=time.strftime('%Y.%m.%d %H:%M:%S'),
                    images_list=images_list
                )
                teams = serial if serial else random.randint(10000, 99999)
                range_html = Path(os.path.join(total_path, title, f"{title}_{teams}.html"))
                async with aiofiles.open(range_html, "w", encoding=const.CHARSET) as range_file:
                    await range_file.write(range_html_temp)
                    logger.info(f"生成聚合报告: {range_html.name}")

                cost_list = [cost["stage"]["cost"] for cost in images_list]
                try:
                    avg = sum(map(float, cost_list)) / len(cost_list)
                except ZeroDivisionError:
                    avg = 0.00000
                    logger.warning("未获取到平均值 ...")

                href_path = os.path.join(total_path.name, title, range_html.name)
                single = {
                    "case": title, "cost_list": cost_list, "avg": f"{avg:.5f}", "href": href_path
                }
                logger.debug("View: " + json.dumps(single, ensure_ascii=False))
            else:
                logger.info("没有可以聚合的报告 ...")

            logger.info(f"✪✪✪✪✪✪✪✪✪✪ {title} ✪✪✪✪✪✪✪✪✪✪")
            return single

        return await handler_start()

    @staticmethod
    async def ask_invent_total_report(file_name: str, views_loc: str, total_loc: str, group: bool):
        try:
            file_path = os.path.join(file_name, "Nexa_Recovery", "nexaflow.log")
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                open_file = await f.read()
        except FileNotFoundError as e:
            return e

        match_list = re.findall(r"(?<=Quick: ).*}", open_file)
        if len(match_list) == 0:
            return FileNotFoundError("没有符合条件的数据 ...")

        parts_list = [json.loads(file.replace("'", '"')) for file in match_list if file]
        grouped_dict = defaultdict(list)
        for part in parts_list:
            part["query"] = Path(part["query"])
            if group:
                if len(part["query"].parts) == 2:
                    parts = part.pop("total_path"), part.pop("title"), part["query"].name
                else:
                    logger.warning(f"不符合分组规则,使用默认分组 ...")
                    parts = part.pop("total_path"), part.pop("title")
            else:
                parts = part.pop("total_path"), part.pop("title")
            grouped_dict[parts].append(part)

        tasks = [
            Report.ask_invent_report(
                Path(os.path.join(file_name, k[0])), k[1], k[2] if len(k) == 3 else "", parts_list, views_loc
            ) for k, parts_list in grouped_dict.items()
        ]
        merge_result = await asyncio.gather(*tasks)
        total_result = [merge for merge in merge_result]

        if group:
            merged = defaultdict(lambda: defaultdict(list))
            for case in total_result:
                for k, v in case.items():
                    if k != 'case':
                        merged[case['case']][k].append(v)
            total_list = [
                {'case': case, **{k: v for k, v in attrs.items()}} for case, attrs in merged.items()
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

        total_list = [single for single in total_list if single]
        if len(total_list) == 0:
            logger.warning("没有可以汇总的报告 ...")
            return False

        total_html_temp = Template(total_loc).render(
            title=f"{const.DESC}",
            report_time=time.strftime('%Y.%m.%d %H:%M:%S'),
            total_list=total_list
        )
        total_html = os.path.join(file_name, "NexaFlow.html")
        async with aiofiles.open(total_html, "w", encoding="utf-8") as f:
            await f.write(total_html_temp)
            logger.info(f"生成汇总报告: {total_html}")

    @staticmethod
    async def ask_create_report(total_path: Path, title: str, serial: str, parts_list: list, major_loc: str):

        async def deal_with_inform(result):
            handler_list = []
            query = result.get("query", "TimeCost")
            stage = result.get("stage", {"start": 1, "end": 2, "cost": "0.00000"})
            frame = result.get("frame", "")
            extra = result.get("extra", "")
            proto = result.get("proto", "")

            async def handler_frame():
                handler_image_list = []
                for image in os.listdir(os.path.join(total_path, title, query, frame)):
                    image_src = os.path.join(query, frame, image)
                    image_ids = re.search(r"\d+(?=_)", image).group()
                    timestamp = float(re.search(r"(?<=_).+(?=\.)", image).group())
                    handler_image_list.append(
                        {
                            "src": image_src,
                            "frames_id": image_ids,
                            "timestamp": f"{timestamp:.5f}"
                        }
                    )
                handler_image_list.sort(key=lambda x: int(x["frames_id"]))
                return handler_image_list

            async def handler_extra():
                handler_extra_list = []
                for ex in os.listdir(os.path.join(total_path, title, query, extra)):
                    extra_src = os.path.join(query, extra, ex)
                    extra_idx = ex.split("(")[0]
                    handler_extra_list.append(
                        {
                            "src": extra_src,
                            "idx": extra_idx
                        }
                    )
                handler_extra_list.sort(key=lambda x: int(x["idx"].split("(")[0]))
                return handler_extra_list

            data = {"query": query, "stage": stage}

            if extra and proto:
                image_list, extra_list = await asyncio.gather(
                    handler_frame(), handler_extra()
                )
                data["extra_list"] = extra_list
                data["proto"] = os.path.join(query, proto)
            else:
                image_list = await handler_frame()

            data["image_list"] = image_list

            handler_list.append(data)
            return handler_list

        async def handler_start():
            single = {}
            if len(parts_list) > 0:
                tasks = [deal_with_inform(result) for result in parts_list]
                results = await asyncio.gather(*tasks)
                images_list = [ele for res in results for ele in res]

                range_html_temp = Template(major_loc).render(
                    name=const.NAME,
                    title=title,
                    images_list=images_list
                )
                teams = serial if serial else random.randint(10000, 99999)
                range_html = Path(os.path.join(total_path, title, f"{title}_{teams}.html"))
                async with aiofiles.open(range_html, "w", encoding=const.CHARSET) as range_file:
                    await range_file.write(range_html_temp)
                    logger.info(f"生成聚合报告: {range_html.name}")

                cost_list = [cost["stage"]["cost"] for cost in images_list]
                try:
                    avg = sum(map(float, cost_list)) / len(cost_list)
                except ZeroDivisionError:
                    avg = 0.00000
                    logger.warning("未获取到平均值 ...")

                href_path = os.path.join(total_path.name, title, range_html.name)
                single = {
                    "case": title, "cost_list": cost_list, "avg": f"{avg:.5f}", "href": href_path
                }
                logger.debug("Recovery: " + json.dumps(single, ensure_ascii=False))
            else:
                logger.info("没有可以聚合的报告 ...")

            logger.info(f"✪✪✪✪✪✪✪✪✪✪ {title} ✪✪✪✪✪✪✪✪✪✪")
            return single

        return await handler_start()

    @staticmethod
    async def ask_create_total_report(file_name: str, major_loc: str, total_loc: str, group: bool):
        try:
            file_path = os.path.join(file_name, "Nexa_Recovery", "nexaflow.log")
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                open_file = await f.read()
        except FileNotFoundError as e:
            return e

        match_list = re.findall(r"(?<=Restore: ).*}", open_file)
        if len(match_list) == 0:
            return FileNotFoundError("没有符合条件的数据 ...")

        parts_list = [json.loads(file.replace("'", '"')) for file in match_list if file]
        grouped_dict = defaultdict(list)
        for part in parts_list:
            part["query"] = Path(part["query"])
            if group:
                if len(part["query"].parts) == 2:
                    parts = part.pop("total_path"), part.pop("title"), part["query"].name
                else:
                    logger.warning(f"不符合分组规则,使用默认分组 ...")
                    parts = part.pop("total_path"), part.pop("title")
            else:
                parts = part.pop("total_path"), part.pop("title")
            grouped_dict[parts].append(part)

        tasks = [
            Report.ask_create_report(
                Path(os.path.join(file_name, k[0])), k[1], k[2] if len(k) == 3 else "", parts_list, major_loc
            ) for k, parts_list in grouped_dict.items()
        ]
        merge_result = await asyncio.gather(*tasks)
        total_result = [merge for merge in merge_result]

        if group:
            merged = defaultdict(lambda: defaultdict(list))
            for case in total_result:
                for k, v in case.items():
                    if k != 'case':
                        merged[case['case']][k].append(v)
            total_list = [
                {'case': case, **{k: v for k, v in attrs.items()}} for case, attrs in merged.items()
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

        total_list = [single for single in total_list if single]
        if len(total_list) == 0:
            logger.warning("没有可以汇总的报告 ...")
            return False

        total_html_temp = Template(total_loc).render(
            report_time=time.strftime('%Y.%m.%d %H:%M:%S'),
            total_list=total_list
        )
        total_html = os.path.join(file_name, "NexaFlow.html")
        async with aiofiles.open(total_html, "w", encoding=const.CHARSET) as f:
            await f.write(total_html_temp)
            logger.info(f"生成汇总报告: {total_html}")

    @staticmethod
    def draw(
        classifier_result,
        proto_path: str,
        template_file: str,
        compress_rate: float = None,
        target_size: tuple[int, int] = None,
        boost_mode: bool = False,
    ) -> str:

        label_stable: str = "稳定阶段"
        label_unstable: str = "不稳定阶段"
        label_unspecific: str = "不明阶段"

        thumbnail_list: list[dict[str, str]] = list()
        extra_dict: dict[str, str] = dict()

        try:
            stage_range = classifier_result.get_stage_range()
        except AssertionError:
            stage_range = [classifier_result.data]

        image_list = []
        if boost_mode:
            for cur_index in range(len(stage_range)):
                each_range = stage_range[cur_index]
                middle = each_range[len(each_range) // 2]
                if middle.is_stable():
                    label = label_stable
                    image = toolbox.compress_frame(
                        middle.get_data(), compress_rate=compress_rate, target_size=target_size
                    )
                    frame = {
                        "frame_id": middle.frame_id,
                        "timestamp": f"{middle.timestamp:.5f}",
                        "image": toolbox.np2b64str(image)
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
                            i.get_data(), compress_rate=compress_rate, target_size=target_size
                        )
                        frame = {
                            "frame_id": i.frame_id,
                            "timestamp": f"{i.timestamp:.5f}",
                            "image": toolbox.np2b64str(image)
                        }
                        image_list.append(frame)

                first, last = each_range[0], each_range[-1]
                title = (f"{label} "
                         f"区间: {first.frame_id}({first.timestamp:.5f}) - {last.frame_id}({last.timestamp:.5f}) "
                         f"耗时: {last.timestamp - first.timestamp:.5f} "
                         f"分类: {first.stage}")
                thumbnail_list.append({title: image_list})

        else:
            for cur_index in range(len(stage_range)):
                each_range = stage_range[cur_index]
                middle = each_range[len(each_range) // 2]

                if middle.is_stable():
                    label = label_stable
                elif middle.stage == const.UNKNOWN_STAGE_FLAG:
                    label = label_unspecific
                else:
                    label = label_unstable

                if cur_index + 1 < len(stage_range):
                    range_for_display = [*each_range, stage_range[cur_index + 1][0]]
                else:
                    range_for_display = each_range

                for i in range_for_display:
                    image = toolbox.compress_frame(
                        i.get_data(), compress_rate=compress_rate, target_size=target_size
                    )
                    frame = {
                        "frame_id": i.frame_id,
                        "timestamp": f"{i.timestamp:.5f}",
                        "image": toolbox.np2b64str(image)
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

        template_content = Template(template_file).render(
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

        with open(report_path, "w", encoding=const.CHARSET) as fh:
            fh.write(template_content)
        logger.info(f"生成单次报告: {Path(report_path).name}")

        return report_path


if __name__ == '__main__':
    pass
