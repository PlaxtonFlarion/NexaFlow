import os
import re
import json
import time
import shutil
import asyncio
import aiofiles
import threading
from loguru import logger
from jinja2 import Template
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Tuple, Optional, Union
from nexaflow import constants, toolbox

FORMAT: str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>"


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

            self.__title: str = ""
            self.__query: str = ""
            self.query_path: str = ""
            self.video_path: str = ""
            self.frame_path: str = ""
            self.extra_path: str = ""

            self.range_list: list[dict] = []
            self.total_list: list[dict] = []

            self.total_path = os.path.join(total_path, f"Nexa_{self.clock()}_{os.getpid()}", "Nexa_Collection")
            # self.total_path = "/Users/acekeppel/PycharmProjects/NexaFlow/report/Nexa_20230822223025/Nexa_Collection"
            os.makedirs(self.total_path, exist_ok=True)

            self.reset_path = os.path.join(os.path.dirname(self.total_path), "Nexa_Recovery")
            os.makedirs(self.reset_path, exist_ok=True)
            log_papers = os.path.join(self.reset_path, "nexaflow.log")
            logger.add(log_papers, format=FORMAT, level="DEBUG")

    @property
    def proto_path(self) -> str:
        return os.path.join(self.query_path, self.query)

    @property
    def title(self):
        return self.__title

    @title.setter
    def title(self, title: str):
        self.__title = title
        self.query_path = os.path.join(self.total_path, self.title)
        os.makedirs(self.query_path, exist_ok=True)
        logger.info(f"✪✪✪✪✪✪✪✪✪✪ {self.title} ✪✪✪✪✪✪✪✪✪✪\n")

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
    def get_template(template_path) -> str:
        with open(template_path, encoding=constants.CHARSET) as t:
            template_file = t.read()
        return template_file

    def load(self, inform: Optional[Dict[str, Union[str | Dict]]]) -> None:
        if inform:
            self.range_list.append(inform)
        logger.info(f"End -> {inform.get('query', '......')}\n")

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

        if len(self.range_list) > 0:
            if len(self.range_list) == 1:
                images_list = start_create(self.range_list[0])
            else:
                with ThreadPoolExecutor() as executor:
                    future = executor.map(start_create, self.range_list)
                images_list = [i for f in future for i in f]

            template = Template(self.get_template(template_file))
            html = template.render(
                title=self.title,
                images_list=images_list
            )
            report_html = os.path.join(self.query_path, f"{self.title}.html")
            with open(file=report_html, mode="w", encoding="utf-8") as f:
                f.write(html)
                logger.info(f"生成聚合报告: {os.path.basename(report_html)}")

            cost_list = [cost['stage']['cost'] for cost in images_list]
            href_path = os.path.join(
                os.path.basename(self.total_path),
                self.title,
                os.path.basename(report_html)
            )
            single = {
                "case": self.title,
                "cost_list": cost_list,
                "avg": f"{sum(map(float, cost_list)) / len(cost_list):.5f}",
                "href": href_path
            }
            logger.debug("Recovery: " + json.dumps(single, ensure_ascii=False))
            self.total_list.append(single)
            self.range_list.clear()
        else:
            logger.info("没有可以聚合的报告 ...")

        logger.info(f"✪✪✪✪✪✪✪✪✪✪ {self.title} ✪✪✪✪✪✪✪✪✪✪\n\n")

    def create_total_report(self, template_file: str) -> None:
        if len(self.total_list) > 0:
            template = Template(self.get_template(template_file))
            html = template.render(
                report_time=time.strftime('%Y.%m.%d %H:%M:%S'),
                total_list=self.total_list
            )
            total_html_path = os.path.join(os.path.dirname(self.total_path), "NexaFlow.html")
            with open(file=total_html_path, mode="w", encoding="utf-8") as f:
                f.write(html)
                logger.info(f"生成汇总报告: {total_html_path}\n\n")
            self.total_list.clear()
        else:
            logger.info("没有可以汇总的报告 ...")

    @staticmethod
    def reset_report(file_name: str, template_file: str) -> None:
        with open(file=os.path.join(file_name, "Nexa_Recovery", "nexaflow.log"), mode="r", encoding="utf-8") as f:
            log_restore = re.findall(r"(?<=Recovery: ).*}", f.read())

        template = Template(template_file)
        total_list = [json.loads(file) for file in log_restore]
        html = template.render(
            report_time=time.strftime('%Y.%m.%d %H:%M:%S'),
            total_list=total_list
        )
        total_html_path = os.path.join(file_name, "NexaFlow.html")
        with open(file=total_html_path, mode="w", encoding="utf-8") as f:
            f.write(html)
            logger.info(f"生成汇总报告: {total_html_path}\n\n")

    @staticmethod
    def merge_report(merge_list: List[str], merge_loc: str, quick: bool) -> None:
        merge_path = os.path.join(
            os.path.dirname(os.path.dirname(merge_list[0])),
            "Merge_Report_" + time.strftime("%Y%m%d%H%M%S"),
            "Nexa_Collection"
        )
        os.makedirs(merge_path, exist_ok=True)

        pattern = "View" if quick else "Recovery"

        log_restore = []
        for merge in merge_list:
            logs = os.path.join(os.path.dirname(merge), "Nexa_Recovery", "nexaflow.log")
            with open(file=logs, mode="r", encoding="utf-8") as f:
                log_restore.extend(re.findall(fr"(?<={pattern}: ).*}}", f.read()))
            shutil.copytree(
                merge, merge_path, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("NexaFlow.html", "nexaflow.log")
            )

        template = Template(merge_loc)
        total_list = [json.loads(file) for file in log_restore]
        html = template.render(
            report_time=time.strftime("%Y.%m.%d %H:%M:%S"),
            total_list=total_list
        )

        total_html_path = os.path.join(os.path.dirname(merge_path), "NexaFlow.html")
        with open(file=total_html_path, mode="w", encoding="utf-8") as f:
            f.write(html)
            logger.info(f"合并汇总报告: {total_html_path}\n\n")

    @staticmethod
    async def ask_invent_report(total_path: str, title: str, query_path: str, parts_list: list, views_loc: str):

        async def handler_inform(result):
            handler_list = []
            query = result.get("query", "query")
            frame = result.get("frame", "frame")
            stage = result.get("stage", {"start": 0, "end": 0, "cost": 0})

            async def handler_frame():
                handler_image_list = []
                for img in os.listdir(
                        os.path.join(
                            query_path, query, os.path.basename(frame)
                        )
                ):
                    img_idx = int(re.search(r"(?<=frame_)\d+", img).group())
                    img_src = os.path.join(query, os.path.basename(frame), img)
                    handler_image_list.append(
                        {
                            "src": img_src,
                            "frames_id": img_idx,
                        }
                    )
                handler_image_list.sort(key=lambda x: x["frames_id"])
                return handler_image_list

            image_list = await handler_frame()

            handler_list.append(
                {
                    "query": query,
                    "stage": stage,
                    "image_list": image_list
                }
            )
            return handler_list

        async def handler_start():
            single = {}
            if len(parts_list) > 0:
                tasks = [handler_inform(result) for result in parts_list]
                results = await asyncio.gather(*tasks)
                images_list = [ele for res in results for ele in res]

                major_template = Template(views_loc)
                range_html_temp = major_template.render(
                    title="Framix",
                    report_time=time.strftime('%Y.%m.%d %H:%M:%S'),
                    images_list=images_list
                )
                range_html = os.path.join(query_path, f"{title}.html")
                async with aiofiles.open(file=range_html, mode="w", encoding="utf-8") as range_file:
                    await range_file.write(range_html_temp)
                    logger.info(f"生成聚合报告: {os.path.basename(range_html)}")

                cost_list = [cost['stage']['cost'] for cost in images_list]
                href_path = os.path.join(
                    os.path.basename(total_path),
                    title,
                    os.path.basename(range_html)
                )
                single = {
                    "case": title,
                    "cost_list": cost_list,
                    "href": href_path
                }
                logger.debug("View: " + json.dumps(single, ensure_ascii=False))
            else:
                logger.info("没有可以聚合的报告 ...")

            logger.info(f"✪✪✪✪✪✪✪✪✪✪ {title} ✪✪✪✪✪✪✪✪✪✪\n\n")
            return single

        return await handler_start()

    @staticmethod
    async def ask_invent_total_report(file_name: str, views_loc: str, total_loc: str):
        try:
            file_path = os.path.join(file_name, "Nexa_Recovery", "nexaflow.log")
            async with aiofiles.open(file=file_path, mode="r", encoding="utf-8") as f:
                open_file = await f.read()
        except FileNotFoundError as e:
            return e
        else:
            match_list = re.findall(r"(?<=Quick: ).*}", open_file)
            parts_list = [json.loads(file.replace("'", '"')) for file in match_list if file]
            grouped_dict = defaultdict(list)
            for part in parts_list:
                parts = part.pop("total_path"), part.pop("title"), part.pop("query_path")
                grouped_dict[parts].append(part)

            tasks = [
                Report.ask_invent_report(
                    os.path.join(file_name, os.path.basename(total_path)),
                    title,
                    os.path.join(file_name, os.path.basename(total_path), title),
                    parts_list,
                    views_loc
                )
                for (total_path, title, query_path), parts_list in grouped_dict.items()
            ]
            merge_result = await asyncio.gather(*tasks)
            total_list = [merge for merge in merge_result]

            if len(total_list) > 0:
                total_template = Template(total_loc)
                total_html_temp = total_template.render(
                    report_time=time.strftime('%Y.%m.%d %H:%M:%S'),
                    total_list=total_list
                )
                total_html = os.path.join(file_name, "NexaFlow_Quick_View.html")
                async with aiofiles.open(file=total_html, mode="w", encoding="utf-8") as total_file:
                    await total_file.write(total_html_temp)
                    logger.info(f"生成汇总报告: {total_html}")
            else:
                logger.info("没有可以汇总的报告 ...")

    @staticmethod
    async def ask_create_report(total_path: str, title: str, query_path: str, parts_list: list, major_loc: str):

        async def handler_inform(result):
            handler_list = []
            query = result.get("query", "TimeCost")
            stage = result.get("stage", {"start": 1, "end": 2, "cost": "0.00000"})
            frame = result.get("frame", "")
            extra = result.get("extra", "")
            proto = result.get("proto", "")

            async def handler_frame():
                handler_image_list = []
                for image in os.listdir(
                        os.path.join(
                            query_path, query, os.path.basename(frame)
                        )
                ):
                    image_src = os.path.join(query, "frame", image)
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
                for ex in os.listdir(
                        os.path.join(
                            query_path, query, os.path.basename(extra)
                        )
                ):
                    extra_src = os.path.join(query, "extra", ex)
                    extra_idx = ex.split("(")[0]
                    handler_extra_list.append(
                        {
                            "src": extra_src,
                            "idx": extra_idx
                        }
                    )
                handler_extra_list.sort(key=lambda x: int(x["idx"].split("(")[0]))
                return handler_extra_list

            image_list, extra_list = await asyncio.gather(
                handler_frame(), handler_extra()
            )

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

        async def handler_start():
            single = {}
            if len(parts_list) > 0:
                tasks = [handler_inform(result) for result in parts_list]
                results = await asyncio.gather(*tasks)
                images_list = [ele for res in results for ele in res]

                major_template = Template(major_loc)
                range_html_temp = major_template.render(
                    title=title,
                    images_list=images_list,
                )
                range_html = os.path.join(query_path, f"{title}.html")
                async with aiofiles.open(file=range_html, mode="w", encoding="utf-8") as range_file:
                    await range_file.write(range_html_temp)
                    logger.info(f"生成聚合报告: {os.path.basename(range_html)}")

                cost_list = [cost['stage']['cost'] for cost in images_list]
                href_path = os.path.join(
                    os.path.basename(total_path),
                    title,
                    os.path.basename(range_html)
                )
                single = {
                    "case": title,
                    "cost_list": cost_list,
                    "avg": f"{sum(map(float, cost_list)) / len(cost_list):.5f}",
                    "href": href_path
                }
                logger.debug("Recovery: " + json.dumps(single, ensure_ascii=False))
            else:
                logger.info("没有可以聚合的报告 ...")

            logger.info(f"✪✪✪✪✪✪✪✪✪✪ {title} ✪✪✪✪✪✪✪✪✪✪\n\n")
            return single

        return await handler_start()

    @staticmethod
    async def ask_create_total_report(file_name: str, major_loc: str, total_loc: str):
        try:
            file_path = os.path.join(file_name, "Nexa_Recovery", "nexaflow.log")
            async with aiofiles.open(file=file_path, mode="r", encoding="utf-8") as f:
                open_file = await f.read()
        except FileNotFoundError as e:
            return e
        else:
            match_list = re.findall(r"(?<=Restore: ).*}", open_file)
            parts_list = [json.loads(file.replace("'", '"')) for file in match_list if file]
            grouped_dict = defaultdict(list)
            for part in parts_list:
                parts = part.pop("total_path"), part.pop("title"), part.pop("query_path")
                grouped_dict[parts].append(part)

            tasks = [
                Report.ask_create_report(
                    os.path.join(file_name, os.path.basename(total_path)),
                    title,
                    os.path.join(file_name, os.path.basename(total_path), title),
                    parts_list,
                    major_loc,
                )
                for (total_path, title, query_path), parts_list in grouped_dict.items()
            ]
            merge_result = await asyncio.gather(*tasks)
            total_list = [merge for merge in merge_result]

            if len(total_list) > 0:
                total_template = Template(total_loc)
                total_html_temp = total_template.render(
                    report_time=time.strftime('%Y.%m.%d %H:%M:%S'),
                    total_list=total_list
                )
                total_html = os.path.join(file_name, "NexaFlow.html")
                async with aiofiles.open(file=total_html, mode="w", encoding="utf-8") as total_file:
                    await total_file.write(total_html_temp)
                    logger.info(f"生成汇总报告: {total_html}")
            else:
                logger.info("没有可以汇总的报告 ...")

    @staticmethod
    def draw(
        classifier_result,
        proto_path: str,
        template_file: str,
        compress_rate: float = None,
        target_size: Tuple[int, int] = None,
        boost_mode: bool = False,
    ) -> str:

        label_stable: str = "稳定阶段"
        label_unstable: str = "不稳定阶段"
        label_unspecific: str = "不明阶段"

        thumbnail_list: List[Dict[str, str]] = list()
        extra_dict: Dict[str, str] = dict()

        # if not compress_rate:
        #     compress_rate = 0.2

        try:
            stage_range = classifier_result.get_stage_range()
        except AssertionError:
            stage_range = [classifier_result.data]

        if boost_mode:
            for cur_index in range(len(stage_range)):
                each = stage_range[cur_index]
                middle = each[len(each) // 2]
                image_list = []
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
                    if middle.stage == constants.UNKNOWN_STAGE_FLAG:
                        label = label_unspecific
                    else:
                        label = label_unstable

                    if cur_index + 1 < len(stage_range):
                        new_each = [*each, stage_range[cur_index + 1][0]]
                    else:
                        new_each = each

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

                first, last = each[0], each[-1]
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
                elif middle.stage == constants.UNKNOWN_STAGE_FLAG:
                    label = label_unspecific
                else:
                    label = label_unstable

                if cur_index + 1 < len(stage_range):
                    range_for_display = [*each_range, stage_range[cur_index + 1][0]]
                else:
                    range_for_display = each_range

                image_list = []
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

        template = Template(template_file)
        template_content = template.render(
            thumbnail_list=thumbnail_list,
            extras=extra_dict,
            background_color=constants.BACKGROUND_COLOR,
            cost_dict=cost_dict,
            timestamp=timestamp,
            version_code="0.1.0-beta"
        )

        default_name = f"{timestamp}.html"
        if os.path.isdir(proto_path):
            report_path = os.path.join(proto_path, default_name)
        else:
            report_path = proto_path

        with open(report_path, "w", encoding=constants.CHARSET) as fh:
            fh.write(template_content)
        logger.info(f"生成单次报告: {os.path.basename(report_path)}")

        return report_path


if __name__ == '__main__':
    pass
