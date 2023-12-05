import os
import re
import json
import time
import shutil
import asyncio
import threading
from loguru import logger
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional, Union
from jinja2 import Template, Environment, FileSystemLoader
from nexaflow import constants, toolbox
from nexaflow.constants import Constants
from nexaflow.classifier.base import ClassifierResult

REPORT: str = os.path.join(Constants.WORK, "report")
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

    def __init__(self, total_path: Optional[str] = None, write_log: bool = True):
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

            if total_path:
                self.total_path = total_path
            else:
                # self.total_path = "/Users/acekeppel/PycharmProjects/NexaFlow/report/Nexa_20230822223025"
                self.total_path = os.path.join(REPORT, f"Nexa_{self.clock()}_{os.getpid()}", "Nexa_Collection")
            os.makedirs(self.total_path, exist_ok=True)

            if write_log:
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
        logger.info(f"{'=' * 36} {self.title} {'=' * 36}\n")

    @title.deleter
    def title(self):
        del self.__title

    @property
    def query(self):
        return self.__query

    @query.setter
    def query(self, query: str):
        # self.__query = query
        self.__query = query + "_" + self.clock()
        self.video_path = os.path.join(self.query_path, self.query, "video")
        self.frame_path = os.path.join(self.query_path, self.query, "frame")
        self.extra_path = os.path.join(self.query_path, self.query, "extra")
        os.makedirs(self.video_path, exist_ok=True)
        os.makedirs(self.frame_path, exist_ok=True)
        os.makedirs(self.extra_path, exist_ok=True)
        logger.info(f"{self.query} Start ... {'-' * 60}")

    @query.deleter
    def query(self):
        del self.__query

    def load(self, inform: Optional[Dict[str, Union[str | Dict]]]) -> None:
        if inform:
            self.range_list.append(inform)
        logger.info(f"{self.query} End ... {'-' * 60}\n")

    def create_report(self) -> None:

        async def handler_inform(result):
            handler_list = []
            query = result.get("query", "TimeCost")
            stage = result.get("stage", {"start": 1, "end": 2, "cost": "0.00000"})
            frame = result.get("frame", "")
            extra = result.get("extra", "")
            proto = result.get("proto", "")

            async def handler_frame():
                handler_image_list = []
                for image in os.listdir(frame):
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
                for ex in os.listdir(extra):
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
                    "proto": os.path.join(query, os.path.basename(proto))
                }
            )
            return handler_list

        async def handler_start():
            tasks = [handler_inform(result) for result in self.range_list]
            results = await asyncio.gather(*tasks)
            images_list = [ele for res in results for ele in res]
            loader = FileSystemLoader(os.path.join(Constants.NEXA, "template"))
            environment = Environment(loader=loader)
            template = environment.get_template("template.html")

            html = template.render(title=self.title, images_list=images_list)
            report_html = os.path.join(self.query_path, f"{self.title}.html")
            with open(file=report_html, mode="w", encoding="utf-8") as f:
                f.write(html)
                logger.info(f"生成聚合报告: {report_html}")

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
            logger.info(f"{'=' * 36} {self.title} {'=' * 36}\n\n")

        loop = asyncio.get_event_loop()
        loop.run_until_complete(handler_start())

    def create_total_report(self) -> None:
        loader = FileSystemLoader(os.path.join(Constants.NEXA, "template"))
        environment = Environment(loader=loader)
        template = environment.get_template("overall.html")
        report_time = time.strftime('%Y.%m.%d %H:%M:%S')
        html = template.render(report_time=report_time, total_list=self.total_list)

        total_html_path = os.path.join(os.path.dirname(self.total_path), "NexaFlow.html")
        with open(file=total_html_path, mode="w", encoding="utf-8") as f:
            f.write(html)
            logger.info(f"生成汇总报告: {total_html_path}\n\n")
        self.total_list.clear()

    @staticmethod
    async def ask_create_report(major_loc, title, total_path, query_path, range_list):

        async def handler_inform(result):
            handler_list = []
            query = result.get("query", "TimeCost")
            stage = result.get("stage", {"start": 1, "end": 2, "cost": "0.00000"})
            frame = result.get("frame", "")
            extra = result.get("extra", "")
            proto = result.get("proto", "")

            async def handler_frame():
                handler_image_list = []
                for image in os.listdir(frame):
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
                for ex in os.listdir(extra):
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
                    "proto": os.path.join(query, os.path.basename(proto))
                }
            )
            return handler_list

        async def handler_start():
            tasks = [handler_inform(result) for result in range_list]
            results = await asyncio.gather(*tasks)
            images_list = [ele for res in results for ele in res]

            major_loader = FileSystemLoader(major_loc)
            major_environment = Environment(loader=major_loader)
            major_template = major_environment.get_template("template.html")

            html = major_template.render(title=title, images_list=images_list)
            report_html = os.path.join(query_path, f"{title}.html")
            with open(file=report_html, mode="w", encoding="utf-8") as f:
                f.write(html)
                logger.info(f"生成聚合报告: {report_html}")

            cost_list = [cost['stage']['cost'] for cost in images_list]
            href_path = os.path.join(
                os.path.basename(total_path),
                title,
                os.path.basename(report_html)
            )
            single = {
                "case": title,
                "cost_list": cost_list,
                "avg": f"{sum(map(float, cost_list)) / len(cost_list):.5f}",
                "href": href_path
            }
            logger.debug("Recovery: " + json.dumps(single, ensure_ascii=False))
            logger.info(f"{'=' * 36} {title} {'=' * 36}\n\n")
            return single

        return await handler_start()

    @staticmethod
    async def ask_create_total_report(file_name: str, major_loc: str, loader_total_loc: str):
        report_time = time.strftime('%Y.%m.%d %H:%M:%S')

        with open(file=os.path.join(file_name, "Nexa_Recovery", "nexaflow.log"), mode="r", encoding="utf-8") as f:
            match_list = re.findall(r"(?<=Restore: ).*}", f.read())
            range_list = [json.loads(file.replace("'", '"')) for file in match_list if file]
            grouped_dict = defaultdict(list)
            for part in range_list:
                parts = part.pop("title"), part.pop("total_path"), part.pop("query_path")
                grouped_dict[parts].append(part)

        tasks = [
            Report.ask_create_report(major_loc, title, total_path, query_path, range_list)
            for (title, total_path, query_path), range_list in grouped_dict.items()
        ]
        merge_result = await asyncio.gather(*tasks)
        total_list = [merge for merge in merge_result]

        total_loader = FileSystemLoader(loader_total_loc)
        total_environment = Environment(loader=total_loader)
        total_template = total_environment.get_template("overall.html")

        html = total_template.render(report_time=report_time, total_list=total_list)
        total_html = os.path.join(file_name, "NexaFlow.html")
        with open(file=total_html, mode="w", encoding="utf-8") as f:
            f.write(html)
            logger.info(f"生成汇总报告: {total_html}")

    @staticmethod
    def reset_report(file_name: str) -> None:
        loader = FileSystemLoader(os.path.join(Constants.NEXA, "template"))
        environment = Environment(loader=loader)
        template = environment.get_template("overall.html")
        report_time = time.strftime('%Y.%m.%d %H:%M:%S')

        with open(
                file=os.path.join(REPORT, file_name, "Nexa_Recovery", "nexaflow.log"),
                mode="r", encoding="utf-8"
        ) as f:
            log_restore = re.findall(r"(?<=Recovery: ).*}", f.read())
        total_list = [json.loads(file) for file in log_restore]
        html = template.render(report_time=report_time, total_list=total_list)

        total_html_path = os.path.join(REPORT, file_name, "NexaFlow.html")
        with open(file=total_html_path, mode="w", encoding="utf-8") as f:
            f.write(html)
            logger.info(f"生成汇总报告: {total_html_path}\n\n")

    @staticmethod
    def merge_report(merge_list: List[str]) -> None:
        merge_path = os.path.join(
            os.path.dirname(os.path.dirname(merge_list[0])),
            "Merge_Nexa_" + time.strftime("%Y%m%d%H%M%S"),
            "Nexa_Collection"
        )
        os.makedirs(merge_path, exist_ok=True)
        log_restore = []
        for merge in merge_list:
            logs = os.path.join(os.path.dirname(merge), "Nexa_Recovery", "nexaflow.log")
            with open(file=logs, mode="r", encoding="utf-8") as f:
                log_restore.extend(re.findall(r"(?<=Recovery: ).*}", f.read()))
            shutil.copytree(
                merge, merge_path, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("NexaFlow.html", "nexaflow.log")
            )

        loader = FileSystemLoader(os.path.join(Constants.NEXA, "template"))
        environment = Environment(loader=loader)
        template = environment.get_template("overall.html")
        report_time = time.strftime('%Y.%m.%d %H:%M:%S')
        total_list = [json.loads(file) for file in log_restore]
        html = template.render(report_time=report_time, total_list=total_list)

        total_html_path = os.path.join(os.path.dirname(merge_path), "NexaFlow.html")
        with open(file=total_html_path, mode="w", encoding="utf-8") as f:
            f.write(html)
            logger.info(f"合并汇总报告: {total_html_path}\n\n")

    @staticmethod
    def draw(
        classifier_result: "ClassifierResult",
        proto_path: str,
        compress_rate: float = None,
        target_size: Tuple[int, int] = None,
        boost_mode: bool = False,
        framix_template: str = None
    ) -> str:

        label_stable: str = "稳定阶段"
        label_unstable: str = "不稳定阶段"
        label_unspecific: str = "不明阶段"

        thumbnail_list: List[Dict[str, str]] = list()
        extra_dict: Dict[str, str] = dict()

        if not compress_rate:
            compress_rate = 0.2

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

        def get_template() -> str:
            template_dirs = os.path.join(Constants.NEXA, "template")
            template_path = os.path.join(template_dirs, "extra.html")
            with open(template_path, encoding=constants.CHARSET) as t:
                template_file = t.read()
            return template_file

        if framix_template:
            template = Template(framix_template)
        else:
            template = Template(get_template())

        template_content = template.render(
            thumbnail_list=thumbnail_list,
            extras=extra_dict,
            background_color=constants.BACKGROUND_COLOR,
            cost_dict=cost_dict,
            timestamp=timestamp,
            version_code="1.0.0"
        )

        default_name = f"{timestamp}.html"
        if os.path.isdir(proto_path):
            report_path = os.path.join(proto_path, default_name)
        else:
            report_path = proto_path

        with open(report_path, "w", encoding=constants.CHARSET) as fh:
            fh.write(template_content)
        logger.info(f"生成单次报告: {report_path}")

        return report_path


if __name__ == '__main__':
    pass
