import os
import time
import shutil
from multiprocessing import Pool
from nexaflow.constants import Constants
from nexaflow.skills.alynex import Alynex
from nexaflow.skills.report import Report

MERGE_TEMPLATE = os.path.join(Constants.NEXA, "template")
MODELS = os.path.join(Constants.WORK, "model", "model.h5")
REPORT = os.path.join(Constants.WORK, "report")
TEMPLATE_MAIN_TOTAL = os.path.join(Constants.NEXA, "template", "template_main_total.html")
TEMPLATE_MAIN = os.path.join(Constants.NEXA, "template", "template_main.html")
ALIEN = os.path.join(Constants.NEXA, "template", "template_alien.html")


def multi_video_task(folder: str) -> str:
    alynex = Alynex()
    alynex.activate(MODELS, REPORT)
    for video in alynex.only_video(os.path.join(Constants.WORK, "data", folder)):
        alynex.report.title = video.title
        for path in video.sheet:
            alynex.report.query = os.path.basename(path).split(".")[0]
            shutil.copy(path, alynex.report.video_path)
            alynex.framix.crop_hook(0, 0.2, 1, 0.8)
            alynex.analyzer(ALIEN)
        alynex.report.create_report(TEMPLATE_MAIN)
    alynex.report.create_total_report(TEMPLATE_MAIN_TOTAL)
    return alynex.report.total_path


if __name__ == '__main__':
    Constants.initial_logger()
    data = ["group_0001", "group_0002"]
    start_time = time.time()

    with Pool(len(data)) as pool:
        results = pool.map(multi_video_task, data)

    Report.merge_report(results, MERGE_TEMPLATE)
    print(f"Total Time Cost: {(time.time() - start_time):.2f} 秒")
