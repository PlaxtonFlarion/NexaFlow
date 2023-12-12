import os
import time
import shutil
from multiprocessing import Pool
from nexaflow.constants import Constants
from nexaflow.skills.alynex import Alynex
from nexaflow.skills.report import Report

MERGE_TEMPLATE = os.path.join(Constants.NEXA, "template")


def multi_video_task(folder: str) -> str:
    alynex = Alynex()
    for video in alynex.only_video(folder):
        alynex.report.title = video.title
        for path in video.sheet:
            alynex.report.query = os.path.basename(path).split(".")[0]
            shutil.copy(path, alynex.report.video_path)
            alynex.framix.crop_hook(0, 0.2, 1, 0.8)
            alynex.analyzer()
        alynex.report.create_report()
    alynex.report.create_total_report()
    return alynex.report.total_path


if __name__ == '__main__':
    Constants.initial_logger()
    data = ["group_0001", "group_0002"]
    start_time = time.time()

    with Pool(len(data)) as pool:
        results = pool.map(multi_video_task, data)

    Report.merge_report(results, MERGE_TEMPLATE)
    print(f"Total Time Cost: {(time.time() - start_time):.2f} 秒")
