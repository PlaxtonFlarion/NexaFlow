import os
import time
import shutil
from multiprocessing import Pool
from engine.activate import active
from nexaflow import const
from nexaflow.report import Report
from plan.skills.alynex import Alynex


def multi_video_task(folder: str) -> str:
    alynex = Alynex(const.MODEL, Report(const.CREDO))

    for video in alynex.only_video(os.path.join(const.ARRAY, folder)):
        alynex.report.title = video.title
        for path in video.sheet:
            alynex.report.query = os.path.basename(path).split(".")[0]
            shutil.copy(path, alynex.report.video_path)
            alynex.framix.crop_hook(0, 0.2, 1, 0.8)
            alynex.analyzer(const.ALIEN)
        alynex.report.create_report(const.TEMPLATE_MAIN)
    alynex.report.create_total_report(const.TEMPLATE_MAIN_TOTAL)
    return alynex.report.total_path


if __name__ == '__main__':
    active("INFO")
    data = ["group_0001", "group_0002"]
    start_time = time.time()

    with Pool(len(data)) as pool:
        results = pool.map(multi_video_task, data)

    Report.merge_report(results, const.TEMPLATE)
    print(f"Total Time Cost: {(time.time() - start_time):.2f} 秒")
