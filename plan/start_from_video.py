import os
import time
import shutil
from multiprocessing import Pool
from engine.activate import active
from nexaflow import const
from nexaflow.report import Report
from plan.skills.alynex import Alynex

MERGE_TEMPLATE = os.path.join(const.NEXA, "template")
MODELS = os.path.join(const.WORK, "archivix", "molds", "Keras_Gray_W256_H256_00000.h5")
REPORT = os.path.join(const.WORK, "report")
TEMPLATE_MAIN_TOTAL = os.path.join(const.NEXA, "template", "template_main_total.html")
TEMPLATE_MAIN = os.path.join(const.NEXA, "template", "template_main.html")
ALIEN = os.path.join(const.NEXA, "template", "template_alien.html")


def multi_video_task(folder: str) -> str:
    alynex = Alynex()
    alynex.activate(MODELS, REPORT)
    for video in alynex.only_video(os.path.join(const.WORK, "data", folder)):
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
    active("INFO")
    data = ["group_0001", "group_0002"]
    start_time = time.time()

    with Pool(len(data)) as pool:
        results = pool.map(multi_video_task, data)

    Report.merge_report(results, MERGE_TEMPLATE)
    print(f"Total Time Cost: {(time.time() - start_time):.2f} 秒")
