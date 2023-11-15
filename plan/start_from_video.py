import os
import time
import shutil
import asyncio
from multiprocessing import Pool
from nexaflow.skills.alynex import Alynex
from nexaflow.skills.report import Report


async def task(folder: str) -> str:

    async with Alynex() as alynex:
        for video in alynex.only_video(folder):
            alynex.report.set_title(video.title)
            for path in video.sheet:
                alynex.report.set_query(os.path.basename(path).split(".")[0])
                shutil.copy(path, alynex.report.video_path)
                await alynex.framix.crop_hook(0, 0.2, 1, 0.8)
                await alynex.analyzer()
            await alynex.report.create_report()
        await alynex.report.create_total_report()
        return alynex.report.total_path


def main(dirs):
    return asyncio.run(task(dirs))


def multi_video_task(data_dirs: list[str]):
    start_time = time.time()

    with Pool(len(data_dirs)) as pool:
        results = pool.map(main, data)

    Report.merge_report(results)
    print(f"Total Time Cost: {(time.time() - start_time):.2f} 秒")


if __name__ == '__main__':
    data = ["202301", "202302"]
    multi_video_task(data)
