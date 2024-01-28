import os
import time
from nexaflow.constants import Constants
from nexaflow.skills.device import Manage
from nexaflow.skills.alynex import Alynex

AUDIO_DIRS = os.path.join(Constants.WORK, "audio")
MODELS = os.path.join(Constants.WORK, "archivix", "molds", "model.h5")
REPORT = os.path.join(Constants.WORK, "report")
TEMPLATE_MAIN_TOTAL = os.path.join(Constants.NEXA, "template", "template_main_total.html")
TEMPLATE_MAIN = os.path.join(Constants.NEXA, "template", "template_main.html")
ALIEN = os.path.join(Constants.NEXA, "template", "template_alien.html")


def multi_audio_task():
    start_time = time.time()

    application: str = ""
    activity: str = ""

    Constants.initial_logger()
    manage = Manage()
    alynex = Alynex()
    alynex.activate(MODELS, REPORT)

    device = manage.operate_device("")
    for query, audio in alynex.player.load_all_audio(AUDIO_DIRS):
        alynex.report.title = query
        for _ in range(3):
            alynex.report.query = query
            device.ask_swipe_unlock()
            alynex.record.start_record(
                alynex.report.video_path,
                device.serial
            )

            device.ask_key_event(231)
            device.ask_sleep(1)
            alynex.player.play_audio(audio)
            device.ask_sleep(10)

            alynex.record.stop_record()
            device.ask_force_filter(application)
            device.ask_start_app(activity)
            alynex.framix.crop_hook(0, 0.2, 1, 0.8)
            alynex.analyzer(ALIEN)
        alynex.report.create_report(TEMPLATE_MAIN)
    alynex.report.create_total_report(TEMPLATE_MAIN_TOTAL)
    print(f"Total Time Cost: {(time.time() - start_time):.2f} 秒")


if __name__ == '__main__':
    multi_audio_task()
