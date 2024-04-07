import os
import time
from nexaflow import const
from engine.activate import active
from plan.skills.alynex import Alynex
from plan.skills.manage import Manage

AUDIO_DIRS = os.path.join(const.WORK, "audio")
MODELS = os.path.join(const.WORK, "archivix", "molds", "Keras_Gray_W256_H256_00000.h5")
REPORT = os.path.join(const.WORK, "report")
TEMPLATE_MAIN_TOTAL = os.path.join(const.NEXA, "template", "template_main_total.html")
TEMPLATE_MAIN = os.path.join(const.NEXA, "template", "template_main.html")
ALIEN = os.path.join(const.NEXA, "template", "template_alien.html")


def multi_audio_task():
    start_time = time.time()

    application = ""
    activity = ""

    active("INFO")
    manage = Manage()
    alynex = Alynex()
    alynex.activate(MODELS, REPORT)

    device = manage.operate_device("")
    for audio in os.listdir(AUDIO_DIRS):
        alynex.report.title = audio.split(".")[0]
        for _ in range(3):
            alynex.report.query = audio.split(".")[0]
            device.swipe_unlock()
            alynex.record.start_record(
                alynex.report.video_path,
                device.serial
            )

            device.key_event(231)
            device.sleep(1)
            alynex.player.play_audio(os.path.join(AUDIO_DIRS, audio))
            device.sleep(10)

            alynex.record.stop_record()
            device.force_filter(application)
            device.start_app(activity)
            alynex.framix.crop_hook(0, 0.2, 1, 0.8)
            alynex.analyzer(ALIEN)
        alynex.report.create_report(TEMPLATE_MAIN)
    alynex.report.create_total_report(TEMPLATE_MAIN_TOTAL)
    print(f"Total Time Cost: {(time.time() - start_time):.2f} 秒")


if __name__ == '__main__':
    multi_audio_task()
