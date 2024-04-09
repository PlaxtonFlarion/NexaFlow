import os
import time
from nexaflow import const
from engine.activate import active
from engine.alynex import Alynex
from nexaflow.report import Report
from plan.skills.manage import Manage


def multi_audio_task():
    start_time = time.time()

    application = ""
    activity = ""

    active("INFO")
    manage = Manage()
    alynex = Alynex(Report(const.CREDO), const.MODEL)

    device = manage.operate_device("")
    for audio in os.listdir(const.AUDIO):
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
            alynex.player.play_audio(os.path.join(const.AUDIO, audio))
            device.sleep(10)

            alynex.record.stop_record()
            device.force_filter(application)
            device.start_app(activity)
            alynex.crop_hook(0, 0.2, 1, 0.8)
            alynex.analyzer(const.TEMPLATE_ATOM_TOTAL)
        alynex.report.create_report(const.TEMPLATE_MAIN)
    alynex.report.create_total_report(const.TEMPLATE_MAIN_TOTAL)
    print(f"Total Time Cost: {(time.time() - start_time):.2f} 秒")


if __name__ == '__main__':
    multi_audio_task()
