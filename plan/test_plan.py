import os
from engine.activate import active
from engine.alynex import Alynex
from nexaflow import const
from nexaflow.report import Report
from plan.skills.device import Device
from plan.skills.manage import Manage


class TestPlan(object):

    # adb shell dumpsys window | findstr mCurrentFocus
    application = "your package name ???"
    activity = "your activity name ???"

    def __init__(self, device: Device = None):
        self.device: "Device" = device
        self.alynex: "Alynex" = Alynex(Report(const.CREDO), const.MODEL)

    def test_01(self):
        audio = os.path.join(const.AUDIO, query := "讲个笑话")
        self.alynex.report.title = query
        for _ in range(1):
            self.alynex.report.query = query
            self.device.swipe_unlock()
            self.alynex.record.start_record(
                self.alynex.report.video_path, self.device.serial
            )

            self.device.key_event(231)
            self.device.sleep(1)
            self.alynex.player.play_audio(audio)
            self.device.sleep(2)

            self.alynex.record.stop_record()
            self.device.force_filter(self.application)
            self.device.start_app(self.activity)

            self.alynex.crop_hook(0, 0.2, 1, 0.8)
            self.alynex.analyzer(const.TEMPLATE_ATOM_TOTAL)
        self.alynex.report.create_report(const.TEMPLATE_MAIN)

    def test_02(self):
        query = "讲个笑话"
        self.alynex.report.title = query
        for i in range(1):
            self.alynex.report.query = f"{i + 4}_{query}"
            self.alynex.crop_hook(0, 0.1, 1, 0.9)
            self.alynex.analyzer(const.TEMPLATE_ATOM_TOTAL)
        self.alynex.report.create_report(const.TEMPLATE_MAIN)

    def __enter__(self):
        # self.device.force_filter(self.application)
        # self.device.start_app(self.activity)
        # self.device.sleep(5)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.device.force_filter(self.application)
        # self.device.force_stop(self.activity)
        self.alynex.report.create_total_report(const.TEMPLATE_MAIN_TOTAL)


if __name__ == '__main__':
    # pip freeze > requirements.txt
    # pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==2.14.0
    # pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade tensorflow
    # pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
    active("INFO")
    manage = Manage()

    with TestPlan(manage.Phone) as test:
        test.test_02()
