import os
from nexaflow.constants import Constants
from nexaflow.skills.alynex import Alynex
from nexaflow.skills.device import Device

AUDIO_DIRS: str = os.path.join(Constants.WORK, "audio")


class TestPlan(object):

    # adb shell dumpsys window | findstr mCurrentFocus
    application: str = "your package name ???"
    activity: str = "your activity name ???"

    def __init__(self, device: Device = None, looper: int = 1):
        self.looper: int = looper
        self.__device: "Device" = device
        self.__alynex: "Alynex" = Alynex()
        self.__alynex.activate_report()

    def test_01(self):
        """讲个笑话"""
        query, audio = self.__alynex.player.load_audio(AUDIO_DIRS, "讲个笑话")
        self.__alynex.report.title = query
        for _ in range(self.looper):
            self.__alynex.report.query = query
            self.__device.ask_swipe_unlock()
            self.__alynex.record.start_record(
                self.__alynex.report.video_path,
                self.__device.serial
            )

            self.__device.key_event(231)
            self.__device.sleep(1)
            self.__alynex.player.play_audio(audio)
            self.__device.ask_sleep(2)

            self.__alynex.record.stop_record()
            self.__device.force_filter(self.application)
            self.__device.start_app(self.activity)

            self.__alynex.framix.crop_hook(0, 0.2, 1, 0.8)
            self.__alynex.analyzer()
        self.__alynex.report.create_report()

    def test_02(self):
        query = "讲个笑话"
        self.__alynex.report.title = query
        for i in range(1):
            self.__alynex.report.query = f"{i + 4}_{query}"
            self.__alynex.framix.crop_hook(0, 0.1, 1, 0.9)
            self.__alynex.analyzer()
        self.__alynex.report.create_report()

    def __enter__(self):
        # self.__device.force_filter(self.application)
        # self.__device.start_app(self.activity)
        # self.__device.sleep(5)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.__device.force_filter(self.application)
        # self.__device.force_stop(self.activity)
        self.__alynex.report.create_total_report()


if __name__ == '__main__':
    pass
