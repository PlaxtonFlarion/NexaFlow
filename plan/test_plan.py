from nexaflow.skills.alynex import Alynex
from nexaflow.skills.device import Device


class TestPlan(object):

    # adb shell dumpsys window | findstr mCurrentFocus
    application: str = "your package name ???"
    activity: str = "your activity name ???"

    def __init__(self, device: Device = None, looper: int = 1):
        self.looper: int = looper
        self.__device: "Device" = device
        self.__alynex: "Alynex" = Alynex()

    async def test_01(self):
        """讲个笑话"""
        query, audio = self.__alynex.player.load_audio("讲个笑话")
        self.__alynex.report.set_title(query)
        for _ in range(self.looper):
            self.__alynex.report.set_query(query)
            await self.__device.ask_swipe_unlock()
            await self.__alynex.record.start_record_silence(
                self.__alynex.report.video_path,
                self.__device.serial
            )

            await self.__device.ask_key_event(231)
            await self.__device.ask_sleep(1)
            await self.__alynex.player.play_audio(audio)
            await self.__device.ask_sleep(2)

            await self.__alynex.record.stop_record()
            await self.__device.ask_force_filter(self.application)
            await self.__device.ask_start_app(self.activity)

            await self.__alynex.framix.crop_hook(0, 0.2, 1, 0.8)
            await self.__alynex.analyzer()
        await self.__alynex.report.create_report()

    async def test_02(self):
        query = "讲个笑话"
        self.__alynex.report.set_title(query)
        for i in range(1):
            self.__alynex.report.set_query(f"{i + 1}_{query}")
            await self.__alynex.framix.crop_hook(0, 0.3, 1, 0.7)
            await self.__alynex.analyzer()
        await self.__alynex.report.create_report()

    async def __aenter__(self):
        # await self.__device.ask_force_filter(self.application)
        # await self.__device.ask_start_app(self.activity)
        # await self.__device.ask_sleep(5)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # await self.__device.ask_force_filter(self.application)
        # await self.__device.ask_force_stop(self.activity)
        await self.__alynex.report.create_total_report()


if __name__ == '__main__':
    pass
