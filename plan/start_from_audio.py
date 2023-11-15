import time
import asyncio
from nexaflow.skills.device import Manage
from nexaflow.skills.alynex import Alynex


async def multi_audio_task(serial: str, application: str, activity: str):
    start_time = time.time()
    manage = Manage()
    alynex = Alynex()

    device = manage.operate_device(serial)
    for query, audio in alynex.player.load_all_audio():
        alynex.report.set_title(query)
        for _ in range(5):
            alynex.report.set_query(query)
            await device.ask_swipe_unlock()
            await alynex.record.start_record_silence(
                alynex.report.video_path,
                device.serial
            )

            await device.ask_key_event(231)
            await device.ask_sleep(1)
            await alynex.player.play_audio(audio)
            await device.ask_sleep(2)

            await alynex.record.stop_record()
            await device.ask_force_filter(application)
            await device.ask_start_app(activity)
            await alynex.framix.crop_hook(0, 0.2, 1, 0.8)
            await alynex.analyzer(shift=True)
        await alynex.report.create_report()
    await alynex.report.create_total_report()
    print(f"Total Time Cost: {(time.time() - start_time):.2f} 秒")


if __name__ == '__main__':
    asyncio.run(multi_audio_task("", "", ""))

