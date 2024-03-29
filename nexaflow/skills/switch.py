import os
import time
import random
from nexaflow.terminal import Terminal


class Switch(object):

    _filters: [str] = ["fps=60"]

    def __init__(self, ffmpeg: str = None, ffprobe: str = None):
        self.__ffmpeg = ffmpeg if ffmpeg else "ffmpeg"
        self.__ffprobe = ffprobe if ffprobe else "ffprobe"

    @property
    def filters(self):
        return self._filters

    @filters.setter
    def filters(self, value: list[str]):
        self._filters = value

    async def ask_audio_reform(self, src: str, dst: str) -> None:
        """
        调整mp3编码格式为标准mp3
        :param src: 原音频路径
        :param dst: 新音频路径
        """
        cmd = [self.__ffmpeg, "-i", src, "-ar", "44100", "-b:a", "128k", dst]
        await Terminal.cmd_line(*cmd)

    async def ask_video_reform(self, src: str, dst: str) -> None:
        """
        转换视频格式
        :param src: 原始视频路径
        :param dst: 新视频路径
        """
        cmd = [self.__ffmpeg, "-i", src, "-r", "60", dst]
        await Terminal.cmd_line(*cmd)

    async def ask_video_change(self, src: str, dst: str) -> None:
        """
        调整视频
        :param src: 原视频路径
        :param dst: 新视频路径
        """
        cmd = [
            self.__ffmpeg, "-i", src, "-vf", "".join(self.filters), "-c:v",
            "libx264", "-crf", "18", "-c:a", "copy", dst
        ]
        await Terminal.cmd_line(*cmd)

    async def ask_video_tailor(self, src: str, dst: str, start: str, end: str) -> None:
        """
        截取视频
        :param src: 原视频路径
        :param dst: 新视频路径
        :param start: 开始
        :param end: 结束
        """
        before = os.path.basename(src).split(".")[0]
        after = os.path.basename(src).split(".")[-1]
        target = os.path.join(
            dst,
            f"{before}_{time.strftime('%Y%m%d%H%M%S')}_{random.randint(100, 999)}.{after}"
        )
        cmd = [self.__ffmpeg, "-i", src, "-ss", start, "-t", end, "-c", "copy", target]
        await Terminal.cmd_line(*cmd)

    async def ask_video_cutter(self, src: str, dst: str, start: str, end: str) -> None:
        """
        流式截取视频
        :param src: 原视频路径
        :param dst: 新视频路径
        :param start: 开始
        :param end: 结束
        """
        before = os.path.basename(src).split(".")[0]
        after = os.path.basename(src).split(".")[-1]
        target = os.path.join(
            dst,
            f"{before}_{time.strftime('%Y%m%d%H%M%S')}_{random.randint(100, 999)}.{after}"
        )
        cmd = [
            self.__ffmpeg, "-i", src,
            "-ss", start, "-t", end,
            "-vf", "".join(self.filters),
            "-c:v", "libx264", "-crf", "18", "-c:a", "copy", target
        ]
        await Terminal.cmd_line(*cmd)

    async def ask_video_detach(self, src: str, dst: str):
        cmd = [
            self.__ffmpeg, "-i", src,
            "-vf", "".join(self.filters),
            f"{os.path.join(dst, 'frame_%05d.png')}"
        ]
        await Terminal.cmd_line(*cmd)

    async def ask_video_length(self, src: str) -> float:
        """
        查看视频的时间长度
        :param src: 原视频路径
        :return: 视频时间长度
        """
        cmd = [
            self.__ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", "-i", src
        ]
        result = await Terminal.cmd_line(*cmd)
        return float(result.strip())


if __name__ == '__main__':
    pass
