import os
import re
from engine.terminal import Terminal


class Switch(object):

    @staticmethod
    async def ask_audio_reform(ffmpeg, src: str, dst: str) -> None:
        cmd = [ffmpeg, "-i", src, "-ar", "44100", "-b:a", "128k", dst]
        await Terminal.cmd_line(*cmd)

    @staticmethod
    async def ask_video_reform(ffmpeg, src: str, dst: str) -> None:
        cmd = [ffmpeg, "-i", src, "-r", "60", dst]
        await Terminal.cmd_line(*cmd)

    @staticmethod
    async def ask_video_change(ffmpeg, fps: int, src: str, dst: str, **kwargs) -> None:
        start = kwargs.get("start", None)
        close = kwargs.get("close", None)
        limit = kwargs.get("limit", None)

        cmd = [ffmpeg]

        if start:
            cmd += ["-ss", start]
        if close:
            cmd += ["-to", close]
        elif limit:
            cmd += ["-t", limit]
        cmd += ["-i", src]
        cmd += ["-vf", f"fps={fps}", "-c:v", "libx264", "-crf", "18", "-c:a", "copy", dst]

        await Terminal.cmd_line(*cmd)

    @staticmethod
    async def ask_video_detach(ffmpeg, video_filter: list, src: str, dst: str, **kwargs) -> None:
        start = kwargs.get("start", None)
        close = kwargs.get("close", None)
        limit = kwargs.get("limit", None)

        cmd = [ffmpeg]

        if start:
            cmd += ["-ss", start]
        if close:
            cmd += ["-to", close]
        elif limit:
            cmd += ["-t", limit]
        cmd += ["-i", src]
        cmd += ["-vf", ",".join(video_filter), f"{os.path.join(dst, 'frame_%05d.png')}"]

        await Terminal.cmd_line(*cmd)

    @staticmethod
    async def ask_video_tailor(ffmpeg, src: str, dst: str, **kwargs) -> None:
        start = kwargs.get("start", None)
        close = kwargs.get("close", None)
        limit = kwargs.get("limit", None)

        cmd = [ffmpeg]

        if start:
            cmd += ["-ss", start]
        if close:
            cmd += ["-to", close]
        elif limit:
            cmd += ["-t", limit]
        cmd += ["-i", src]
        cmd += ["-c", "copy", dst]

        await Terminal.cmd_line(*cmd)

    @staticmethod
    async def ask_video_length(ffprobe, src: str) -> float | Exception:
        cmd = [
            ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", "-i", src
        ]
        result = await Terminal.cmd_line(*cmd)
        try:
            fmt_result = float(result.strip())
        except ValueError as e:
            return e
        return fmt_result

    @staticmethod
    async def ask_video_larger(ffprobe, src: str) -> tuple[int, int] | Exception:
        cmd = [
            ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height", "-of", "default=noprint_wrappers=1", src
        ]
        result = await Terminal.cmd_line(*cmd)
        match_w = re.search(r"(?<=width=)\d+", result)
        match_h = re.search(r"(?<=height=)\d+", result)
        try:
            w, h = int(match_w.group()), int(match_h.group())
        except (AttributeError, ValueError) as e:
            return e
        return w, h


if __name__ == '__main__':
    pass
