import os
import re
from typing import Optional
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

    @staticmethod
    async def ask_magic_point(
            start: Optional[int | float],
            close: Optional[int | float],
            limit: Optional[int | float],
            duration: Optional[int | float]
    ) -> tuple[Optional[int | float], Optional[int | float], Optional[int | float]]:

        start_point = close_point = limit_point = None

        if not duration or duration <= 0:
            return None, None, None

        if start:
            if 0 <= start <= duration:
                start_point = start
            else:
                start_point = 0

        if close:
            min_start = start_point if start_point else 0
            if 0 <= close <= duration:
                close_point = max(min_start, close)
            else:
                close_point = duration

            if close_point - min_start < 0.09:
                return None, None, None

        elif limit:
            if start_point:
                if limit >= 0 and start_point + limit <= duration:
                    limit_point = limit
                else:
                    limit_point = duration - start_point
            else:
                limit_point = min(limit, duration)

            if limit_point < 0.09:
                return None, None, None

        return start_point, close_point, limit_point

    @staticmethod
    async def ask_magic_frame(
            original_frame_size: tuple, entrance_frame_size: tuple
    ) -> tuple[int, int, float]:

        original_w, original_h = original_frame_size
        original_ratio = original_w / original_h

        if original_frame_size == entrance_frame_size:
            return original_w, original_h, original_ratio

        frame_w, frame_h = entrance_frame_size
        max_w = max(original_w * 0.1, min(frame_w, original_w))
        max_h = max(original_h * 0.1, min(frame_h, original_h))

        if max_w / max_h > original_ratio:
            adjusted_h = max_h
            adjusted_w = adjusted_h * original_ratio
        else:
            adjusted_w = max_w
            adjusted_h = adjusted_w / original_ratio

        return int(adjusted_w), int(adjusted_h), original_ratio


if __name__ == '__main__':
    pass
