#
#   ____          _ _       _
#  / ___|_      _(_) |_ ___| |__
#  \___ \ \ /\ / / | __/ __| '_ \
#   ___) \ V  V /| | || (__| | | |
#  |____/ \_/\_/ |_|\__\___|_| |_|
#

import os
import json
import typing
from engine.terminal import Terminal


class Switch(object):

    @staticmethod
    async def ask_audio_reform(ffmpeg, src: str, dst: str) -> str:
        cmd = [ffmpeg, "-i", src, "-ar", "44100", "-b:a", "128k", dst]
        return await Terminal.cmd_line(cmd)

    @staticmethod
    async def ask_video_reform(ffmpeg, fps: int, src: str, dst: str) -> str:
        cmd = [ffmpeg, "-i", src, "-r", f"{fps}", dst]
        return await Terminal.cmd_line(cmd)

    @staticmethod
    async def ask_video_change(ffmpeg, video_filter: list, src: str, dst: str, **kwargs) -> str:
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
        cmd += ["-vf", ",".join(video_filter), "-c:v", "libx264", "-crf", "18", "-c:a", "copy", dst]

        return await Terminal.cmd_line(cmd)

    @staticmethod
    async def ask_video_detach(ffmpeg, video_filter: list, src: str, dst: str, **kwargs) -> str:
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
        cmd += ["-vf", ",".join(video_filter), os.path.join(dst, "frame_%05d.png")]

        return await Terminal.cmd_line(cmd)

    @staticmethod
    async def ask_video_tailor(ffmpeg, src: str, dst: str, **kwargs) -> str:
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

        return await Terminal.cmd_line(cmd)

    @staticmethod
    async def ask_video_stream(ffprobe, src: str) -> dict:
        cmd = [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,codec_type,width,height,r_frame_rate,avg_frame_rate,nb_read_frames",
            "-show_entries", "format=duration,size,bit_rate",
            "-show_entries", "frame=key_frame,pts_time,pict_type",
            "-of", "json", "-count_frames", src
        ]
        result = await Terminal.cmd_line(cmd)
        try:
            json_file = json.loads(result)
            stream_dict = json_file["streams"][0]
            format_dict = json_file["format"]
            frames_list = json_file["frames"]
            video_streams = {
                "key_frames": [
                    {"type": frame["pict_type"], "idx": f"{idx + 1}", "time": frame["pts_time"]}
                    for idx, frame in enumerate(frames_list)
                ],
                "codec_name": stream_dict["codec_name"],
                "codec_type": stream_dict["codec_type"],
                "original": (int(stream_dict["width"]), int(stream_dict["height"])),
                "rlt_frame_rate": stream_dict["r_frame_rate"],
                "avg_frame_rate": stream_dict["avg_frame_rate"],
                "nb_read_frames": int(stream_dict["nb_read_frames"]),
                "duration": float(format_dict["duration"]),
                "size": round(int(format_dict["size"]) / 1048576, 2),
                "bit_rate": int(format_dict["bit_rate"]) / 1000000,
            }
        except (AttributeError, KeyError, ValueError, json.JSONDecodeError):
            return {}
        return video_streams

    @staticmethod
    async def ask_magic_point(
            start: typing.Optional[int | float],
            close: typing.Optional[int | float],
            limit: typing.Optional[int | float],
            duration: typing.Optional[int | float]
    ) -> tuple[typing.Optional[int | float], typing.Optional[int | float], typing.Optional[int | float]]:

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
