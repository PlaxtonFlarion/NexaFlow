#
#   ____          _ _       _
#  / ___|_      _(_) |_ ___| |__
#  \___ \ \ /\ / / | __/ __| '_ \
#   ___) \ V  V /| | || (__| | | |
#  |____/ \_/\_/ |_|\__\___|_| |_|
#

"""
版权所有 (c) 2024  Framix(画帧秀)
此文件受 Framix(画帧秀) 许可证的保护。您可以在 LICENSE.md 文件中查看详细的许可条款。

Copyright (c) 2024  Framix(画帧秀)
This file is licensed under the Framix(画帧秀) License. See the LICENSE.md file for more details.

# Copyright (c) 2024  Framix(画帧秀)
# このファイルは Framix(画帧秀) ライセンスの下でライセンスされています。詳細は LICENSE.md ファイルを参照してください。
"""

import os
import json
import typing
from engine.terminal import Terminal


class Switch(object):
    """
    Switch 工具类，用于执行音频与视频的格式转换任务。

    Notes
    -----
    - 所有方法为异步静态方法，不依赖类状态。
    - 实际执行依赖 `ffmpeg` 命令和外部 `Terminal.cmd_line` 工具。
    """

    @staticmethod
    async def ask_audio_reform(ffmpeg: str, src: str, dst: str) -> typing.Any:
        """
        异步执行音频重编码操作。

        使用 ffmpeg 将输入音频重新编码为 44100Hz 采样率、128kbps 比特率的标准格式。

        Parameters
        ----------
        ffmpeg : str
            ffmpeg 执行路径。
        src : str
            输入音频文件路径。
        dst : str
            输出音频文件路径。

        Returns
        -------
        typing.Any
            终端命令执行结果，通常为标准输出或错误信息。

        Notes
        -----
        - 输出音频格式与原始格式相同，但采样率和比特率被标准化。
        - 若源文件不可访问或 ffmpeg 参数错误，可能返回 None。

        Workflow
        --------
        1. 构造 ffmpeg 命令：以 `-ar 44100` 和 `-b:a 128k` 参数进行音频重编码。
        2. 异步执行命令并返回结果。
        """
        cmd = [ffmpeg, "-i", src, "-ar", "44100", "-b:a", "128k", dst]
        return await Terminal.cmd_line(cmd)

    @staticmethod
    async def ask_video_reform(ffmpeg: str, fps: int, src: str, dst: str) -> typing.Any:
        """
        异步执行视频帧率重编码操作。

        使用 ffmpeg 修改视频帧率为指定值，生成新的视频文件。

        Parameters
        ----------
        ffmpeg : str
            ffmpeg 执行路径。

        fps : int
            目标帧率（帧/秒）。

        src : str
            输入视频文件路径。

        dst : str
            输出视频文件路径。

        Returns
        -------
        typing.Any
            终端命令执行结果，通常为标准输出或错误信息。

        Notes
        -----
        - 输出视频格式与原始格式相同，但帧率被强制为指定值。
        - 若源文件损坏或参数非法，返回结果可能为 None。

        Workflow
        --------
        1. 构造 ffmpeg 命令，添加 `-r` 参数设定新帧率。
        2. 异步执行命令并返回处理结果。
        """
        cmd = [ffmpeg, "-i", src, "-r", f"{fps}", dst]
        return await Terminal.cmd_line(cmd)

    @staticmethod
    async def ask_video_change(ffmpeg: str, video_filter: list, src: str, dst: str, **kwargs) -> typing.Any:
        """
        异步执行视频画面调整任务，支持裁剪时段并应用指定的视频滤镜。

        该方法通过调用 ffmpeg 实现视频的局部片段处理，允许设定起始时间、结束时间或持续时长，并根据提供的视频过滤器进行图像增强或变换操作。

        Parameters
        ----------
        ffmpeg : str
            ffmpeg 执行路径。

        video_filter : list
            视频滤镜命令列表，最终将通过 -vf 参数传入 ffmpeg。

        src : str
            输入视频路径。

        dst : str
            输出视频路径。

        **kwargs : dict
            可选关键字参数：
            - start (str): 视频起始时间（如 "00:00:10"），对应 -ss 参数。
            - close (str): 视频结束时间（如 "00:00:20"），对应 -to 参数。
            - limit (str): 视频处理持续时长（如 "00:00:05"），对应 -t 参数。

        Returns
        -------
        typing.Any
            异步执行结果，通常为终端标准输出字符串，若失败可能返回 None。

        Raises
        ------
        无直接抛出异常，但内部依赖 `Terminal.cmd_line`，其异常需外部捕获处理。

        Notes
        -----
        - 如果同时提供 close 和 limit，仅使用 close。
        - 视频编码使用 H.264（libx264），并设定固定压缩率（CRF 18），音频流保持不变。
        - 使用该函数可实现带滤镜的视频片段剪辑与画质调整，适用于视频预处理和提取任务。

        Workflow
        --------
        1. 初始化命令列表，追加时间裁剪参数（如 -ss、-to 或 -t）。
        2. 指定输入视频并拼接 -vf 滤镜命令。
        3. 使用 libx264 编码输出视频，同时保留原始音频（-c:a copy）。
        4. 异步执行构造的 ffmpeg 命令，返回输出结果。
        """
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
    async def ask_video_detach(ffmpeg: str, video_filter: list, src: str, dst: str, **kwargs) -> typing.Any:
        """
        异步拆解视频为图像帧，并按命名规则保存到指定路径。

        该方法通过 ffmpeg 提取视频帧，支持设置处理时段和应用过滤器，适用于视频帧分析、图像训练集构建等任务。

        Parameters
        ----------
        ffmpeg : str
            ffmpeg 执行路径。

        video_filter : list
            视频滤镜列表（如裁剪、缩放、灰度等），用于通过 -vf 传入 ffmpeg。

        src : str
            输入视频路径。

        dst : str
            输出帧图像保存目录。

        **kwargs : dict
            可选关键字参数：
            - start (str): 处理起始时间（格式如 "00:00:05"），对应 ffmpeg -ss。
            - close (str): 处理结束时间（格式如 "00:00:15"），对应 ffmpeg -to。
            - limit (str): 处理时长限制（格式如 "00:00:10"），对应 ffmpeg -t。
            注意：如果同时提供 close 和 limit，仅使用 close。

        Returns
        -------
        typing.Any
            异步执行结果，通常为终端输出字符串，若失败可能返回 None。

        Raises
        ------
        无直接异常抛出，但内部依赖 Terminal.cmd_line，可能因命令失败返回 None 或执行异常。

        Notes
        -----
        - 输出图像命名格式为 frame_00001.png、frame_00002.png 等，按序保存至 dst 目录。
        - 可配合 scale、crop、gray 等过滤器生成模型训练图像或帧级分析数据。
        - 视频帧率决定输出图片数量，高帧率视频可能生成大量图片。

        Workflow
        --------
        1. 构建 ffmpeg 命令，添加时间裁剪参数（-ss、-to 或 -t）。
        2. 加载输入视频，追加 -vf 滤镜命令。
        3. 设置输出帧图片保存格式为 PNG，命名为 frame_%05d.png。
        4. 异步执行命令，完成帧拆解任务。
        """
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
    async def ask_video_tailor(ffmpeg: str, src: str, dst: str, **kwargs) -> typing.Any:
        """
        异步裁剪视频指定时间段并输出为新视频文件。

        该方法通过 ffmpeg 对输入视频进行无损剪辑，支持设置起始、结束或时长限制，不重新编码，适合快速提取片段或对齐时间范围。

        Parameters
        ----------
        ffmpeg : str
            ffmpeg 可执行路径。

        src : str
            输入视频路径。

        dst : str
            裁剪后的视频输出路径。

        **kwargs : dict
            可选关键字参数：
            - start (str): 起始时间点，如 "00:00:10"，对应 ffmpeg 的 -ss。
            - close (str): 结束时间点，如 "00:00:20"，对应 ffmpeg 的 -to。
            - limit (str): 最大处理时长，如 "00:00:10"，对应 ffmpeg 的 -t。
            注意：若同时提供 close 与 limit，仅优先使用 close。

        Returns
        -------
        typing.Any
            异步执行结果，通常为终端输出字符串，若命令失败可能返回 None。

        Raises
        ------
        无显式异常抛出，但内部依赖 Terminal.cmd_line，命令执行失败时可能返回 None 或抛出异常。

        Notes
        -----
        - 本方法使用 `-c copy` 进行流复制，不进行转码，处理速度更快，文件质量保持不变。
        - start、close、limit 的时间格式需为 HH:MM:SS 或 S 秒数格式。
        - 可用于提取特定时间段的视频内容，用作后续分析或训练数据构建。

        Workflow
        --------
        1. 构造基础 ffmpeg 命令，设置起始（-ss）和结束（-to）或时长限制（-t）。
        2. 加载输入视频文件（-i）。
        3. 使用 `-c copy` 保留原始编码，设置输出路径。
        4. 异步执行命令，完成视频裁剪操作。
        """
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
    async def ask_video_stream(ffprobe: str, src: str) -> dict:
        """
        异步获取视频的详细元信息，包括编码信息、尺寸、帧率、时长和关键帧数据。

        该方法调用 `ffprobe` 工具对视频进行分析，提取视频主流的结构化信息，并返回标准化的数据字典。

        Parameters
        ----------
        ffprobe : str
            ffprobe 执行路径，通常为 "ffprobe" 或绝对路径。

        src : str
            视频源文件路径，支持任意格式的本地视频文件。

        Returns
        -------
        dict
            包含以下字段的视频信息字典（若失败则返回空字典）：
            - key_frames : list[dict]
                所有帧的索引、时间戳、帧类型（I/B/P）。
            - codec_name : str
                视频使用的编码器名称（如 h264）。
            - codec_type : str
                媒体类型，通常为 "video"。
            - original : tuple[int, int]
                视频原始尺寸 (width, height)。
            - rlt_frame_rate : str
                实际帧率，格式如 "30/1"。
            - avg_frame_rate : str
                平均帧率，格式如 "29/1"。
            - nb_read_frames : int
                视频总帧数。
            - duration : float
                视频时长（单位：秒）。
            - size : float
                视频文件大小（单位：MB）。
            - bit_rate : float
                视频比特率（单位：Mbps）。

        Raises
        ------
        无显式异常抛出：
            - 若 ffprobe 结果为空或结构异常，捕获错误后返回空字典。

        Notes
        -----
        - 本方法仅分析视频主流（v:0），忽略音频或其他数据流。
        - 使用 `-count_frames` 启用帧级统计。
        - `key_frames` 实际包含所有帧，但附带每帧是否为关键帧的标记。
        - 尽管字段为 key_frames，但包含所有帧的类型与时间。
        - 若 `result` 无法解析为 JSON 格式或缺失字段，将返回空字典。

        Workflow
        --------
        1. 构造 `ffprobe` 命令，用于提取编码、尺寸、帧率、格式和帧级数据。
        2. 异步执行命令，获取 JSON 格式结果。
        3. 解析 JSON，提取 `streams`, `format`, `frames` 三个部分。
        4. 构造标准化字典 `video_streams` 并返回。
        5. 若解析失败，则返回空字典。
        """
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
        """
        根据视频时长计算合法的起始、结束和限制时间点。

        该方法用于根据提供的起始时间（start）、结束时间（close）或限制时间（limit）来计算视频的有效处理时间段。
        它会自动修正非法时间范围，并确保时间段满足最小处理要求。

        Parameters
        ----------
        start : Optional[int | float]
            视频处理起始时间点（单位：秒），若为 None 则从头开始。

        close : Optional[int | float]
            视频处理结束时间点（单位：秒），若为 None 表示不指定结束时间。

        limit : Optional[int | float]
            视频最大处理时长限制（单位：秒），优先级低于 close。

        duration : Optional[int | float]
            视频总时长（单位：秒），必须为正数，否则所有处理将被忽略。

        Returns
        -------
        tuple[Optional[int | float], Optional[int | float], Optional[int | float]]
            返回处理时间点的三元组 (start_point, close_point, limit_point)，每个字段可能为 None，表示该项未被指定或无效：
            - start_point: 修正后的起始时间点。
            - close_point: 修正后的结束时间点。
            - limit_point: 修正后的限制时长。

        Raises
        ------
        无显式异常抛出，内部自动处理所有边界值和非法输入。

        Notes
        -----
        - 若提供的 `duration` 为 None 或小于等于 0，则直接返回 (None, None, None)。
        - 若 `close` 时间存在但小于起始时间超过 0.09 秒，返回无效。
        - 若设置了 `limit`，则从起始时间开始向后推进 `limit` 秒；若超出总时长则自动修剪。
        - 若最终计算出的时间段不足 0.09 秒，也将返回 (None, None, None)。

        Workflow
        --------
        1. 若 duration 无效，直接返回三项为 None。
        2. 校验并设置 start_point，若超出范围则设为 0。
        3. 若指定 close，设置 close_point 并校验与起始点的差值。
        4. 若未指定 close 但指定了 limit，则基于 start_point 推算 limit_point。
        5. 若 close 或 limit 范围不足 0.09 秒，视为无效时间段。
        6. 返回最终计算出的三元组。
        """
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
        """
        根据目标尺寸自动调整为符合原始宽高比的新尺寸。

        该方法用于在保持原始视频宽高比不变的前提下，根据入口尺寸限制，自动计算一个最佳适配尺寸，避免拉伸变形。通常用于图像缩放或裁剪前的尺寸预设。

        Parameters
        ----------
        original_frame_size : tuple
            原始帧的尺寸，格式为 (width, height)，表示原始图像或视频帧的宽度和高度。

        entrance_frame_size : tuple
            入口目标尺寸，格式为 (width, height)，用于控制输出图像最大宽高限制。

        Returns
        -------
        tuple[int, int, float]
            返回包含三个值的元组：
            - width : int
                调整后图像的宽度。
            - height : int
                调整后图像的高度。
            - ratio : float
                原始图像的宽高比。

        Raises
        ------
        无显式异常抛出。所有计算均自动处理非法输入，避免报错。

        Notes
        -----
        - 如果目标尺寸与原始尺寸相同，则直接返回原始宽高及比例。
        - 若目标尺寸小于原始尺寸，则限制输出图像最大尺寸不超过原始尺寸，但保留宽高比。
        - 自动限制目标尺寸最小不低于原始尺寸的 10%，避免图像过度压缩。
        - 宽高比使用 `原始宽 / 原始高` 计算，适用于横屏或竖屏图像。

        Workflow
        --------
        1. 解析原始宽度、高度和宽高比。
        2. 若目标尺寸与原始尺寸相同，直接返回。
        3. 计算目标宽度和高度在原始尺寸下的最大限制（不低于 10%）。
        4. 根据原始宽高比调整目标尺寸：
            - 若目标宽高比较宽，则以高度为基础调整宽度。
            - 若目标宽高比较窄，则以宽度为基础调整高度。
        5. 返回调整后的整数尺寸和原始宽高比。
        """
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
