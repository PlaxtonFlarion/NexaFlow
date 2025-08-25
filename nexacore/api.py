#      _          _
#     / \   _ __ (_)
#    / _ \ | '_ \| |
#   / ___ \| |_) | |
#  /_/   \_\ .__/|_|
#          |_|
#
# ==== Notes: License ====
# Copyright (c) 2024  Framix :: 画帧秀
# This file is licensed under the Framix :: 画帧秀 License. See the LICENSE.md file for more details.

import os
import re
import json
import uuid
import httpx
import numpy
import typing
import asyncio
import aiofiles
import tempfile
from pathlib import Path
from loguru import logger
from engine.channel import (
    Channel, Messenger
)
from engine.tinker import FileAssist
from nexacore import authorize
from nexaflow.classifier.base import (
    SingleClassifierResult, ClassifierResult
)
from nexaflow.cutter.cut_range import VideoCutRange
from nexaflow.video import VideoObject
from nexaflow import (
    const, toolbox
)


class Api(object):
    """
    Api

    通用异步接口适配器类，封装各类与远程服务交互的静态方法，包括数据获取、
    文件生成、命令配置加载等逻辑，适用于微服务通信、TTS 服务、自动化平台等场景。

    当前支持的服务包括语音格式元信息拉取、语音合成任务、业务用例命令获取等。
    所有接口方法均通过异步方式与后端 API 通信，支持 JSON 响应解析及异常处理。

    Notes
    -----
    - 提供统一的参数打包与请求流程，封装远程接口调用的细节。
    - 支持动态参数拼接、异常捕获、数据缓存与文件写入。
    - 可根据实际业务场景扩展其他静态方法，如上传日志、获取配置、拉取资源等。
    """

    background: list = []

    @staticmethod
    async def ask_request_get(
        url: str, key: typing.Optional[str] = None, *_, **kwargs
    ) -> dict:
        """
        通用异步 GET 请求方法。

        构造带参数的异步 GET 请求，自动附带默认参数并发送到指定 URL。支持从响应中提取指定字段，
        用于统一的业务数据获取流程，如模板信息、配置元数据等。

        Parameters
        ----------
        url : str
            请求的目标接口地址。

        key : str, optional
            可选的响应字段键名，若提供则返回对应字段的内容，否则返回整个响应字典。

        *_
            保留参数，未使用。

        **kwargs
            追加到请求参数中的动态键值对，用于拼接请求 query 参数。

        Returns
        -------
        dict
            远程服务返回的 JSON 数据（或提取后的字段值）。
        """
        params = Channel.make_params() | kwargs
        async with Messenger() as messenger:
            resp = await messenger.poke("GET", url, params=params)
            return resp.json()[key] if key else resp.json()

    @staticmethod
    async def formatting() -> typing.Optional[dict]:
        """
        获取远程 TTS 服务的可用状态及支持的音频格式列表。

        Returns
        -------
        dict or None
            {
                "enabled": bool,       # 服务可用状态
                "formats": list[str],  # 支持的音频格式列表
                ...                    # 其他元信息
            }
            若服务不可用或请求异常，则返回 None。
        """
        try:
            sign_data = await Api.ask_request_get(const.SPEECH_META_URL)
            auth_info = authorize.verify_signature(sign_data)
        except Exception as e:
            return logger.debug(e)

        return auth_info.get("mode", {})

    @staticmethod
    async def profession(case: str) -> dict:
        """
        根据指定用例名从业务接口获取命令列表。

        通过异步请求远程业务系统，加载指定 case 的命令配置数据。

        Parameters
        ----------
        case : str
            用例名称，用于作为参数查询业务命令配置。

        Returns
        -------
        dict
            返回包含命令配置的字典组成结构。
        """
        params = Channel.make_params() | {"case": case}
        async with Messenger() as messenger:
            resp = await messenger.poke("GET", const.BUSINESS_CASE_URL, params=params)
            return resp.json()

    @staticmethod
    async def fetch_template_file(url: str, template_name: str) -> str:
        """
        异步获取远程模板文件内容。

        本方法通过指定的 URL 与模板名称组合构建请求参数，
        并使用 Messenger 客户端发起 GET 请求，获取模板内容。

        Parameters
        ----------
        url : str
            远程请求的基础地址（如模板服务接口地址）。

        template_name : str
            模板文件名称，用于作为查询参数 "page" 的值。

        Returns
        -------
        str
            请求返回的模板文件内容字符串。
        """
        params = Channel.make_params() | {"page": template_name}
        async with Messenger() as messenger:
            resp = await messenger.poke("GET", url, params=params)
            return resp.text

    @staticmethod
    async def synthesize(speak: str, allowed_extra: list) -> tuple[str, bytes]:
        """
        合成文本语音并返回文件名及二进制内容。

        Parameters
        ----------
        speak : str
            待合成的文本内容。可带有扩展名（如 "你好.mp3"），如无扩展名，则自动添加默认扩展名。

        allowed_extra : list of str
            支持的音频文件扩展名列表，例如 ["mp3", "wav"]，用于判定输出格式。

        Returns
        -------
        tuple of (str, bytes)
            - 文件名（已去除非法字符和自动拼接扩展名），如 "你好.mp3"
            - 合成后音频文件的二进制内容
        """
        allowed_ext = {ext.lower().lstrip(".") for ext in allowed_extra}
        logger.debug(f"Allowed ext -> {allowed_ext}")

        stem, waver = os.path.splitext((speak or "").strip())
        waver = waver.lower().lstrip(".")
        if waver not in allowed_ext or not waver:
            waver = (next(iter(allowed_ext)) if allowed_ext else const.WAVER_FMT)

        # 文件名去除非法字符
        final_speak = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "", stem)

        # 构建 payload
        payload = {"speak": final_speak, "waver": waver} | Channel.make_params()
        logger.debug(f"Remote synthesize: {payload}")

        try:
            async with Messenger() as messenger:
                resp = await messenger.poke("POST", const.SPEECH_VOICE_URL, json=payload)
                logger.debug(f"Download url: {(download_url := resp.json()['url'])}")

                resp = await messenger.poke("GET", download_url)

        except Exception as e:
            logger.debug(e)

        return final_speak + "." + waver, resp.content

    @staticmethod
    async def remote_config() -> typing.Optional[dict]:
        """
        获取远程配置中心的全局配置数据。
        """
        try:
            sign_data = await Api.ask_request_get(const.GLOBAL_CF_URL)
            auth_info = authorize.verify_signature(sign_data)
        except Exception as e:
            return logger.debug(e)

        return auth_info.get("configuration", {})

    @staticmethod
    async def online_template_meta() -> typing.Optional[dict]:
        """
        获取模版元信息。
        """
        try:
            sign_data = await Api.ask_request_get(const.TEMPLATE_META_URL)
            auth_info = authorize.verify_signature(sign_data)
        except Exception as e:
            return logger.debug(e)

        return auth_info.get("template", {})

    @staticmethod
    async def online_toolkit_meta(platform: str) -> typing.Optional[dict]:
        """
        获取工具元信息。
        """
        try:
            sign_data = await Api.ask_request_get(const.TOOLKIT_META_URL, platform=platform)
            auth_info = authorize.verify_signature(sign_data)
        except Exception as e:
            return logger.debug(e)

        return auth_info.get("toolkit", {})

    @staticmethod
    async def online_model_meta() -> typing.Optional[dict]:
        """
        获取模型元信息。
        """
        try:
            sign_data = await Api.ask_request_get(const.MODEL_META_URL)
            auth_info = authorize.verify_signature(sign_data)
        except Exception as e:
            return logger.debug(e)

        return auth_info.get("models", {})

    @staticmethod
    async def proxy_predict() -> typing.Optional[dict]:
        """
        获取远程推理服务地址。
        """
        try:
            sign_data = await Api.ask_request_get(const.PREDICT_URL)
            auth_info = authorize.verify_signature(sign_data)
        except Exception as e:
            return logger.debug(e)

        return auth_info

    @staticmethod
    async def online_predict(
            online: dict,
            video: "VideoObject",
            valid_range: list["VideoCutRange"],
            step: typing.Optional[int] = None,
            keep_data: typing.Optional[bool] = None,
            boost_mode: typing.Optional[bool] = None,
    ) -> typing.Optional["ClassifierResult"]:
        """
        调用远程推理服务，对视频帧进行分类并返回结果。

        该方法会将视频帧数据打包为 `.npz` 格式，通过 HTTP 请求上传至远程推理服务。
        同时发送包含视频元信息与裁剪区间的数据，并以流式方式逐帧接收推理结果。

        Parameters
        ----------
        online : dict
            远程推理服务参数，包括接口地址、请求方法、鉴权 Token 等信息。
            示例字段包括 `"method"`、`"url"`、`"auth_header"`、`"token"`、`"timeout"`。

        video : VideoObject
            视频对象，包含全部帧数据与元信息。

        valid_range : list of VideoCutRange
            有效帧区间列表，表示需要推理的帧段落及其相关图像指标（SSIM、PSNR、MSE 等）。

        step : int, optional
            帧处理步长，用于控制推理帧频（默认 None）。

        keep_data : bool, optional
            是否保留原始图像帧数据（默认 False），远程服务一般无需图像返回，可设置为 False。

        boost_mode : bool, optional
            是否启用增强模式，在推理时激活更高阶策略（例如多模型融合或优化路径）。

        Returns
        -------
        ClassifierResult or None
            推理完成的结构化结果，包含每帧的分类结果与时间戳；
            若请求失败或发生异常，返回 None 并打印日志。

        Raises
        ------
        RuntimeError
            - 若远程服务响应非 200；
            - 若响应中包含错误标记（"ERROR:"、"FATAL:"）；
            - 若上传或流处理过程中出错。

        Notes
        -----
        - 使用临时 `.npz` 文件压缩帧数据后上传，上传完成后自动清理文件；
        - 响应结果按流式处理逐帧推理结果，避免等待完整处理；
        - 异常处理采用日志记录并关闭进度条；
        - 推理结果为 `ClassifierResult` 类型的结构体，适用于后续展示或报告生成。
        """
        method = online["method"]
        url = online["url"]
        headers = {online["auth_header"]: online["token"]}
        timeout = online.get("timeout", 60.0)

        final_result: list[dict] = []

        frame_data_list = [
            {"frame_id": frame.frame_id, "data": frame.data} for frame in video.frames_data
        ]

        metadata = {
            "video_name": (video_name := f"{video.name}_{uuid.uuid4().hex[:8]}"),
            "video_path": (video_path := os.path.basename(video.path)),
            "frame_count": video.frame_count,
            "frame_shape": video.frame_detail()[-1],
            "frames_data": [
                {
                    "frame_id": frame.frame_id,
                    "timestamp": frame.timestamp
                } for frame in video.frames_data
            ],
            "valid_range": [
                {
                    "start": cut_range.start,
                    "end": cut_range.end,
                    "ssim": cut_range.ssim,
                    "psnr": cut_range.psnr,
                    "mse": cut_range.mse,
                    "start_time": cut_range.start_time,
                    "end_time": cut_range.end_time,
                } for index, cut_range in enumerate(valid_range, start=1)
            ],
            "step": step,
            "keep_data": keep_data,
            "boost_mode": boost_mode
        }

        pbar = toolbox.show_progress(total=video.frame_count, color=120)

        try:
            with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp_file:
                npz_path = tmp_file.name
                npz_dict = {
                    str(frame["frame_id"]): frame["data"] for frame in frame_data_list
                }
                numpy.savez_compressed(npz_path, **npz_dict)

                async with aiofiles.open(npz_path, "rb") as npz_file:
                    content = await npz_file.read()

                data = {
                    "frame_meta": json.dumps(metadata, ensure_ascii=False)
                }
                files = {
                    "frame_file": (f"{video_name}.npz", content, "application/octet-stream")
                }

            classifier_mark, error_mark, fatal_mark = "SingleClassifierResult: ", "ERROR: ", "FATAL: "
            async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
                async with client.stream(method, url, data=data, files=files) as resp:
                    if resp.status_code != 200:
                        # 读取完整响应并抛出错误
                        ample = await resp.aread()
                        raise RuntimeError(f"❌ {resp.status_code} -> {ample.decode(const.CHARSET)}")

                    # 逐行处理流式响应
                    async for line in resp.aiter_lines():
                        if line.startswith(classifier_mark):
                            json_str = line.removeprefix(classifier_mark).strip()
                            final_result.append(json.loads(json_str))
                            pbar.update(1)
                        elif line.startswith(error_mark):
                            json_str = line.removeprefix(error_mark).strip()
                            raise RuntimeError(json.loads(json_str).get("error"))
                        elif line.startswith(fatal_mark):
                            json_str = line.removeprefix(fatal_mark).strip()
                            raise RuntimeError(json.loads(json_str).get("fatal"))

            pbar.close()
            return ClassifierResult([SingleClassifierResult(
                r.get("video_path", video_path), r.get("frame_id", frame.frame_id),
                r.get("timestamp", frame.timestamp), r["result"], frame.data
            ) for r, frame in zip(final_result, video.frames_data)])

        except Exception as e:
            pbar.close()
            return logger.error(e)

        finally:
            Api.background.append(
                asyncio.create_task(asyncio.to_thread(os.remove, npz_path))
            )

    @staticmethod
    async def fetch_range(
            sem: "asyncio.Semaphore",
            client: "httpx.AsyncClient",
            progress: dict,
            filename: str,
            url: str,
            *args,
            **kwargs
    ) -> typing.Optional[tuple[int, bytes]]:
        """
        执行单个下载分块的请求，支持失败重试与进度更新。

        使用 HTTP Range 请求下载文件的指定区间段，支持并发控制与最大重试机制，
        并在成功后更新指定文件的下载进度。

        Parameters
        ----------
        sem : asyncio.Semaphore
            用于限制并发请求数量的信号量。

        client : httpx.AsyncClient
            已初始化的 HTTP 异步客户端，用于执行请求。

        progress : dict
            下载进度字典，结构如 {filename: {"range": int, "total": int}}，
            会在下载成功后自动加 1。

        filename : str
            当前任务所属文件名，用于在 `progress` 中标识。

        url : str
            下载地址，需支持 Range 请求。

        *args
            预留参数，当前未使用。

        **kwargs
            包含以下关键字参数：
            - index : int
                当前分块索引，用于还原写入顺序。
            - start : int
                当前块的起始字节。
            - close : int
                当前块的结束字节。
            - max_retries : int, optional
                最大重试次数，默认为 3。

        Returns
        -------
        tuple[int, bytes]
            若下载成功，返回 `(index, content)`，用于写入对应位置。

        Raises
        ------
        RuntimeError
            若在最大重试次数内仍无法成功获取分块，将抛出异常。

        Notes
        -----
        - 仅支持返回 206 状态码的分块响应。
        - 每次失败后会进行指数退避等待再重试。
        """
        _ = args

        index = kwargs.get("index")
        start = kwargs.get("start")
        close = kwargs.get("close")
        max_retries = kwargs.get("max_retries", 3)

        headers = {"Range": f"bytes={start}-{close}"}

        for attempt in range(1, max_retries + 1):
            try:
                async with sem:
                    resp = await client.request("GET", url, headers=headers, timeout=60.0)
                    if resp.status_code == 206:
                        progress[filename]["range"] += 1
                        return index, resp.content
                    raise RuntimeError(f"Status {resp.status_code}")
            except Exception as e:
                if attempt == max_retries:
                    raise RuntimeError(f"Chunk {index} failed: {e}")
                await asyncio.sleep(1.5 * attempt)

    @staticmethod
    async def join_range_download(
            progress: dict,
            status: dict,
            filename: str,
            url: str,
            output: str,
            total_size: int,
            expected_sha256: str
    ) -> None:
        """
        执行并发分块下载任务，并校验哈希与自动解压。

        此方法用于从指定 URL 下载大文件，支持 Range 请求的并发下载，
        实时更新下载进度，完成后进行 SHA256 校验与 ZIP 解压操作。

        Parameters
        ----------
        progress : dict
            用于记录下载进度的共享状态字典，结构如 {filename: {"range": int, "total": int}}。

        status : dict
            状态信息记录字典，例如 {"status": "Success"} 或 {"status": "Error"}。

        filename : str
            当前下载的文件名，用于标记进度状态。

        url : str
            下载链接地址，必须支持 HTTP Range 请求。

        output : str
            下载文件的临时保存路径（最终会被删除）。

        total_size : int
            文件总大小（字节数）。

        expected_sha256 : str
            下载完成后用于校验的 SHA256 哈希值。

        Raises
        ------
        ValueError
            若下载完成后的哈希校验失败，则抛出此异常。

        Exception
            下载过程或解压过程中出现任何异常均会被捕获并写入 `status` 字典。

        Notes
        -----
        - 支持自动限流（Semaphore）控制并发数。
        - 下载失败时会清理本地临时文件。
        - 使用 `httpx.AsyncClient` 和 `aiofiles` 实现异步 IO。
        """
        chunk_size = 10 * 1024 * 1024
        ranges: list[tuple[int, int]] = [
            (i, min(i + chunk_size - 1, total_size - 1)) for i in range(0, total_size, chunk_size)
        ]

        chunks: list[typing.Union[None, bytes]] = [None] * len(ranges)
        sem: "asyncio.Semaphore" = asyncio.Semaphore(4)

        progress[filename]["total"] = len(ranges)

        try:
            async with httpx.AsyncClient() as client:
                results = await asyncio.gather(
                    *(Api.fetch_range(
                        sem, client, progress, filename, url, index=index, start=start, close=close
                    ) for index, (start, close) in enumerate(ranges))
                )

            for idx, chunk in results:
                chunks[idx] = chunk

            async with aiofiles.open(output, "wb") as f:
                for chunk in chunks:
                    await f.write(chunk)

            actual_hash = await FileAssist.calculate_sha256(output)
            logger.debug(f"校验哈希: {actual_hash}")
            if actual_hash != expected_sha256:
                raise ValueError(f"Hash 校验失败: {actual_hash}")

            logger.debug(f"解压文件: {output}")
            extract_file = await FileAssist.extract_zip(output)
            logger.debug(f"下载成功: {extract_file}")
            status["status"] = "Success"
            progress[filename]["range"] = len(ranges)

        except Exception as e:
            status["status"] = "Error"
            progress[filename]["range"] = "❌ 下载失败"
            raise e

        finally:
            if Path(output).exists():
                await asyncio.to_thread(os.remove, output)


if __name__ == '__main__':
    pass
