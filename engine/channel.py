#    ____ _                            _
#   / ___| |__   __ _ _ __  _ __   ___| |
#  | |   | '_ \ / _` | '_ \| '_ \ / _ \ |
#  | |___| | | | (_| | | | | | | |  __/ |
#   \____|_| |_|\__,_|_| |_|_| |_|\___|_|
#
# ==== Notes: License ====
# Copyright (c) 2024  Framix :: 画帧秀
# This file is licensed under the Framix :: 画帧秀 License. See the LICENSE.md file for more details.

import time
import httpx
import typing
import secrets
from engine.tinker import FramixError
from nexaflow import const


class Channel(object):
    """
    参数通道生成器，用于构建基础通信参数集。
    """

    @staticmethod
    def make_params() -> dict[str, typing.Any]:
        """
        构造通信参数集合。

        Returns
        -------
        dict[str, typing.Any]
            包含默认参数的字典，字段包括：
            - `a`: 应用描述常量
            - `t`: 当前时间戳（秒）
            - `n`: 随机生成的16位十六进制字符串
        """
        return {
            "a": const.DESC, "t": int(time.time()), "n": secrets.token_hex(8)
        }


class Messenger(object):
    """
    异步通信信使，封装 HTTP 请求发送与连接管理。

    Attributes
    ----------
    client : httpx.AsyncClient | None
        异步 HTTP 客户端，生命周期受上下文管理控制。
    """

    def __init__(self):
        """
        初始化 Messenger 实例，默认未建立客户端连接。
        """
        self.client: typing.Optional["httpx.AsyncClient"] = None

    async def __aenter__(self):
        """
        异步上下文进入函数，初始化 HTTP 客户端。

        Returns
        -------
        Messenger
            当前实例自身，可用于发送异步请求。
        """
        self.client = httpx.AsyncClient(headers=const.BASIC_HEADERS, timeout=10)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """
        异步上下文退出函数，自动关闭客户端连接。
        """
        if self.client:
            await self.client.aclose()

    async def poke(self, method: str, url: typing.Union["httpx.URL", str], *args, **kwargs) -> "httpx.Response":
        """
        发送异步 HTTP 请求，并统一封装异常处理。

        Parameters
        ----------
        method : str
            请求方法，如 "GET"、"POST" 等。

        url : URL | str
            请求地址，可为字符串或 URL 对象。

        *args : Any
            位置参数，传递给 httpx。

        **kwargs : Any
            关键字参数，传递给 httpx。

        Returns
        -------
        httpx.Response
            请求响应对象，需调用者手动解析。

        Raises
        ------
        FramixError
            请求失败或连接异常时抛出。
        """
        assert self.client, f"Client instance is missing. Did you forget to initialize it?"

        try:
            response = await self.client.request(method, url, *args, **kwargs)
            response.raise_for_status()
            return response

        except httpx.HTTPStatusError as e:
            raise FramixError(f"❌ {e.response.status_code} -> {e.response.text}")
        except httpx.HTTPError as e:
            raise FramixError(f"❌ {e}")


if __name__ == '__main__':
    pass
