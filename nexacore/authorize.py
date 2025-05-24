#      _         _   _                _
#     / \  _   _| |_| |__   ___  _ __(_)_______
#    / _ \| | | | __| '_ \ / _ \| '__| |_  / _ \
#   / ___ \ |_| | |_| | | | (_) | |  | |/ /  __/
#  /_/   \_\__,_|\__|_| |_|\___/|_|  |_/___\___|
#
# ==== Notes: License ====
# Copyright (c) 2024  Framix :: 画帧秀
# This file is licensed under the Framix :: 画帧秀 License. See the LICENSE.md file for more details.

import time
import json
import uuid
import httpx
import base64
import socket
import struct
import typing
import secrets
import hashlib
import platform
from pathlib import Path
from datetime import (
    datetime, timezone
)
from cryptography.hazmat.primitives import (
    hashes, serialization
)
from cryptography.hazmat.primitives.asymmetric import padding
from engine.terminal import Terminal
from engine.tinker import FramixError
from nexacore.design import Design
from nexaflow import const


def fingerprint() -> str:
    """
    生成当前设备的唯一指纹（用于授权绑定），采用多项系统属性拼接后哈希。
    """
    parts = [
        platform.system(),
        platform.machine(),
        str(uuid.getnode()),
        platform.processor() or "-",
    ]
    raw = "::".join(parts).encode(const.CHARSET)
    return hashlib.sha256(raw).hexdigest()


def network_time() -> typing.Optional["datetime"]:
    """
    尝试通过多个 NTP 授时服务器获取当前的 UTC 网络时间。
    """
    ntp_servers = [
        "ntp.aliyun.com",  # 阿里云授时服务器
        "cn.ntp.org.cn",  # NTP授时快速域名服务
        "cn.pool.ntp.org",  # 开源NTP服务器
        "pool.ntp.org",  # 开源NTP服务器
        "time.windows.com",  # Windows授时服务器
        "time.nist.gov",  # Windows授时服务器
        "time.apple.com",  # macOS授时服务器
        "time.asia.apple.com",  # macOS授时服务器
        "time1.cloud.tencent.com",  # 腾讯云授时服务器
        "edu.ntp.org.cn",  # 中国 NTP 快速授时服务
        "time.google.com",  # Google授时服务器
        "ntp.nict.jp"  # 日本信息通信研究机构授时服务器
    ]
    port = 123
    buffer_size = 48
    ntp_epoch = 2208988800  # NTP 时间起点（1900-01-01）

    # 构造请求数据包
    data = b'\x1b' + 47 * b'\0'

    for server in ntp_servers:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(5)
                s.sendto(data, (server, port))
                recv_data, _ = s.recvfrom(buffer_size)
                if recv_data:
                    t = struct.unpack('!12I', recv_data)[10]
                    t -= ntp_epoch
                    return datetime.fromtimestamp(t, timezone.utc)
        except Exception as e:
            Design.Doc.wrn(e)
            continue

    return None


def verify_signature(lic_input: typing.Union["Path", dict]) -> dict:
    """
    验证授权文件的合法性与有效性。
    """
    try:
        # 加载公钥
        pubkey = serialization.load_pem_public_key(const.PUBLIC_KEY)

        lic = json.loads(
            lic_input.read_text()
        ) if isinstance(lic_input, Path) else lic_input

        data = base64.b64decode(lic["data"])
        signature = base64.b64decode(lic["signature"])

        # 验签
        pubkey.verify(signature, data, padding.PKCS1v15(), hashes.SHA256())
        # 解析授权信息
        auth_info = json.loads(data)

    except Exception as e:
        raise FramixError(f"❌ 通行证无效 -> {e}")

    return auth_info


async def verify_license(lic_file: "Path") -> typing.Any:
    """
    验证本地授权文件是否合法、未过期，并视情况进行更新续签。
    """
    Design.Doc.log(
        f"[bold #FFAF5F]Initiating license checkpoint ..."
    )

    if not lic_file.exists():
        raise FramixError(f"❌ 需要申请通行证 ...")

    auth_info = verify_signature(lic_file)

    expire = datetime.strptime(
        (exp := auth_info["expire"]), "%Y-%m-%d"
    ).replace(tzinfo=timezone.utc)

    code, license_id = auth_info["code"], auth_info["license_id"]

    if not (now_time := network_time()):
        raise FramixError(f"❌ 无法连接服务器 ...")
    if now_time > expire:
        raise FramixError(f"⚠️ 通行证过期 -> {exp}")

    Design.Doc.log(
        f"[bold #87FF87]License verified. Access granted until [bold #5FD7FF]{exp}.\n"
    )

    issued, interval = auth_info["issued"], auth_info["interval"]
    delta_seconds = (now_time - datetime.fromisoformat(issued)).total_seconds()
    if delta_seconds > interval:
        await receive_license(code, lic_file)

    return auth_info


async def hide_lic_file(lic_file: "Path") -> typing.Optional[typing.Any]:
    """
    将授权文件（.lic）在本地系统中隐藏，防止用户误删或篡改。
    """
    if (ops := platform.system()) == "Windows":
        return await Terminal.cmd_line(["attrib", "+h", str(lic_file)])
    elif ops == "Darwin":
        return await Terminal.cmd_line(["chflags", "hidden", str(lic_file)])

    return None


async def send(
        client: "httpx.AsyncClient", method: str, url: str, *args, **kwargs
) -> typing.Any:
    """
    通过 httpx.AsyncClient 发送异步 HTTP 请求，并统一处理异常。
    """
    try:
        response = await client.request(method, url, *args, **kwargs)
        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as e:
        raise FramixError(f"❌ {e.response.status_code} -> {e.response.text}")
    except Exception as e:
        raise FramixError(f"❌ {e}")


async def receive_license(code: str, lic_file: "Path") -> typing.Optional["Path"]:
    """
    使用激活码从远程授权服务器获取授权文件，并保存至本地路径。
    """
    params = {
        "a": const.DESC,
        "t": int(time.time()),
        "n": secrets.token_hex(8)
    }
    payload = {
        "code": code.strip(),
        "castle": fingerprint(),
    } | params

    if lic_file.exists():
        auth_info = verify_signature(lic_file)
        payload["license_id"] = auth_info["license_id"]

    Design.Doc.log(
        f"[bold #FFAF5F]Transmitting glyph to central authority ..."
    )

    async with httpx.AsyncClient(headers=const.BASIC_HEADERS, timeout=30) as client:
        bs_lic_data = await send(client, "GET", const.BOOTSTRAP_URL, params=params)
        auth_info = verify_signature(bs_lic_data)
        Design.console.print_json(data=auth_info)
        activation_url = auth_info["url"]

        ac_lic_data = await send(client, "POST", activation_url, json=payload)
        auth_info = verify_signature(ac_lic_data)
        Design.console.print_json(data=auth_info)
        lic_file.write_text(json.dumps(ac_lic_data, indent=2), encoding=const.CHARSET)

        await hide_lic_file(lic_file)

        Design.Doc.log(
            f"[bold #87FF87]Validation succeeded. activation seal embedded.\n"
        )
        return lic_file


if __name__ == '__main__':
    pass
