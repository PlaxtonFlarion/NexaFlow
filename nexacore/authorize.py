#      _         _   _                _
#     / \  _   _| |_| |__   ___  _ __(_)_______
#    / _ \| | | | __| '_ \ / _ \| '__| |_  / _ \
#   / ___ \ |_| | |_| | | | (_) | |  | |/ /  __/
#  /_/   \_\__,_|\__|_| |_|\___/|_|  |_/___\___|
#
# ==== Notes: License ====
# Copyright (c) 2024  Framix :: 画帧秀
# This file is licensed under the Framix :: 画帧秀 License. See the LICENSE.md file for more details.

import json
import uuid
import base64
import socket
import struct
import typing
import hashlib
import urllib.request
from pathlib import Path
from datetime import (
    datetime, timezone
)
from cryptography.hazmat.primitives import (
    hashes, serialization
)
from cryptography.hazmat.primitives.asymmetric import padding
from engine.tinker import FramixError
from nexacore.design import Design
from nexaflow import const


def machine_id() -> str:
    """
    获取当前设备唯一识别码（基于 MAC 地址 + 哈希），推荐长度12位。
    """
    machine = uuid.getnode()
    return hashlib.sha256(str(machine).encode()).hexdigest()[:12]


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


def receive_license(code: str, lic_path: "Path") -> typing.Optional["Path"]:
    """
    使用激活码从远程授权服务器获取授权文件，并保存至本地路径。
    """
    Design.Doc.log(f"[bold #FFAF5F]Transmitting signature to the authority ...")

    payload = json.dumps({"code": code.strip(), "castle": machine_id()}).encode(const.CHARSET)

    req = urllib.request.Request(
        url=const.ACTIVATION_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            lic = json.loads(resp.read().decode())
            lic_path.write_text(json.dumps(lic, indent=2), encoding=const.CHARSET)
            Design.Doc.log(f"[bold #87FF87]License validated successfully. System activated.\n")

    except urllib.request.HTTPError as e:
        raise FramixError(f"❌ [{e.code}] -> {e.read().decode()}")
    except Exception as e:
        raise FramixError(f"❌ {e}")


def verify_license(lic_path: typing.Union[str, "Path"]) -> typing.Any:
    """
    验证授权文件的合法性与有效性。
    """
    Design.Doc.log(f"[bold #FFAF5F]Online check authorization ...")

    try:
        # 加载公钥
        pubkey = serialization.load_pem_public_key(const.PUBLIC_KEY)

        # 加载授权文件
        lic = json.loads(Path(lic_path).read_text())
        data = base64.b64decode(lic["data"])
        signature = base64.b64decode(lic["signature"])

        # 验签
        pubkey.verify(signature, data, padding.PKCS1v15(), hashes.SHA256())

        # 解析授权信息
        auth_info = json.loads(data)
        expire = datetime.strptime((exp := auth_info["expire"]), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        castle = auth_info["castle"]

    except Exception as e:
        raise FramixError(f"❌ 授权验证失败 -> {e}")

    if castle != machine_id():
        raise FramixError("❌ 当前设备与授权文件不匹配 ...")

    if not (now := network_time()):
        raise FramixError(f"❌ 无法连接授时服务器 ...")
    if now > expire:
        raise FramixError(f"⚠️ 授权已过期 -> {exp}")

    Design.Doc.log(f"[bold #87FF87]License file verified. Execution is authorized. Valid until {exp}\n")
    return auth_info


if __name__ == '__main__':
    pass
