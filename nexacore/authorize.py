import json
import base64
import socket
import struct
import typing
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

PUBLIC_KEY = b'''
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAwfWNE3uQEqRYYn/oBSIw
JlB6zcCJ/BpwItzrzlMLD1Q0nc7xtmraOzUPVQBe7xFXmI/I+C8lrlr3nSzu+Bpa
HZHzpsdg2MOqCb1IDvpSdhlXOoK8Xdnt0nNDqqgdLZjbQwA43D+ggEdlA/WK85ZO
mH8OPeQ9dWweEHDJN9xXx6LaNtEd3xtojNz+YSRdMONEYD0e7EiYXsWtKPRXf5Yl
TOjxCPl0IPgG3syUdy+fZjU2NGD631cN1l1H9Pjeuv2M4gAHgWbO+NjaZFPV7uXJ
KRcsac7eZ1ShnFZtFeuBTJjhQcDr0PhmqrgDPzd0VG2kA/LdGtfdCvO/cKaepBXo
fwIDAQAB
-----END PUBLIC KEY-----
'''


def __network_time() -> typing.Optional["datetime"]:
    """
    尝试通过多个 NTP 授时服务器获取当前的 UTC 网络时间。

    Returns
    -------
    Optional[datetime]
        若成功连接到授时服务器并获取响应，返回对应的 UTC 时间对象；
        若所有服务器都连接失败，则抛出 FramixError 异常。

    Notes
    -----
    - 使用 UDP 协议连接多个公共 NTP 服务器（包括阿里云、Windows、Apple 等）；
    - 请求包固定为 48 字节，其中首字节设置为 0x1b（NTP 协议标准）；
    - 使用 struct 解包服务器响应，提取第 11 个字段作为时间戳；
    - 所得时间基于 UTC，NTP 时间起点为 1900-01-01。

    Raises
    ------
    FramixError
        若所有服务器连接失败或解析异常，将抛出此异常提示网络时间不可用。
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
        "time1.baidu.com",  # 百度授时服务器
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
            Design.console.print(f"[bold #FF8787]{e}[/]")
            continue

    return None


def verify_license(lic_path: typing.Union[str, "Path"]) -> typing.Any:
    """
    验证授权文件的合法性与有效性，包括签名校验和过期检查。

    Parameters
    ----------
    lic_path : Union[str, Path]
        授权文件的路径，支持字符串或 pathlib.Path 对象。

    Returns
    -------
    Any
        若授权验证通过，返回授权信息（通常为字典形式）；
        若失败，则抛出 FramixError 异常。

    Notes
    -----
    - 使用内置公钥 `PUBLIC_KEY` 对授权文件中 `data` 和 `signature` 进行校验；
    - `data` 字段为 JSON 编码后的授权信息，`signature` 为其签名；
    - 验签通过后解析有效期（expire 字段），并与当前网络时间进行比对；
    - 若网络时间晚于过期日期，视为授权失效。

    Raises
    ------
    FramixError
        - 若授权格式错误、验签失败、文件读取异常；
        - 或当前时间已超过授权有效期。
    """
    Design.console.print(
        f"[bold #87D7FF][{const.DESC}::Authorize][/]"
        f"[bold #FFAF5F]Online check authorization ...[/]"
    )

    try:
        # 加载公钥
        pubkey = serialization.load_pem_public_key(PUBLIC_KEY)

        # 加载授权文件
        lic = json.loads(Path(lic_path).read_text())
        data = base64.b64decode(lic["data"])
        signature = base64.b64decode(lic["signature"])

        # 验签
        pubkey.verify(signature, data, padding.PKCS1v15(), hashes.SHA256())

        # 解析授权信息
        auth_info = json.loads(data)
        expire = datetime.strptime((exp := auth_info["expire"]), "%Y-%m-%d").replace(tzinfo=timezone.utc)

    except Exception as e:
        raise FramixError(f"❌ 授权验证失败 -> {e}")

    if not (now := __network_time()):
        raise FramixError(f"❌ 无法连接授时服务器 ...")
    if now > expire:
        raise FramixError(f"⚠️ 授权已过期 -> {exp}")

    Design.console.print(
        f"[bold #87D7FF][{const.DESC}::Authorize][/]"
        f"[bold #87FF87]Authorization verification passed. Valid until {exp}\n"
    )
    return auth_info


if __name__ == '__main__':
    pass
