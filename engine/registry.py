#   ____            _     _
#  |  _ \ ___  __ _(_)___| |_ _ __ _   _
#  | |_) / _ \/ _` | / __| __| '__| | | |
#  |  _ <  __/ (_| | \__ \ |_| |  | |_| |
#  |_| \_\___|\__, |_|___/\__|_|   \__, |
#             |___/                |___/
#
# ==== Notes: License ====
# Copyright (c) 2024  Framix :: 画帧秀
# This file is licensed under the Framix :: 画帧秀 License. See the LICENSE.md file for more details.

import json
import asyncio
from pathlib import Path
from engine.channel import (
    Channel, Messenger
)
from nexaflow import const


class Registry(object):
    """
    模板注册表，用于本地模板版本记录与远程模板自动同步。
    """

    def __init__(self, template_dir: "Path"):
        """
        初始化模板注册表。

        Parameters
        ----------
        template_dir : Path
            本地模板文件夹路径，用于存放下载的模板文件。
        """
        self.template_dir = template_dir
        self.version_file = self.template_dir / const.X_TEMPLATE_VERSION

    async def __load_local_versions(self) -> dict:
        """
        读取本地模板版本映射。

        Returns
        -------
        dict
            模板名到版本信息的映射，若无文件则返回空字典。
        """
        if self.version_file.exists():
            return json.loads(self.version_file.read_text())
        return {}

    async def __save_local_versions(self, version_map: dict) -> None:
        """
        保存本地模板版本映射至版本文件。

        Parameters
        ----------
        version_map : dict
            要保存的模板版本映射。
        """
        self.version_file.parent.mkdir(parents=True, exist_ok=True)
        self.version_file.write_text(json.dumps(version_map, indent=2))

    @staticmethod
    async def __fetch_remote_versions() -> dict:
        """
        从远程服务器拉取模板版本元信息。

        Returns
        -------
        dict
            远程模板名与版本号及其下载链接的映射。
        """
        params = Channel.make_params()
        async with Messenger() as messenger:
            resp = await messenger.poke("GET", const.TEMPLATE_META_URL, params=params)
            resp.raise_for_status()
            return resp.json()

    async def __download_template(self, url: str, template_name: str) -> None:
        """
        下载指定模板文件到本地模板目录。

        Parameters
        ----------
        url : str
            模板文件下载地址。

        template_name : str
            模板文件名称，用于本地保存。
        """
        params = Channel.make_params()
        async with Messenger() as messenger:
            resp = await messenger.poke("GET", url, params=params | {"page": template_name})
            resp.raise_for_status()
            template_path = self.template_dir / template_name
            template_path.parent.mkdir(parents=True, exist_ok=True)
            template_path.write_text(resp.text)

    async def sync_templates(self) -> None:
        """
        同步本地模板版本与远程版本。

        - 对比本地与远程模板版本；
        - 若远程版本更新，则自动下载模板；
        - 同步本地版本记录文件。
        """
        r_versions = await self.__fetch_remote_versions()

        try:
            l_versions = await self.__load_local_versions()
        except (FileNotFoundError, json.JSONDecodeError):
            l_versions = {}

        updated_local_version = l_versions.copy()

        if l_versions:
            download_map = {
                name: info for name, info in r_versions.items()
                if info["version"] > l_versions.get(name, {}).get("version", "")
            }
        else:
            download_map = r_versions

        for k, v in download_map.items():
            updated_local_version[k] = {"version": v["version"]}

        await asyncio.gather(
            *(self.__download_template(v["url"], k) for k, v in download_map.items())
        )
        return await self.__save_local_versions(updated_local_version)


if __name__ == '__main__':
    pass
