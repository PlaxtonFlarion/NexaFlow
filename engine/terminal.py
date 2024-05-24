import os
import sys
import asyncio
import subprocess
from loguru import logger


class Terminal(object):

    @staticmethod
    async def cmd_line(*cmd: str, **kwargs):
        logger.debug(" ".join([os.path.basename(c) for c in cmd]))
        transports = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **kwargs
        )

        encode = "GBK" if sys.platform == "win32" else "UTF-8"
        stdout, stderr = await transports.communicate(**kwargs)

        if stdout:
            return stdout.decode(encoding=encode, errors="ignore").strip()
        if stderr:
            return stderr.decode(encoding=encode, errors="ignore").strip()

    @staticmethod
    async def cmd_link(*cmd: str, **kwargs):
        logger.debug(" ".join([os.path.basename(c) for c in cmd]))
        transports = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **kwargs
        )
        return transports

########################################################################################################################

    @staticmethod
    async def cmd_line_shell(cmd: str, **kwargs):
        logger.debug(" ".join([os.path.basename(c) for c in cmd.split()]))
        transports = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **kwargs
        )

        encode = "GBK" if sys.platform == "win32" else "UTF-8"
        stdout, stderr = await transports.communicate()

        if stdout:
            return stdout.decode(encoding=encode, errors="ignore").strip()
        if stderr:
            return stderr.decode(encoding=encode, errors="ignore").strip()

    @staticmethod
    async def cmd_link_shell(cmd: str, **kwargs):
        logger.debug(" ".join([os.path.basename(c) for c in cmd.split()]))
        transports = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **kwargs
        )
        return transports

########################################################################################################################

    @staticmethod
    def cmd_oneshot(cmd: list[str], **kwargs):
        logger.debug(" ".join([os.path.basename(c) for c in cmd]))
        transports = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs
        )

        if transports.returncode != 0:
            return transports.stderr
        return transports.stdout

    @staticmethod
    def cmd_connect(cmd: list[str], **kwargs):
        logger.debug(" ".join([os.path.basename(c) for c in cmd]))
        transports = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1, **kwargs
        )
        return transports


if __name__ == '__main__':
    pass
