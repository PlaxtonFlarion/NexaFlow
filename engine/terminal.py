import os
import sys
import typing
import asyncio
import subprocess
from loguru import logger


class Terminal(object):

    @staticmethod
    async def cmd_line(*cmd: str, transmit: typing.Optional[bytes] = None, **kwargs):
        """
        异步执行命令行指令，并获取其输出。

        该方法使用异步子进程执行传入的命令行指令，支持传递输入数据，并返回标准输出或标准错误信息。

        参数:
            *cmd (str): 命令行指令及其参数，作为不定长字符串参数传入。
            transmit (Optional[bytes]): 传递给子进程的标准输入数据，默认为None。
            **kwargs: 传递给 asyncio.create_subprocess_exec 的其他关键字参数，例如cwd、env等。

        返回:
            str: 子进程的标准输出或标准错误信息，经过解码和去除首尾空格。

        注意:
            - 使用 asyncio.create_subprocess_exec 创建子进程，并将其标准输出和标准错误重定向到管道。
            - 根据操作系统，选择适当的编码方式（Windows使用GBK，其他系统使用UTF-8）。
            - 执行命令后，等待子进程完成并读取其标准输出和标准错误内容。
            - 若标准输出有内容，返回解码后的标准输出；否则，返回解码后的标准错误。

        异常处理:
            - 解码时忽略错误，确保不会因解码问题导致程序崩溃。
        """
        logger.debug(" ".join([os.path.basename(c) for c in cmd]))
        transports = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **kwargs
        )

        encode = "GBK" if sys.platform == "win32" else "UTF-8"
        stdout, stderr = await transports.communicate(transmit)

        if stdout:
            return stdout.decode(encoding=encode, errors="ignore").strip()
        if stderr:
            return stderr.decode(encoding=encode, errors="ignore").strip()

    @staticmethod
    async def cmd_link(*cmd: str, **kwargs):
        """
        异步执行命令行指令，并返回子进程对象。

        该方法使用异步子进程执行传入的命令行指令，并返回子进程对象以便进一步处理。

        参数:
            *cmd (str): 命令行指令及其参数，作为不定长字符串参数传入。
            **kwargs: 传递给 asyncio.create_subprocess_exec 的其他关键字参数，例如cwd、env等。

        返回:
            transports: 子进程对象，用于获取子进程的状态和输出。

        注意:
            - 使用 asyncio.create_subprocess_exec 创建子进程，并将其标准输出和标准错误重定向到管道。
            - 执行命令后，返回子进程对象，便于后续操作如等待子进程完成或获取其输出。
        """
        logger.debug(" ".join([os.path.basename(c) for c in cmd]))
        transports = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **kwargs
        )
        return transports

########################################################################################################################

    @staticmethod
    async def cmd_line_shell(cmd: str, transmit: typing.Optional[bytes] = None, **kwargs):
        """
        异步执行Shell命令，并返回命令输出。

        该方法使用异步子进程执行传入的Shell命令，并返回命令的标准输出或标准错误。

        参数:
            cmd (str): Shell命令字符串。
            transmit (Optional[bytes]): 传递给子进程的输入数据，默认为None。
            **kwargs: 传递给 asyncio.create_subprocess_shell 的其他关键字参数，例如cwd、env等。

        返回:
            str: 子进程的标准输出或标准错误的解码结果。

        注意:
            - 使用 asyncio.create_subprocess_shell 创建子进程，并将其标准输出和标准错误重定向到管道。
            - 根据系统平台选择合适的编码方式（Windows使用GBK，其他平台使用UTF-8）。
            - 执行命令后，等待子进程完成并获取其输出。
            - 如果标准输出不为空，则返回标准输出的解码结果；否则，返回标准错误的解码结果。
        """
        logger.debug(" ".join([os.path.basename(c) for c in cmd.split()]))
        transports = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **kwargs
        )

        encode = "GBK" if sys.platform == "win32" else "UTF-8"
        stdout, stderr = await transports.communicate(transmit)

        if stdout:
            return stdout.decode(encoding=encode, errors="ignore").strip()
        if stderr:
            return stderr.decode(encoding=encode, errors="ignore").strip()

    @staticmethod
    async def cmd_link_shell(cmd: str, **kwargs):
        """
        异步执行Shell命令，并返回子进程对象。

        该方法使用异步子进程执行传入的Shell命令，并返回创建的子进程对象。

        参数:
            cmd (str): Shell命令字符串。
            **kwargs: 传递给 asyncio.create_subprocess_shell 的其他关键字参数，例如cwd、env等。

        返回:
            asyncio.subprocess.Process: 创建的子进程对象。

        注意:
            - 使用 asyncio.create_subprocess_shell 创建子进程，并将其标准输出和标准错误重定向到管道。
            - 子进程对象可以用于进一步与子进程交互，例如读取输出或等待其完成。
        """
        logger.debug(" ".join([os.path.basename(c) for c in cmd.split()]))
        transports = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **kwargs
        )
        return transports

########################################################################################################################

    @staticmethod
    def cmd_oneshot(cmd: list[str], **kwargs):
        """
        同步执行命令，并返回执行结果。

        该方法使用同步方式执行传入的命令，并返回执行结果的标准输出或标准错误。

        参数:
            cmd (list[str]): 命令及其参数列表。
            **kwargs: 传递给 subprocess.run 的其他关键字参数，例如cwd、env等。

        返回:
            str: 如果命令执行成功，返回标准输出；如果命令执行失败，返回标准错误。

        注意:
            - 使用 subprocess.run 同步执行命令，并将其标准输出和标准错误重定向到管道。
            - 如果命令返回码不为0，返回标准错误；否则返回标准输出。
        """
        logger.debug(" ".join([os.path.basename(c) for c in cmd]))
        transports = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs
        )

        if transports.returncode != 0:
            return transports.stderr
        return transports.stdout

    @staticmethod
    def cmd_connect(cmd: list[str], **kwargs):
        """
        同步启动命令，并返回进程对象。

        该方法使用同步方式启动传入的命令，并返回一个进程对象，用于进一步的标准输出和标准错误处理。

        参数:
            cmd (list[str]): 命令及其参数列表。
            **kwargs: 传递给 subprocess.Popen 的其他关键字参数，例如cwd、env等。

        返回:
            subprocess.Popen: 表示已启动进程的对象。

        注意:
            - 使用 subprocess.Popen 启动命令，并将其标准输出和标准错误重定向到管道。
            - 进程对象可以用于进一步的交互操作，如读取输出流、等待进程结束等。
        """
        logger.debug(" ".join([os.path.basename(c) for c in cmd]))
        transports = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1, **kwargs
        )
        return transports


if __name__ == '__main__':
    pass
