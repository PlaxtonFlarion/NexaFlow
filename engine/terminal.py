#
#   _____                   _             _
#  |_   _|__ _ __ _ __ ___ (_)_ __   __ _| |
#    | |/ _ \ '__| '_ ` _ \| | '_ \ / _` | |
#    | |  __/ |  | | | | | | | | | | (_| | |
#    |_|\___|_|  |_| |_| |_|_|_| |_|\__,_|_|
#

import typing
import asyncio
import subprocess
from loguru import logger
from nexaflow import const


class Terminal(object):
    """
    异步命令行执行工具类。

    该类封装了基于 asyncio 的子进程命令执行方法，提供标准输入输出交互接口，
    可用于跨平台执行系统命令、调试命令行工具、或与外部程序（如 ffmpeg、adb）通信。

    Notes
    -----
    - 所有方法均为静态方法，可直接通过类名调用。
    - 默认使用 UTF-8 或 `const.CHARSET` 进行解码，自动忽略无效字符。
    - 支持传入 stdin 数据、附加执行参数，以及自定义管道通信方式。
    - 推荐用于执行一次性命令（`cmd_line`）或需要控制生命周期的子进程（`cmd_link`）。
    """

    @staticmethod
    async def cmd_line(cmd: list[str], transmit: typing.Optional[bytes] = None) -> typing.Any:
        """
        异步执行命令行指令，并返回标准输出或标准错误信息。

        该方法通过 asyncio 的子进程功能，执行一个命令行指令（可附带输入），并在命令执行完成后返回对应输出。

        Parameters
        ----------
        cmd : list[str]
            要执行的命令及其参数列表。

        transmit : Optional[bytes], default=None
            可选的标准输入数据，若命令需要通过 stdin 传递参数可使用。

        Returns
        -------
        typing.Any
            若命令执行成功，返回 `stdout` 解码后的字符串；若有错误信息，返回 `stderr` 解码后的字符串；否则返回 None。

        Notes
        -----
        - 所有输出均使用项目中定义的默认字符集 `const.CHARSET` 进行解码，忽略非法字符。
        - 若 `stdout` 和 `stderr` 均为空，将返回 None。
        - 该方法适合用于一次性执行任务并等待返回结果。

        Workflow
        --------
        1. 异步创建子进程并执行命令。
        2. 若提供 `transmit` 数据，则通过 stdin 传输。
        3. 等待子进程完成并获取 `stdout` 和 `stderr`。
        4. 返回输出结果（优先返回 stdout，否则返回 stderr）。
        """
        logger.debug(cmd)

        transports = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await transports.communicate(transmit)

        if stdout:
            return stdout.decode(encoding=const.CHARSET, errors="ignore").strip()
        if stderr:
            return stderr.decode(encoding=const.CHARSET, errors="ignore").strip()

    @staticmethod
    async def cmd_link(cmd: list[str]) -> "asyncio.subprocess.Process":
        """
        异步执行命令行指令，并返回子进程传输句柄。

        该方法与 `cmd_line` 类似，但不等待命令执行结束，而是直接返回进程对象（用于流式或手动控制的场景）。

        Parameters
        ----------
        cmd : list[str]
            要执行的命令及其参数列表。

        Returns
        -------
        asyncio.subprocess.Process
            返回 asyncio 创建的子进程对象（transports），可手动调用 `.communicate()`、`.stdin.write()` 等方法继续交互。

        Notes
        -----
        - 与 `cmd_line` 不同，该方法不自动处理输出，也不会解码。
        - 适合用于需要长时间运行或交互式输入输出的子进程场景。

        Workflow
        --------
        1. 异步启动子进程并打开 stdout/stderr 管道。
        2. 返回子进程 transport 对象供外部使用。
        """
        logger.debug(cmd)

        transports = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        return transports

########################################################################################################################

    @staticmethod
    async def cmd_line_shell(cmd: str, transmit: typing.Optional[bytes] = None) -> typing.Any:
        """
        异步执行 Shell 字符串命令，并返回执行结果。

        Parameters
        ----------
        cmd : str
            要执行的命令字符串，支持 Shell 特性（如管道、重定向等）。

        transmit : bytes, optional
            传递给子进程标准输入的数据，默认为 None。

        Returns
        -------
        typing.Any
            解码后的命令行输出结果（stdout 优先返回，其次 stderr），若无输出则返回 None。

        Notes
        -----
        - 使用 `create_subprocess_shell` 执行字符串命令，适用于包含管道或多条命令的复杂 Shell 调用。
        - 命令完成后自动收集 stdout 和 stderr 并返回文本结果。
        - 若标准输出与错误输出都为空，返回 None。

        Workflow
        --------
        1. 构造 Shell 命令并启动异步子进程。
        2. 将 `transmit` 数据写入标准输入（若有）。
        3. 等待命令执行完成，并获取 stdout / stderr。
        4. 优先返回标准输出，其次返回错误输出。
        """
        logger.debug(cmd)

        transports = await asyncio.create_subprocess_shell(
            cmd,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await transports.communicate(transmit)

        if stdout:
            return stdout.decode(encoding=const.CHARSET, errors="ignore").strip()
        if stderr:
            return stderr.decode(encoding=const.CHARSET, errors="ignore").strip()

    @staticmethod
    async def cmd_link_shell(cmd: str) -> "asyncio.subprocess.Process":
        """
        异步启动 Shell 子进程，返回进程对象以供后续控制。

        Parameters
        ----------
        cmd : str
            要执行的 Shell 命令字符串，支持管道、逻辑操作等。

        Returns
        -------
        asyncio.subprocess.Process
            异步子进程对象，可用于后续控制（如读取输出、手动传输数据等）。

        Notes
        -----
        - 适用于需要持续交互或手动读取/写入 stdout/stderr 的场景。
        - 子进程不会自动关闭，需使用者手动控制其生命周期。
        - 与 `cmd_line_shell` 不同，本方法不自动等待命令完成。

        Workflow
        --------
        1. 构造 Shell 命令并异步启动子进程。
        2. 返回 subprocess 对象以供调用者进一步操作（如 `.communicate()`、`.kill()`）。
        """
        logger.debug(cmd)

        transports = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        return transports

########################################################################################################################

    @staticmethod
    def cmd_oneshot(cmd: list[str]) -> typing.Any:
        """
        同步执行一次性命令，并返回标准输出或错误信息。

        Parameters
        ----------
        cmd : list[str]
            要执行的命令列表（非 Shell 字符串），如 ["adb", "devices"]。

        Returns
        -------
        typing.Any
            命令执行结果的字符串：
            - 若命令成功（return_code == 0），返回 stdout；
            - 若命令失败（return_code != 0），返回 stderr。

        Notes
        -----
        - 此方法适用于短时、阻塞式的命令执行。
        - 标准输出与错误输出均以字符串形式返回，自动解码为 UTF-8。
        - 若命令格式错误或无法执行，可能抛出 `subprocess.SubprocessError` 等异常。

        Workflow
        --------
        1. 调用 `subprocess.run` 执行指定命令。
        2. 等待命令完成并获取 stdout 与 stderr。
        3. 根据退出状态码返回对应结果。
        """
        logger.debug(cmd)

        transports = subprocess.run(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        return transports.stderr if transports.returncode != 0 else transports.stdout

    @staticmethod
    def cmd_connect(cmd: list[str]) -> "subprocess.Popen":
        """
         启动一个长生命周期的子进程连接，返回进程对象以便持续交互。

         Parameters
         ----------
         cmd : list[str]
             要执行的命令列表，通常用于长期运行的进程（如设备监听、持续输出任务）。

         Returns
         -------
         subprocess.Popen
             返回 `subprocess.Popen` 创建的子进程对象，可供调用者读取标准输出或手动终止进程。

         Notes
         -----
         - 该方法适合处理持续输出或交互式任务，如 ADB logcat、服务启动器等。
         - 子进程不会自动关闭，需用户手动控制其生命周期。
         - 输出为行缓冲（`1`），以便实时处理每行数据。

         Workflow
         --------
         1. 使用 `subprocess.Popen` 启动命令行进程。
         2. 设置标准输出和标准错误为管道，并启用 UTF-8 编码。
         3. 返回进程对象供调用者后续操作（如读取输出、关闭进程）。
         """
        logger.debug(cmd)

        transports = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding=const.CHARSET, bufsize=1
        )

        return transports


if __name__ == '__main__':
    pass
