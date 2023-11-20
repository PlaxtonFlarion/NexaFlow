import asyncio
import subprocess
from loguru import logger


class Terminal(object):

    @staticmethod
    async def cmd_line(*cmd: str):
        logger.debug(" ".join(cmd))
        try:
            transports = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await transports.communicate()
        except KeyboardInterrupt:
            logger.info("Stop with CTRL_C_EVENT ...")
            transports = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await transports.communicate()

        if stdout:
            return stdout.decode(encoding="UTF-8", errors="ignore").strip()
        if stderr:
            return stderr.decode(encoding="UTF-8", errors="ignore").strip()

    @staticmethod
    async def cmd_link(*cmd: str):
        logger.debug(" ".join(cmd))
        transports = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        async def input_stream():
            async for line in transports.stdout:
                logger.info(
                    line.decode(encoding="UTF-8", errors="ignore").strip()
                )

        async def error_stream():
            async for line in transports.stderr:
                logger.info(
                    line.decode(encoding="UTF-8", errors="ignore").strip()
                )

        input_task = asyncio.create_task(input_stream(), name="input_task")
        error_task = asyncio.create_task(error_stream(), name="error_task")
        return transports, input_task, error_task

    @staticmethod
    async def cmd_line_shell(cmd: str):
        logger.debug(cmd)
        try:
            transports = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await transports.communicate()
        except KeyboardInterrupt:
            logger.info("Stop with CTRL_C_EVENT ...")
            transports = await asyncio.create_subprocess_exec(
                cmd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await transports.communicate()

        if stdout:
            return stdout.decode(encoding="UTF-8", errors="ignore").strip()
        if stderr:
            return stderr.decode(encoding="UTF-8", errors="ignore").strip()

    @staticmethod
    async def cmd_link_shell(cmd: str):
        logger.debug(cmd)
        transports = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        async def input_stream():
            async for line in transports.stdout:
                logger.info(
                    line.decode(encoding="UTF-8", errors="ignore").strip()
                )

        async def error_stream():
            async for line in transports.stderr:
                logger.info(
                    line.decode(encoding="UTF-8", errors="ignore").strip()
                )

        input_task = asyncio.create_task(input_stream(), name="input_task")
        error_task = asyncio.create_task(error_stream(), name="error_task")
        return transports, input_task, error_task

###################################################################################

    @staticmethod
    def cmd_oneshot(cmd: list[str]):
        logger.debug(" ".join(cmd))
        transports = subprocess.run(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True
        )

        if transports.returncode != 0:
            return transports.stderr
        return transports.stdout

    @staticmethod
    def cmd_connect(cmd: list[str]):
        logger.debug(" ".join(cmd))
        transports = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1
        )
        return transports


if __name__ == '__main__':
    pass
