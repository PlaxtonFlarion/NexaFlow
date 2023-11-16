import asyncio
import subprocess
from loguru import logger


class Terminal(object):

    @staticmethod
    async def cmd_line(*cmd: str):
        logger.debug(" ".join(cmd))
        try:
            events = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await events.communicate()
        except KeyboardInterrupt:
            logger.info("Stop with CTRL_C_EVENT ...")
            events = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await events.communicate()

        if stdout:
            return stdout.decode(encoding="UTF-8", errors="ignore").strip()
        if stderr:
            return stderr.decode(encoding="UTF-8", errors="ignore").strip()

    @staticmethod
    async def cmd_link(*cmd: str):
        logger.debug(" ".join(cmd))
        events = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        async def input_stream():
            async for line in events.stdout:
                logger.info(
                    line.decode(encoding="UTF-8", errors="ignore").strip()
                )

        async def error_stream():
            async for line in events.stderr:
                logger.info(
                    line.decode(encoding="UTF-8", errors="ignore").strip()
                )

        input_task = asyncio.create_task(input_stream(), name="input_task")
        error_task = asyncio.create_task(error_stream(), name="error_task")
        return events, input_task, error_task

    @staticmethod
    async def cmd_line_shell(cmd: str):
        logger.debug(cmd)
        try:
            events = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await events.communicate()
        except KeyboardInterrupt:
            logger.info("Stop with CTRL_C_EVENT ...")
            events = await asyncio.create_subprocess_exec(
                cmd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await events.communicate()

        if stdout:
            return stdout.decode(encoding="UTF-8", errors="ignore").strip()
        if stderr:
            return stderr.decode(encoding="UTF-8", errors="ignore").strip()

    @staticmethod
    async def cmd_link_shell(cmd: str):
        logger.debug(cmd)
        events = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        async def input_stream():
            async for line in events.stdout:
                logger.info(
                    line.decode(encoding="UTF-8", errors="ignore").strip()
                )

        async def error_stream():
            async for line in events.stderr:
                logger.info(
                    line.decode(encoding="UTF-8", errors="ignore").strip()
                )

        input_task = asyncio.create_task(input_stream(), name="input_task")
        error_task = asyncio.create_task(error_stream(), name="error_task")
        return events, input_task, error_task

###################################################################################

    @staticmethod
    def cmd_oneshot(cmd: list[str]):
        logger.debug(" ".join(cmd))
        try:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", bufsize=1
            )
        except KeyboardInterrupt:
            logger.info("Stop with CTRL_C_EVENT ...")
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", bufsize=1
            )

        if process.returncode != 0:
            return process.stderr
        return process.stdout

    @staticmethod
    def cmd_connect(cmd: list[str]):
        logger.debug(" ".join(cmd))
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1
        )
        return process


if __name__ == '__main__':
    pass
