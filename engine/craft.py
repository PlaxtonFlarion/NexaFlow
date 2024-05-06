import asyncio
import os
import typing
import aiofiles
from rich.prompt import Prompt
from frameflow.skills.show import Show
from nexaflow import const


class Craft(object):

    @staticmethod
    async def achieve(
            template: typing.Union[str, "os.PathLike"]
    ) -> typing.Union[str, "Exception"]:

        try:
            async with aiofiles.open(template, "r", encoding=const.CHARSET) as f:
                template_file = await f.read()
        except FileNotFoundError as e:
            return e
        return template_file

    @staticmethod
    async def ask_input(tips):
        return await asyncio.get_running_loop().run_in_executor(
            None, Prompt.ask, f"[bold #5FD7FF]{tips}[/]", Show.console
        )


if __name__ == '__main__':
    pass
