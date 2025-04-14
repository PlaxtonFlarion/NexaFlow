#   ____        _ _     _
#  | __ ) _   _(_) | __| |
#  |  _ \| | | | | |/ _` |
#  | |_) | |_| | | | (_| |
#  |____/ \__,_|_|_|\__,_|
#

"""
版权所有 (c) 2024  Framix(画帧秀)
此文件受 Framix(画帧秀) 许可证的保护。您可以在 LICENSE.md 文件中查看详细的许可条款。

Copyright (c) 2024  Framix(画帧秀)
This file is licensed under the Framix(画帧秀) License. See the LICENSE.md file for more details.

# Copyright (c) 2024  Framix(画帧秀)
# このファイルは Framix(画帧秀) ライセンスの下でライセンスされています。詳細は LICENSE.md ファイルを参照してください。
"""

import os
import sys
import shutil
import typing
import asyncio
from pathlib import Path
from rich.progress import (
    BarColumn, TimeElapsedColumn,
    Progress, SpinnerColumn, TextColumn,
)
from engine.tinker import Active
from engine.terminal import Terminal
from nexacore.design import Design
from nexaflow import const


async def packaging() -> tuple[
    "Path", typing.Union["Path"], typing.Union["Path"], typing.Union["Path"]
]:

    operation_system, exe = sys.platform, sys.executable

    venv_base_path = Path(".venv") if Path(".venv").exists() else Path("venv")

    site_packages, target, rename = None, None, None

    compile_cmd = [exe, "-m", "nuitka", "--standalone"]

    if operation_system == "win32":
        if (lib_path := venv_base_path / "Lib" / "site-packages").exists():
            target = Path(f"applications/{const.DESC}.dist")
            site_packages, rename = lib_path.resolve(), Path(target.parent).joinpath(const.DESC)
            compile_cmd += [
                "--windows-icon-from-ico=resources/icons/framix_icn_2.ico",
            ]

    elif operation_system == "darwin":
        if (lib_path := venv_base_path / "lib").exists():
            for sub in lib_path.iterdir():
                if "site-packages" in str(sub):
                    target = Path(f"applications/{const.DESC}.app/Contents/MacOS")
                    site_packages, compile_cmd = sub.resolve(), None
                    compile_cmd += [
                        "--macos-create-app-bundle",
                        f"--macos-app-name={const.DESC}",
                        f"--macos-app-version={const.VERSION}",
                        "--macos-app-icon=resources/images/macos/framix_macos_icn.png",
                    ]

    else:
        raise RuntimeError(f"Unsupported platforms: {operation_system}")

    compile_cmd += [
        "--nofollow-import-to=tensorflow,uiautomator2",
        "--include-module=pdb,deprecation",
        "--include-package=ml_dtypes,distutils,site,google,absl,wrapt,gast,astunparse,termcolor,opt_einsum,flatbuffers,h5py,adbutils,pygments",
        "--show-progress", "--show-memory", "--output-dir=applications", f"{const.NAME}.py"
    ]

    return site_packages, target, rename, compile_cmd


async def post_build() -> typing.Coroutine | None:

    async def input_stream() -> typing.Coroutine | None:
        """读取标准流"""
        async for line in transports.stdout:
            line_fmt = line.decode(encoding=const.CHARSET, errors="ignore").strip()
            Design.console.print(f"[bold]{const.DESC} | [bold #FFAF5F]Compiler[/] | {line_fmt}")

    async def error_stream() -> typing.Coroutine | None:
        """读取异常流"""
        async for line in transports.stderr:
            line_fmt = line.decode(encoding=const.CHARSET, errors="ignore").strip()
            Design.console.print(f"[bold]{const.DESC} | [bold #FFAF5F]Compiler[/] | {line_fmt}")

    async def examine_dependencies() -> typing.Coroutine | None:
        """自动查找虚拟环境中的 site-packages 路径，仅支持 Windows 与 macOS，兼容 .venv / venv。 """
        if not site_packages or not target:
            raise FileNotFoundError(f"Site packages path not found in virtual environment")

        for dep in dependencies:
            src, dst = site_packages / dep, target / dep
            if src.exists():
                done_list.append((src, dst))
            else:
                fail_list.append(dep)
                Design.console.print(f"[bold #FF6347][!] Dependency not found -> {dep}")

        if schematic.exists():
            done_list.append(schematic.name)
        else:
            fail_list.append(schematic.name)
            Design.console.print(f"[bold #FF6347][!] Dependency not found -> {schematic.name}")

        if len(done_list) != len(dependencies + [schematic.name]):
            raise FileNotFoundError(f"Incomplete dependencies required {fail_list}")

    async def forward_dependencies() -> typing.Coroutine | None:
        """将指定依赖从虚拟环境复制到目标目录。"""
        with Progress(
                TextColumn(
                    text_format=f"[bold #80C0FF]{const.DESC} | {{task.description}}", justify="right"
                ),
                SpinnerColumn(
                    style="bold #FFA07A", speed=1, finished_text="[bold #7CFC00]✓"
                ),
                BarColumn(
                    bar_width=int(Design.console.width * 0.5), style="bold #ADD8E6",
                    complete_style="bold #90EE90", finished_style="bold #00CED1"
                ),
                TimeElapsedColumn(),
                TextColumn(
                    "[progress.percentage][bold #F0E68C]{task.completed:>2.0f}[/]/[bold #FFD700]{task.total}[/]"
                ),
                expand=False
        ) as progress:

            task = progress.add_task("Copy Dependencies", total=len(dependencies))

            for src, dst in done_list:
                shutil.copytree(src, dst, dirs_exist_ok=True)
                progress.advance(task)

            shutil.copytree(schematic, target.joinpath(const.F_SCHEMATIC), dirs_exist_ok=True)
            progress.advance(task)

        # 文件夹重命名
        if rename:
            shutil.move(target, rename)
            Design.console.print(f"[bold #00D787][✓] Rename completed {target.name} → {rename.name}")

    done_list, fail_list = [], []

    schematic, dependencies = Path(os.path.dirname(__file__)).joinpath(const.F_SCHEMATIC), [
        "uiautomator2", "keras", "tensorflow", "tensorboard",
    ]

    site_packages, target, rename, compile_cmd = await packaging()

    await examine_dependencies()

    Active.active("INFO")

    transports = await Terminal.cmd_link(compile_cmd)
    await asyncio.gather(
        *(asyncio.create_task(task()) for task in [input_stream, error_stream])
    )
    await transports.wait()

    await forward_dependencies()


if __name__ == "__main__":
    asyncio.run(post_build())
