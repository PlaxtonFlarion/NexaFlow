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

import re
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

compile_log: typing.Any = lambda x: Design.console.print(
    f"[bold]{const.DESC} | [bold #FFAF5F]Compiler[/] | {x}"
)


async def rename_so_files(root_dir: str) -> None:
    """
    将目录中所有形如 xxx.cpython-XXX-darwin.so 的文件重命名为 xxx.so

    Parameters
    ----------
    root_dir : str
        要扫描的根目录路径
    """
    pattern = re.compile(r"^(.*)\.cpython-\d{3}-darwin\.so$")

    for file in Path(root_dir).rglob("*.so"):
        if match := pattern.match(file.name):
            file.rename(file.with_name(new_name := f"{match.group(1)}.so"))
            compile_log(f"[bold #00D787][✓]Renamed {file.name} → {new_name}")


async def packaging() -> tuple[
    str, "Path", "Path", typing.Union["Path"], typing.Union["Path"], typing.Union["Path"]
]:
    app = Path(f"applications")
    # if (app := Path(f"applications")).exists():
    #     shutil.rmtree(app)
    # app.mkdir(exist_ok=True)

    venv_base_path = Path(".venv") if Path(".venv").exists() else Path("venv")

    site_packages, target, rename = None, None, None

    compile_cmd = [exe := sys.executable, "-m", "nuitka", "--standalone"]

    if (ops := sys.platform) == "win32":
        if (lib_path := venv_base_path / "Lib" / "site-packages").exists():
            site_packages = lib_path.resolve()

            target = app.joinpath(f"{const.DESC}.dist")
            rename = app.joinpath(f"{const.DESC}Engine")

            compile_cmd += [
                f"--windows-icon-from-ico=schematic/resources/icons/framix_icn_2.ico",
            ]

    elif ops == "darwin":
        if (lib_path := venv_base_path / "lib").exists():
            for sub in lib_path.iterdir():
                if sub.name.startswith("python"):
                    site_packages = (sub / "site-packages").resolve()
                elif "site-packages" in str(sub):
                    site_packages = sub.resolve()

                target = app.joinpath(f"{const.DESC}.app", f"Contents", f"MacOS")
                rename = app.joinpath(f"{const.DESC}.app")

                compile_cmd += [
                    f"--macos-create-app-bundle",
                    f"--macos-app-name={const.DESC}",
                    f"--macos-app-version={const.VERSION}",
                    f"--macos-app-icon=schematic/resources/images/macos/framix_macos_icn.png",
                ]

    else:
        raise RuntimeError(f"Unsupported platforms: {ops}")

    compile_cmd += [
        f"--nofollow-import-to=tensorflow,uiautomator2",
        f"--include-module=pdb,deprecation",
        f"--include-package=ml_dtypes,distutils,site,google,absl,wrapt,gast,astunparse,termcolor,opt_einsum,flatbuffers,h5py,adbutils,pygments",
        f"--show-progress", f"--show-memory", f"--output-dir={app}", f"{const.NAME}.py"
    ]

    compile_log(f"System={ops}")
    compile_log(f"Folder={app}")
    compile_log(
        f"{list(site_packages.parts[-4:]) if site_packages else site_packages}"
    )
    compile_log(f"Target={target}")
    compile_log(f"Rename={rename}")

    try:
        if writer := await asyncio.wait_for(
                Terminal.cmd_line([exe, "-m", "pip", "show", compile_cmd[2]]), timeout=5
        ):
            compile_log(f"Writer={writer}")
    except asyncio.TimeoutError as e:
        compile_log(f"Writer={compile_cmd[2]} {e}")
    else:
        compile_log(f"Writer={compile_cmd[2]}")

    return ops, app, site_packages, target, rename, compile_cmd


async def post_build() -> typing.Coroutine | None:

    async def input_stream() -> typing.Coroutine | None:
        """读取标准流"""
        async for line in transports.stdout:
            compile_log(line.decode(encoding=const.CHARSET, errors="ignore").strip())

    async def error_stream() -> typing.Coroutine | None:
        """读取异常流"""
        async for line in transports.stderr:
            compile_log(line.decode(encoding=const.CHARSET, errors="ignore").strip())

    async def examine_dependencies() -> typing.Coroutine | None:
        """自动查找虚拟环境中的 site-packages 路径，仅支持 Windows 与 macOS，兼容 .venv / venv。 """
        if not site_packages or not target:
            raise FileNotFoundError(f"Site packages path not found in virtual environment")

        for key, value in dependencies.items():
            for src, dst in value:
                if src.exists():
                    compile_log(f"[bold #00D787][✓] Read {key} -> {src.name}")
                    done_list.append((src, dst))
                else:
                    fail_list.append(f"{key}: {src.name}")
                    compile_log(f"[bold #FF6347][!] Dependency not found -> {src.name}")

        if fail_list:
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

            task = progress.add_task("Copy Dependencies", total=len(done_list))
            for src, dst in done_list:
                shutil.copytree(
                    src, dst, dirs_exist_ok=True
                ) if src.is_dir() else shutil.copy2(src, dst)
                progress.advance(task)

        main_dir = target if ops == "win32" else target.parent.parent
        shutil.move(main_dir, rename)
        compile_log(f"[bold #00D787][✓] Rename completed {main_dir.name} → {rename.name}")

        compile_log(f"[i]准备清理删除缓存目录 {apps.absolute()}")
        for cache in apps.iterdir():
            if cache.is_dir() and cache.name.endswith("build"):
                try:
                    shutil.rmtree(cache, ignore_errors=True)
                except Exception as e:
                    compile_log(f"[bold #FF6347][!] 无法清理: {e}")
                compile_log(f"[bold #00D787][✓] 构建缓存 {cache.name} 已清理")
        compile_log("[i]未发现 *.build 跳过清理")

        if ops == "darwin":
            compile_log(f"将目录中所有形如 xxx.cpython-XXX-darwin.so 的文件重命名为 xxx.so")
            await rename_so_files(target)

    Active.active("INFO")

    ops, apps, site_packages, target, rename, compile_cmd = await packaging()

    done_list, fail_list = [], []

    schematic = apps.parent.joinpath(const.F_SCHEMATIC)
    structure = apps.parent.joinpath(const.F_STRUCTURE, const.F_SRC_MODEL_PLACE)

    local_pack = [
        (schematic, target / schematic.name),
        (structure, target.parent / const.F_STRUCTURE / structure.name)
    ]

    wake = schematic.joinpath("resources", "automation")
    local_file = [
        (wake.joinpath(f"{const.NAME}.bat"), target.parent)
        if ops == "win32" else (wake.joinpath(f"{const.NAME}.sh"), target)
    ]

    dependencies = {
        "本地模块": local_pack,
        "本地文件": local_file,
        "第三方库": [
            (site_packages / dep, target / dep) for dep in [
                "keras", "tensorflow", "tensorboard", "uiautomator2", "sklearn"
            ]
        ]
    }

    await examine_dependencies()

    transports = await Terminal.cmd_link(compile_cmd)
    await asyncio.gather(
        *(asyncio.create_task(task()) for task in [input_stream, error_stream])
    )
    await transports.wait()

    await forward_dependencies()


if __name__ == "__main__":
    asyncio.run(post_build())
