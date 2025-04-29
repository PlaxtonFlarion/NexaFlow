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
import time
import shutil
import typing
import asyncio
from pathlib import Path
from rich.progress import (
    BarColumn, TimeElapsedColumn,
    Progress, SpinnerColumn, TextColumn,
)
from engine.tinker import (
    Active, FramixError
)
from engine.terminal import Terminal
from nexacore.design import Design
from nexaflow import const

compile_log: typing.Any = lambda x: Design.console.print(
    f"[bold]{const.DESC} | [bold #FFAF5F]Compiler[/] | {x}"
)


async def is_virtual_env() -> typing.Coroutine | None:
    """
    检查当前 Python 运行环境是否为虚拟环境。
    """
    if sys.prefix != sys.base_prefix:
        return compile_log("[bold #00D787][✓] 当前运行在虚拟环境中")

    raise FramixError("[!] 当前不是虚拟环境")


async def find_site_packages() -> typing.Coroutine | "Path":
    """
    自动查找当前虚拟环境中的 `site-packages` 路径。
    """
    base_venv, base_libs, base_site = Path(sys.prefix), "lib", "site-packages"

    for lib_path in base_venv.iterdir():
        if base_libs in lib_path.name.lower():
            for sub in lib_path.iterdir():
                if base_site in sub.name.lower():
                    return sub.resolve()
                elif sub.name.lower().startswith("python"):
                    return (sub / base_site).resolve()

    raise FramixError(f"[!] Site packages path not found in virtual environment")


async def rename_sensitive(src: "Path", dst: "Path") -> typing.Coroutine | None:
    """
    执行双重重命名操作以避免系统锁定或覆盖冲突。
    """
    temporary = src.with_name(f"__temp_{time.strftime('%Y%m%d%H%M%S')}__")
    compile_log(f"[bold #00D787][✓] 生成临时目录 {temporary.name}")

    src.rename(temporary)
    compile_log(f"[bold #00D787][✓] Rename completed {src.name} → {temporary.name}")
    temporary.rename(dst)
    compile_log(f"[bold #00D787][✓] Rename completed {temporary.name} → {dst.name}")


async def sweep_cache_tree(target: "Path") -> typing.Coroutine | None:
    """
    清理指定路径下所有名为 `*build` 的缓存目录。
    """
    compile_log(f"[!] 准备清理删除缓存目录 {target.as_posix()}")
    if caches := [
        cache for cache in target.iterdir() if cache.is_dir() and cache.name.lower().endswith("build")
    ]:
        for cache in await asyncio.gather(
                *(asyncio.to_thread(
                    shutil.rmtree, cache, ignore_errors=True) for cache in caches), return_exceptions=True
        ):
            if isinstance(cache, Exception):
                compile_log(f"[bold #FF6347][✗] 无法清理 {cache}")
            else:
                compile_log(f"[bold #00D787][✓] 构建缓存已清理 {cache}")
    else:
        compile_log("[!] 未发现 *.build 跳过清理")


async def rename_so_files(ops: str, target: "Path") -> typing.Coroutine | None:
    """
    将 Darwin 系统下编译生成的 *.cpython-XXX-darwin.so 文件统一重命名为 *.so。
    """
    if ops != "darwin":
        return None

    compile_log(f"[!] 将目录中所有形如 xxx.cpython-XXX-darwin.so 的文件重命名为 xxx.so")

    pattern = re.compile(r"^(.*)\.cpython-\d{3}-darwin\.so$")

    for file in target.rglob("*.so"):
        if match := pattern.match(file.name):
            file.rename(file.with_name(new_name := f"{match.group(1)}.so"))
            compile_log(f"[bold #00D787][✓] Renamed {file.name} → {new_name}")


async def authorized_tools(ops: str, *args: "Path", **__) -> typing.Coroutine | None:
    """
    检查目录下的所有文件是否具备执行权限，如果文件没有执行权限，则自动添加 +x 权限。
    """
    if ops != "darwin":
        return None

    for resp in (ensure := [
        ["chmod", "-R", "+x", arg.as_posix()] if arg.is_dir() else [
            "chmod", "+x", arg.as_posix()] for arg in args
    ]):
        compile_log(f"[!] Authorizing {resp}")

    for resp in await asyncio.gather(
            *(Terminal.cmd_line(kit) for kit in ensure)
    ):
        compile_log(f"[!] Authorize resp={resp}")


async def packaging() -> tuple[
    str, "Path", "Path", typing.Union["Path"],
    typing.Union[tuple["Path", "Path"]], list[str], tuple["Path", "Path"], str
]:
    """
    构建独立应用的打包编译命令与目录结构信息。
    """
    if (app := Path(f"applications")).exists():
        shutil.rmtree(app)
    app.mkdir(exist_ok=True)

    site_packages = await find_site_packages()

    launch = app.parent / const.F_SCHEMATIC / "resources" / "automation"

    compile_cmd = [exe := sys.executable, "-m", "nuitka", "--standalone"]

    if (ops := sys.platform) == "win32":
        target = app / f"{const.DESC}.dist"
        rename = target, app / f"{const.DESC}Engine"

        compile_cmd += [
            f"--windows-icon-from-ico=schematic/resources/icons/framix_icn_2.ico",
        ]

        launch = launch / f"{const.NAME}.bat", target.parent

        support = "Windows"

    elif ops == "darwin":
        target = app / f"{const.DESC}.app" / f"Contents" / f"MacOS"
        rename = target.parent.parent, app / f"{const.DESC}.app"

        compile_cmd += [
            f"--macos-create-app-bundle",
            f"--macos-app-name={const.DESC}",
            f"--macos-app-version={const.VERSION}",
            f"--macos-app-icon=schematic/resources/images/macos/framix_macos_icn.png",
        ]

        launch = launch / f"{const.NAME}.sh", target

        support = "MacOS"

    else:
        raise FramixError(f"Unsupported platforms {ops}")

    compile_cmd += [
        f"--nofollow-import-to=keras,tensorflow,tensorboard,uiautomator2",
        f"--include-module=pdb,deprecation",
        f"--include-package=engine,nexacore,nexaflow,sklearn,sklearn.tree",
        f"--include-package=ml_dtypes,distutils,site,google,absl,wrapt,gast",
        f"--include-package=astunparse,termcolor,opt_einsum,flatbuffers,h5py",
        f"--include-package=adbutils,pygments",
        f"--show-progress", f"--show-memory", f"--output-dir={app}", f"{const.NAME}.py"
    ]

    compile_log(f"system={ops}")
    compile_log(f"folder={app}")
    compile_log(f"packet={site_packages}")
    compile_log(f"target={target}")
    compile_log(f"rename={rename}")
    compile_log(f"launch={launch}")

    try:
        if writer := await asyncio.wait_for(
                Terminal.cmd_line([exe, "-m", "pip", "show", compile_cmd[2]]), timeout=5
        ):
            compile_log(f"writer={writer}")
        else:
            compile_log(f"writer={compile_cmd[2]}")
    except asyncio.TimeoutError as e:
        compile_log(f"writer={compile_cmd[2]} {e}")

    return ops, app, site_packages, target, rename, compile_cmd, launch, support


async def post_build() -> typing.Coroutine | None:
    """
    应用打包后的自动依赖检查与部署流程。
    """

    async def input_stream() -> typing.Coroutine | None:
        """
        异步读取终端标准输出流内容，并打印实时构建日志。
        """
        async for line in transports.stdout:
            compile_log(line.decode(const.CHARSET, "ignore").strip())

    async def error_stream() -> typing.Coroutine | None:
        """
        异步读取终端错误输出流内容，实时反馈构建异常信息。
        """
        async for line in transports.stderr:
            compile_log(line.decode(const.CHARSET, "ignore").strip())

    async def examine_dependencies() -> typing.Coroutine | None:
        """
        检查所有指定依赖是否存在于虚拟环境中。
        """
        for key, value in dependencies.items():
            for src, dst in value:
                if src.exists():
                    done_list.append((src, dst))
                    compile_log(f"[bold #00D787][✓] Read {key} -> {src.name}")
                else:
                    fail_list.append(f"{key}: {src.name}")
                    compile_log(f"[bold #FF6347][!] Dependency not found -> {src.name}")

        if fail_list:
            raise FramixError(f"[!] Incomplete dependencies required {fail_list}")

    async def forward_dependencies() -> typing.Coroutine | None:
        """
        拷贝所有依赖文件与目录至编译产物路径，并执行重命名与缓存清理。
        """
        bar_width = int(Design.console.width * 0.3)

        with Progress(
                TextColumn(text_format=f"[bold #80C0FF]{const.DESC} | {{task.description}}", justify="right"),
                SpinnerColumn(style="bold #FFA07A", speed=1, finished_text="[bold #7CFC00]✓"),
                BarColumn(bar_width, style="bold #ADD8E6", complete_style="bold #90EE90", finished_style="bold #00CED1"),
                TimeElapsedColumn(),
                TextColumn(
                    "[progress.percentage][bold #F0E68C]{task.completed:>2.0f}[/]/[bold #FFD700]{task.total}[/]"
                ), expand=False
        ) as progress:

            task = progress.add_task(description="Dependencies", total=len(done_list))
            for src, dst in done_list:
                shutil.copytree(
                    src, dst, dirs_exist_ok=True) if src.is_dir() else shutil.copy2(src, dst)
                progress.advance(task)

        await authorized_tools(
            ops, target / schematic.name / kit, target / const.NAME, target / launch[0].name
        )  # Note: macOS Only

        await rename_so_files(ops, target)  # Note: macOS Only

        await rename_sensitive(*rename)
        await sweep_cache_tree(app)

    # ==== Note: Start from here ====
    build_start_time = time.time()

    Active.active("INFO")

    await Design.show_quantum_intro()

    ops, app, site_packages, target, rename, compile_cmd, launch, support = await packaging()

    done_list, fail_list = [], []

    schematic, kit = app.parent / const.F_SCHEMATIC, "supports"
    r, s, t = schematic / "resources", schematic / kit / support, schematic / "templates"

    local_pack, local_file = [
        (r, target / schematic.name / r.name),
        (s, target / schematic.name / kit / s.name),
        (t, target / schematic.name / t.name),
    ], [
        launch
    ]

    for folder in [const.F_SRC_OPERA_PLACE, const.F_SRC_MODEL_PLACE, const.F_SRC_TOTAL_PLACE]:
        if not (child := target.parent / const.F_STRUCTURE / folder).exists():
            await asyncio.to_thread(child.mkdir, parents=True, exist_ok=True)

    dependencies = {
        "本地模块": local_pack,
        "本地文件": local_file,
        "第三方库": [
            (site_packages / dep, target / dep) for dep in [
                "keras", "tensorflow", "tensorboard", "uiautomator2"
            ]
        ]
    }

    await examine_dependencies()

    # Note: Start compiling
    transports = await Terminal.cmd_link(compile_cmd)
    await asyncio.gather(
        *(asyncio.create_task(task()) for task in [input_stream, error_stream])
    )
    await transports.wait()

    await forward_dependencies()

    compile_log(f"TimeCost={(time.time() - build_start_time) / 60:.2f} m")


if __name__ == "__main__":
    try:
        asyncio.run(post_build())
    except FramixError as _e:
        compile_log(_e)
        Design.fail()
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(Design.exit())
    else:
        sys.exit(Design.done())
