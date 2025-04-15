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
from engine.tinker import Active
from engine.terminal import Terminal
from nexacore.design import Design
from nexaflow import const

compile_log: typing.Any = lambda x: Design.console.print(
    f"[bold]{const.DESC} | [bold #FFAF5F]Compiler[/] | {x}"
)


class Compile(object):
    """Compile"""

    ops: str
    app: "Path"
    site_packages: "Path"
    target: typing.Union["Path"]
    rename: typing.Union[tuple["Path", "Path"]]
    compile_cmd: list[str]
    launch: list[str]

    def __init__(self, *args, **__):
        self.ops, self.app, self.site_packages, *_ = args
        _, _, _, self.target, self.rename, *_ = args
        *_, self.compile_cmd, self.launch = args


async def rename_so_files(root_dir: str) -> None:
    """
    将 Darwin 系统下编译生成的 *.cpython-XXX-darwin.so 文件统一重命名为 *.so。

    Parameters
    ----------
    root_dir : str
        要遍历的根目录路径，通常为 Nuitka 编译输出的 MacOS 应用目录。

    Notes
    -----
    ## ⚙️ SKLEARN `.so` 文件命名说明

    在 macOS 系统中，通过 `pip install scikit-learn` 安装包时，会生成包含平台信息的 `.so` 文件，例如

    ```
    sklearn.tree._utils.cpython-311-darwin.so
    ```

    这并非 Nuitka 打包的问题，而是 Python 构建系统中的默认行为。
    此命名方式是为了保证 `.so` 文件能够与当前 Python 解释器版本和操作系统匹配，特别是对 Cython/C 扩展模块而言。

    ---

    ### 为何需要重命名？

    虽然该命名方式合法，但在实际使用或部署过程中可能出现以下问题：

    - 某些模块的 `__init__.py` 或动态加载逻辑中仅尝试加载 `xxx.so`，导致加载失败。
    - Nuitka 或自定义启动器在查找扩展模块时可能忽略带平台后缀的 `.so` 文件。
    - 部署到其他平台或做二次封装时，文件名一致性会影响加载成功率。

    ---

    ### 如何处理？

    Framix 在 macOS 构建完成后，会自动执行如下重命名操作

    ```bash
    # 原始文件
    sklearn.tree._utils.cpython-311-darwin.so

    # 重命名后
    sklearn.tree._utils.so
    ```

    此操作通过 `rename_so_files()` 函数实现。
    确保所有 `.cpython-XXX-darwin.so` 文件都被重命名为标准格式 `.so`，增强模块导入兼容性。

    ---

    ### 总结

    - 这是 `pip` 安装 `scikit-learn` 的正常行为。
    - 对于打包后部署、多平台适配、模型分发等场景，建议重命名。
    - Framix 已内置自动处理机制，无需手动干预。
    - 本函数适用于 macOS 平台，用于修正动态链接库的命名。
    - 会打印出重命名的成功日志。
    """
    pattern = re.compile(r"^(.*)\.cpython-\d{3}-darwin\.so$")

    for file in Path(root_dir).rglob("*.so"):
        if match := pattern.match(file.name):
            file.rename(file.with_name(new_name := f"{match.group(1)}.so"))
            compile_log(f"[bold #00D787][✓]Renamed {file.name} → {new_name}")


async def packaging() -> tuple[
    str, "Path", "Path",
    typing.Union["Path"], typing.Union[tuple["Path", "Path"]], list[str], list[str]
]:
    """
    构建 Framix 独立应用的打包编译命令与目录结构信息。

    Returns
    -------
    tuple
        返回包含以下内容：
        - ops : str
            当前操作系统（win32 或 darwin）。

        - app : Path
            应用输出目录（applications/）。

        - site_packages : Path
            虚拟环境中的 site-packages 路径。

        - target : Path
            编译产物的原始输出路径。

        - rename : tuple[Path, Path]
            编译产物重命名前后的路径元组。

        - compile_cmd : list[str]
            Nuitka 编译命令列表。

        - launch : list[str]
            启动脚本相对路径列表。

    Notes
    -----
    - 自动根据平台构建打包参数。
    - 支持图标、App Bundle、macOS 签名信息等定制选项。
    - 会清空原有的 `applications/` 目录。
    """
    if (app := Path(f"applications")).exists():
        shutil.rmtree(app)
    app.mkdir(exist_ok=True)

    venv_base_path = Path(".venv") if Path(".venv").exists() else Path("venv")

    site_packages, target, rename, launch = None, None, None, ["resources", "automation"]

    compile_cmd = [exe := sys.executable, "-m", "nuitka", "--standalone"]

    if (ops := sys.platform) == "win32":
        if (lib_path := venv_base_path / "Lib" / "site-packages").exists():
            site_packages = lib_path.resolve()

            target = app.joinpath(f"{const.DESC}.dist")
            rename = target, app.joinpath(f"{const.DESC}Engine")

        compile_cmd += [
            f"--windows-icon-from-ico=schematic/resources/icons/framix_icn_2.ico",
        ]

        launch += [f"{const.NAME}.bat"]

    elif ops == "darwin":
        if (lib_path := venv_base_path / "lib").exists():
            for sub in lib_path.iterdir():
                if sub.name.startswith("python"):
                    site_packages = (sub / "site-packages").resolve()
                elif "site-packages" in str(sub):
                    site_packages = sub.resolve()

        target = app.joinpath(f"{const.DESC}.app", f"Contents", f"MacOS")
        rename = target.parent.parent, app.joinpath(f"{const.DESC}.app")

        compile_cmd += [
            f"--macos-create-app-bundle",
            f"--macos-app-name={const.DESC}",
            f"--macos-app-version={const.VERSION}",
            f"--macos-app-icon=schematic/resources/images/macos/framix_macos_icn.png",
        ]

        launch += [f"{const.NAME}.sh"]

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
    compile_log(f"Launch={launch}")

    try:
        if writer := await asyncio.wait_for(
                Terminal.cmd_line([exe, "-m", "pip", "show", compile_cmd[2]]), timeout=5
        ):
            compile_log(f"Writer={writer}")
    except asyncio.TimeoutError as e:
        compile_log(f"Writer={compile_cmd[2]} {e}")
    else:
        compile_log(f"Writer={compile_cmd[2]}")

    return ops, app, site_packages, target, rename, compile_cmd, launch


async def post_build() -> typing.Coroutine | None:
    """
    Framix 应用打包后的自动依赖检查与部署流程。

    Returns
    -------
    Coroutine or None
        异步构建过程的最终协程对象。

    Workflow
    --------
    1. 调用 `packaging()` 构造 Nuitka 打包命令。
    2. 检查依赖项完整性，准备拷贝路径。
    3. 调用 `Terminal.cmd_link()` 执行打包命令并实时输出日志。
    4. 拷贝本地资源与依赖模块。
    5. MacOS 平台执行 so 文件重命名修正。
    6. 清除临时 build 缓存目录。
    """

    async def input_stream() -> typing.Coroutine | None:
        """
        异步读取终端标准输出流内容，并打印实时构建日志。
        """
        async for line in transports.stdout:
            compile_log(line.decode(encoding=const.CHARSET, errors="ignore").strip())

    async def error_stream() -> typing.Coroutine | None:
        """
        异步读取终端错误输出流内容，实时反馈构建异常信息。
        """
        async for line in transports.stderr:
            compile_log(line.decode(encoding=const.CHARSET, errors="ignore").strip())

    async def examine_dependencies() -> typing.Coroutine | None:
        """
        检查所有指定依赖是否存在于虚拟环境中。

        Raises
        ------
        FileNotFoundError
            如果依赖不完整或路径丢失，将触发异常。

        Notes
        -----
        - 依赖包括本地模块、本地启动脚本与第三方库。
        - 成功与失败的依赖分别记录在 `done_list` 和 `fail_list`。
        """
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
        """
        拷贝所有依赖文件与目录至编译产物路径，并执行重命名与缓存清理。

        Notes
        -----
        - 使用 Rich 的进度条展示拷贝过程。
        - 若为 macOS，还会执行 so 文件重命名。
        - 构建完成后清理 *.build 缓存目录。
        """
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

        rename_src, rename_dst = rename
        shutil.move(rename_src, rename_dst)
        compile_log(f"[bold #00D787][✓] Rename completed {rename_src.name} → {rename_dst.name}")

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

    build_start_time = time.time()

    Active.active("INFO")

    ops, apps, site_packages, target, rename, compile_cmd, launch = await packaging()

    done_list, fail_list = [], []

    schematic = apps.parent.joinpath(const.F_SCHEMATIC)
    structure = apps.parent.joinpath(const.F_STRUCTURE, const.F_SRC_MODEL_PLACE)

    local_pack, local_file = [
        (schematic, target / schematic.name),
        (structure, target.parent / const.F_STRUCTURE / structure.name)
    ], [
        (schematic.joinpath(*launch), target)
    ]

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

    transports = await Terminal.cmd_link(compile_cmd)
    await asyncio.gather(
        *(asyncio.create_task(task()) for task in [input_stream, error_stream])
    )
    await transports.wait()

    await forward_dependencies()

    compile_log(f"Time consuming={(time.time() - build_start_time) / 60:.2f} m")


if __name__ == "__main__":
    asyncio.run(post_build())
