#   ____        _ _     _
#  | __ ) _   _(_) | __| |
#  |  _ \| | | | | |/ _` |
#  | |_) | |_| | | | (_| |
#  |____/ \__,_|_|_|\__,_|
#

# ==== Notes: 版权申明 ====
# 版权所有 (c) 2024  Framix(画帧秀)
# 此文件受 Framix(画帧秀) 许可证的保护。您可以在 LICENSE.md 文件中查看详细的许可条款。

# ==== Notes: License ====
# Copyright (c) 2024  Framix(画帧秀)
# This file is licensed under the Framix(画帧秀) License. See the LICENSE.md file for more details.

# ==== Notes: ライセンス ====
# Copyright (c) 2024  Framix(画帧秀)
# このファイルは Framix(画帧秀) ライセンスの下でライセンスされています。詳細は LICENSE.md ファイルを参照してください。

import re
import sys
import time
import shutil
import typing
import asyncio
import plistlib
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

try:
    import nuitka
    nuitka_version = "2.7"
except ImportError:
    raise FramixError(f"Use Nuitka {nuitka_version} for stable builds")

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


async def check_architecture(ops: str) -> None:
    """
    仅在 Windows 下检测 Python 是否为 64 位。
    """
    if ops != "win32":
        return None

    is_64bit = sys.maxsize > 2**32
    python_version = sys.version.split()[0]

    compile_log(f"<Python> {python_version} ({'64-bit' if is_64bit else '32-bit'})")

    if is_64bit:
        return compile_log(f"✅ 当前 Python 是 64 位，符合 {const.DESC} 打包要求。")

    raise FramixError(f"❌ 当前为 32 位 Python，建议更换为 64 位版本。")


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


async def find_vcvars64() -> str:
    """
    查找 vcvars64.bat 的完整路径，用于配置 MSVC 构建环境。
    """
    vswhere = Path(r"C:\Program Files (x86)", "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not vswhere.exists():
        raise FramixError("未找到 vswhere.exe -> 请安装 Visual Studio Build Tools")

    cmd = [
        str(vswhere), "-latest", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "-products", "*", "-property", "installationPath"
    ]

    find_result = await Terminal.cmd_line(cmd)

    vcvars = Path(find_result.strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"

    if not vcvars.exists():
        raise FramixError(f"找不到 vcvars64.bat -> {vcvars}")

    return str(vcvars)


async def find_dumpbin() -> str:
    """
    查找最新 MSVC 工具链中的 dumpbin.exe 路径。
    """
    vswhere = Path(r"C:\Program Files (x86)", "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not vswhere.exists():
        raise FramixError("未找到 vswhere.exe -> 请安装 Visual Studio Build Tools")

    cmd = [
        str(vswhere), "-latest", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "-products", "*", "-property", "installationPath"
    ]

    find_result = await Terminal.cmd_line(cmd)

    if not (tools_dir := Path(find_result.strip()) / "VC" / "Tools" / "MSVC").exists():
        raise FramixError("找不到 MSVC 工具目录 -> VC/Tools/MSVC")

    if not (version_dirs := [d for d in tools_dir.iterdir() if d.is_dir()]):
        raise FramixError("未检测到任何 VC 工具版本目录")

    if not (dumpbin := sorted(version_dirs)[-1] / "bin" / "Hostx64" / "x64" / "dumpbin.exe").exists():
        raise FramixError(f"找不到 dumpbin.exe -> {dumpbin}")

    return str(dumpbin)


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


async def edit_plist_fields(ops: str, app: str, updates: dict[str, str]) -> typing.Coroutine | None:
    """
    编辑 macOS 应用的 Info.plist 文件字段。
    """
    if ops != "darwin":
        return None

    if not (plist := Path(app) / "Contents" / "Info.plist").exists():
        raise FramixError(f"未找到 Info.plist 文件: {plist}")

    # 读取原始 plist
    with plist.open("rb") as f:
        plist_data = plistlib.load(f)

    # 显示原始值
    compile_log(f"正在修改 Info.plist: {plist}")
    for key, new_value in updates.items():
        old_value = plist_data.get(key, "<未定义>")
        compile_log(f" - {key}: {old_value}  ->  {new_value}")
        plist_data[key] = new_value

    # 写入修改后的内容
    with plist.open("wb") as f:
        plistlib.dump(plist_data, f)

    compile_log("修改完成，已写入 Info.plist。")


async def packaging() -> tuple[
    str, "Path", "Path", typing.Union["Path"],
    typing.Union[tuple["Path", "Path"]], list[str], tuple["Path", "Path"], list, str
]:
    """
    构建独立应用的打包编译命令与目录结构信息。
    """
    if (app := Path(f"applications")).exists():
        shutil.rmtree(app)
    app.mkdir(exist_ok=True)

    site_packages = await find_site_packages()

    launch = app.parent / const.F_SCHEMATIC / "resources" / "automation"

    compile_cmd = [exe := sys.executable, "-m", "nuitka"]

    if (ops := sys.platform) == "win32":
        await check_architecture(ops)
        _, dumpbin = await asyncio.gather(find_vcvars64(), find_dumpbin())

        target = app / f"{const.DESC}.dist"
        rename = target, app / f"{const.DESC}Engine"

        compile_cmd += [
            f"--mode=standalone",
            f"--windows-icon-from-ico=schematic/resources/icons/framix_icn_2.ico",
        ]

        launch = launch / f"{const.NAME}.bat", target.parent
        binary_file = rename[1] / f"{const.NAME}.exe"
        arch_info = [dumpbin, "/headers", f"{str(Path(__file__).parent / binary_file)}"]

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
        binary_file = target / f"{const.NAME}"
        arch_info = ["file", f"{str(Path(__file__).parent / binary_file)}"]

        support = "MacOS"

    else:
        raise FramixError(f"Unsupported platforms {ops}")

    compile_cmd += [
        f"--nofollow-import-to=keras,tensorflow,tensorboard,uiautomator2",
        f"--include-module=pdb,deprecation",
        f"--include-package=pygments",
        f"--include-package=absl,gast,flatbuffers,opt_einsum,h5py,ml_dtypes,wrapt",
        f"--include-package=astunparse,distutils,google,site,termcolor",
        f"--include-package=adbutils",
        f"--assume-yes-for-downloads", f"--show-progress", f"--show-memory",
        f"--output-dir={app}", f"{const.NAME}.py"
    ]

    compile_log(f"system={ops}")
    compile_log(f"folder={app}")
    compile_log(f"packet={site_packages}")
    compile_log(f"target={target}")
    compile_log(f"rename={rename}")
    compile_log(f"launch={launch}")

    writer = await Terminal.cmd_line([exe, "-m", "pip", "show", compile_cmd[2]])
    if ver := re.search(r"(?<=Version:\s).*", writer):
        if ver.group().strip() != nuitka_version:
            raise FramixError(f"Use Nuitka {nuitka_version} for stable builds")
        compile_log(f"writer={writer}")

    return ops, app, site_packages, target, rename, compile_cmd, launch, arch_info, support


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

        for folder in [const.F_SRC_OPERA_PLACE, const.F_SRC_MODEL_PLACE, const.F_SRC_TOTAL_PLACE]:
            if not (child := target.parent / const.F_STRUCTURE / folder).exists():
                await asyncio.to_thread(child.mkdir, parents=True, exist_ok=True)

        if Path(arch_info[-1]).exists():
            compile_log(await Terminal.cmd_line(arch_info))

        await edit_plist_fields(ops, rename[-1], {"CFBundleExecutable": launch[0].name})

    # ==== Note: Start from here ====
    build_start_time = time.time()

    Active.active("INFO")

    await Design.show_quantum_intro()

    compiles = await packaging()
    ops, app, site_packages, target, rename, *_ = compiles
    *_, compile_cmd, launch, arch_info, support = compiles

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
