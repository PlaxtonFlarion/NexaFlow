import os
import re
import json
from loguru import logger
from rich.table import Table
from frameflow.skills.show import Show


class Deploy(object):

    _deploys = {
        "boost": False,
        "color": False,
        "shape": None,
        "scale": None,
        "model_size": (350, 700),
        "fps": 60,
        "threshold": 0.97,
        "offset": 3,
        "block": 6,
        "crops": [],
        "omits": []
    }

    def __init__(self, deploy_file: str):
        self.load_deploy(deploy_file)

    @property
    def boost(self):
        return self._deploys["boost"]

    @property
    def color(self):
        return self._deploys["color"]

    @property
    def shape(self):
        return self._deploys["shape"]

    @property
    def scale(self):
        return self._deploys["scale"]

    @property
    def model_size(self):
        return self._deploys["model_size"]

    @property
    def fps(self):
        return self._deploys["fps"]

    @property
    def threshold(self):
        return self._deploys["threshold"]

    @property
    def offset(self):
        return self._deploys["offset"]

    @property
    def block(self):
        return self._deploys["block"]

    @property
    def crops(self):
        return self._deploys["crops"]

    @property
    def omits(self):
        return self._deploys["omits"]

    @boost.setter
    def boost(self, value):
        if isinstance(value, bool):
            self._deploys["boost"] = value
        elif isinstance(value, str):
            mode = value.lower() if isinstance(value, str) else "false"
            self._deploys["boost"] = True if mode == "true" else False
        else:
            self._deploys["boost"] = False

    @color.setter
    def color(self, value):
        if isinstance(value, bool):
            self._deploys["color"] = value
        elif isinstance(value, str):
            mode = value.lower() if isinstance(value, str) else "false"
            self._deploys["color"] = True if mode == "true" else False
        else:
            self._deploys["color"] = False

    @shape.setter
    def shape(self, value):
        if isinstance(value, tuple):
            self._deploys["shape"] = value
        elif isinstance(value, str):
            # 在字符串中找到所有数字
            match_list = re.findall(r"-?\d*\.?\d+", value)
            # 将匹配到的数字转换为整数或浮点数，并形成一个元组
            if len(match_list) >= 2:
                converted = []
                for num in match_list:
                    # 尽可能将数字转换为整数，否则转换为浮点数
                    try:
                        converted_num = int(num)
                    except ValueError:
                        converted_num = float(num)
                    converted.append(converted_num)
                # 确保元组仅包含两个元素（宽度、高度）
                self._deploys["shape"] = tuple(converted[:2])

    @scale.setter
    def scale(self, value):
        if isinstance(value, int | float):
            self._deploys["scale"] = max(0.1, min(1.0, value))
        elif isinstance(value, str):
            if value.strip().upper() != "NONE":
                try:
                    scale_value = float(value)
                    self._deploys["scale"] = max(0.1, min(1.0, scale_value))
                except ValueError:
                    self._deploys["scale"] = None
                    raise ValueError("scale 的值必须是一个可以转换为浮点数的数值 ...")

    @model_size.setter
    def model_size(self, value):
        if isinstance(value, tuple):
            self._deploys["model_size"] = value
        elif isinstance(value, str):
            # 在字符串中找到所有数字
            match_list = re.findall(r"-?\d*\.?\d+", value)
            # 将匹配到的数字转换为整数或浮点数，并形成一个元组
            if len(match_list) >= 2:
                converted = []
                for num in match_list:
                    # 尽可能将数字转换为整数，否则转换为浮点数
                    try:
                        converted_num = int(num)
                    except ValueError:
                        converted_num = float(num)
                    converted.append(converted_num)
                # 确保元组仅包含两个元素（宽度、高度）
                self._deploys["model_size"] = tuple(converted[:2])

    @fps.setter
    def fps(self, value):
        try:
            fps_value = int(value)
        except ValueError:
            self._deploys["fps"] = 60
            raise ValueError("fps 的值必须是一个可以转换为整数的数值 ...")

        self._deploys["fps"] = max(15, min(60, fps_value))

    @threshold.setter
    def threshold(self, value):
        try:
            threshold_value = float(value)
        except ValueError:
            self._deploys["threshold"] = 0.97
            raise ValueError("threshold 的值必须是一个可以转换为浮点数的数值 ...")

        if isinstance(threshold_value, float):
            threshold_value = round(threshold_value, 2)

        self._deploys["threshold"] = max(0.1, min(1.0, threshold_value))

    @offset.setter
    def offset(self, value):
        try:
            offset_value = int(value)
        except ValueError:
            self._deploys["offset"] = 3
            raise ValueError("offset 的值必须是一个可以转换为整数的数值 ...")

        self._deploys["offset"] = max(1, offset_value)

    @block.setter
    def block(self, value):
        try:
            block_value = int(value)
        except ValueError:
            self._deploys["block"] = 6
            raise ValueError("block 的值必须是一个可以转换为整数的数值 ...")

        self._deploys["block"] = max(15, min(60, block_value))

    @crops.setter
    def crops(self, value):
        hooks_list, effective = value, []
        for hook in hooks_list:
            if isinstance(hook, dict):
                data_list = [value for value in hook.values() if isinstance(value, int | float)]
                if len(data_list) == 4 and sum(data_list) > 0:
                    effective.append((hook["x"], hook["y"], hook["x_size"], hook["y_size"]))
            elif isinstance(hook, tuple):
                effective.append(hook)

        self._deploys["crops"] = list(set(effective)).copy()
        effective.clear()

    @omits.setter
    def omits(self, value):
        hooks_list, effective = value, []
        for hook in hooks_list:
            if isinstance(hook, dict):
                data_list = [value for value in hook.values() if isinstance(value, int | float)]
                if len(data_list) == 4 and sum(data_list) > 0:
                    effective.append((hook["x"], hook["y"], hook["x_size"], hook["y_size"]))
            elif isinstance(hook, tuple):
                effective.append(hook)

        self._deploys["omits"] = list(set(effective)).copy()
        effective.clear()

    def dump_deploy(self, deploy_file: str) -> None:
        os.makedirs(os.path.dirname(deploy_file), exist_ok=True)

        with open(file=deploy_file, mode="w", encoding="utf-8") as f:
            f.writelines('{')
            for k, v in self._deploys.items():
                f.writelines('\n')
                if isinstance(v, bool) or v is None:
                    f.writelines(f'    "{k}": "{v}",')
                elif k == "model_size":
                    f.writelines(f'    "{k}": "{v}",')
                elif k == "crops" or k == "omits":
                    if len(v) == 0:
                        default = '{"x": 0, "y": 0, "x_size": 0, "y_size": 0}'
                        f.writelines(f'    "{k}": [\n')
                        f.writelines(f'        {default}\n')
                        f.writelines('    ],') if k == "crops" else f.writelines('    ]')
                    else:
                        f.writelines(f'    "{k}": [\n')
                        for index, i in enumerate(v):
                            x, y, x_size, y_size = i
                            new_size = f'{{"x": {x}, "y": {y}, "x_size": {x_size}, "y_size": {y_size}}}'
                            if (index + 1) == len(v):
                                f.writelines(f'        {new_size}\n')
                            else:
                                f.writelines(f'        {new_size},\n')
                        f.writelines('    ],') if k == "crops" else f.writelines('    ]')
                else:
                    f.writelines(f'    "{k}": {v},')
            f.writelines('\n}')

    def load_deploy(self, deploy_file: str) -> None:
        try:
            with open(file=deploy_file, mode="r", encoding="utf-8") as f:
                data = json.loads(f.read())
                self.boost = data.get("boost", "false")
                self.color = data.get("color", "false")
                self.shape = data.get("shape", None)
                self.scale = data.get("scale", None)
                self.model_size = data.get("model_size", (350, 700))
                self.fps = data.get("fps", 60)
                self.threshold = data.get("threshold", 0.97)
                self.offset = data.get("offset", 3)
                self.block = data.get("block", 6)
                self.crops = data.get("crops", [])
                self.omits = data.get("omits", [])
        except FileNotFoundError:
            logger.debug("未找到部署文件,使用默认参数 ...")
        except json.decoder.JSONDecodeError:
            logger.warning("部署文件解析错误,文件格式不正确,使用默认参数 ...")
        except Exception as e:
            logger.error(e)
        else:
            logger.info("读取部署文件,使用部署参数 ...")

    def view_deploy(self) -> None:

        title_color = "#af5fd7"
        col_1_color = "#d75f87"
        col_2_color = "#87afd7"
        col_3_color = "#00af5f"

        table = Table(
            title=f"[bold {title_color}]Framix Analyzer Deploy",
            header_style=f"bold {title_color}", title_justify="center",
            show_header=True
        )
        table.add_column("配置", no_wrap=True)
        table.add_column("参数", no_wrap=True, max_width=12)
        table.add_column("范围", no_wrap=True)
        table.add_column("效果", no_wrap=True)

        table.add_row(
            f"[bold {col_1_color}]跳帧模式",
            f"[bold {col_2_color}]{self.boost}",
            f"[bold][[bold {col_3_color}]T | F[/bold {col_3_color}] ]",
            f"[bold green]开启[/bold green]" if self.boost else "[bold red]关闭[/bold red]",
        )
        table.add_row(
            f"[bold {col_1_color}]彩色模式",
            f"[bold {col_2_color}]{self.color}",
            f"[bold][[bold {col_3_color}]T | F[/bold {col_3_color}] ]",
            f"[bold green]开启[/bold green]" if self.color else "[bold red]关闭[/bold red]",
        )
        table.add_row(
            f"[bold {col_1_color}]图片尺寸",
            f"[bold {col_2_color}]{self.shape}" if self.shape else f"[bold {col_2_color}]Auto",
            f"[bold][[bold {col_3_color}]? , ?[/bold {col_3_color}] ]",
            f"[bold]宽 [bold yellow]{self.shape[0]}[/bold yellow] 高 [bold yellow]{self.shape[1]}[/bold yellow]" if self.shape else f"[bold green]自动[/bold green]",
        )
        table.add_row(
            f"[bold {col_1_color}]压缩比例",
            f"[bold {col_2_color}]{self.scale}"if self.scale else f"[bold {col_2_color}]Auto",
            f"[bold][[bold {col_3_color}]0 , 1[/bold {col_3_color}] ]",
            f"[bold]压缩图片至 [bold yellow]{self.scale}[/bold yellow]" if self.scale else f"[bold green]自动[/bold green]",
        )
        table.add_row(
            f"[bold {col_1_color}]模型尺寸",
            f"[bold {col_2_color}]{self.model_size}",
            f"[bold][[bold {col_3_color}]? , ?[/bold {col_3_color}] ]",
            f"[bold]宽 [bold yellow]{self.model_size[0]}[/bold yellow] 高 [bold yellow]{self.model_size[1]}[/bold yellow]",
        )
        table.add_row(
            f"[bold {col_1_color}]帧采样率",
            f"[bold {col_2_color}]{self.fps}",
            f"[bold][[bold {col_3_color}]15, 60[/bold {col_3_color}]]",
            f"[bold]每秒 [bold yellow]{self.fps}[/bold yellow] 帧",
        )
        table.add_row(
            f"[bold {col_1_color}]相似度",
            f"[bold {col_2_color}]{self.threshold}",
            f"[bold][[bold {col_3_color}]0 , 1[/bold {col_3_color}] ]",
            f"[bold]阈值超过 [bold yellow]{self.threshold}[/bold yellow] 的帧为稳定帧",
        )
        table.add_row(
            f"[bold {col_1_color}]补偿值",
            f"[bold {col_2_color}]{self.offset}",
            f"[bold][[bold {col_3_color}]0 , ?[/bold {col_3_color}] ]",
            f"[bold]合并 [bold yellow]{self.offset}[/bold yellow] 个变化不大的稳定区间",
        )
        table.add_row(
            f"[bold {col_1_color}]切分程度",
            f"[bold {col_2_color}]{self.block}",
            f"[bold][[bold {col_3_color}]1 , 6[/bold {col_3_color}] ]",
            f"[bold]每个帧图像切分为 [bold yellow]{self.block}[/bold yellow] 块",
        )
        table.add_row(
            f"[bold {col_1_color}]获取区域",
            f"[bold {col_2_color}]{['!' for _ in range(len(self.crops))]}",
            f"[bold][[bold {col_3_color}]0 , 1[/bold {col_3_color}] ]",
            f"[bold]共 [bold yellow]{len(self.crops)}[/bold yellow] 个区域的图像参与计算",
        )
        table.add_row(
            f"[bold {col_1_color}]忽略区域",
            f"[bold {col_2_color}]{['!' for _ in range(len(self.omits))]}",
            f"[bold][[bold {col_3_color}]0 , 1[/bold {col_3_color}] ]",
            f"[bold]共 [bold yellow]{len(self.omits)}[/bold yellow] 个区域的图像不参与计算",
        )
        Show.console.print(table)


class Option(object):

    _options = {
        "Total Path": ""
    }

    def __init__(self, option_file: str):
        self.load_option(option_file)

    @property
    def total_path(self):
        return self._options["Total Path"]

    @total_path.setter
    def total_path(self, value):
        if value and os.path.isdir(value):
            if not os.path.exists(value):
                os.makedirs(value, exist_ok=True)
            self._options["Total Path"] = value

    def load_option(self, option_file: str) -> None:
        try:
            with open(file=option_file, mode="r", encoding="utf-8") as f:
                data = json.loads(f.read())
        except FileNotFoundError:
            logger.debug("未找到配置文件,使用默认路径 ...")
            self.dump_option(option_file)
        except json.decoder.JSONDecodeError:
            logger.debug("配置文件解析错误,文件格式不正确,使用默认路径 ...")
        else:
            logger.debug("读取配置文件,使用配置参数 ...")
            self.total_path = data.get("Total Path", "")

    def dump_option(self, option_file: str) -> None:
        os.makedirs(os.path.dirname(option_file), exist_ok=True)

        with open(file=option_file, mode="w", encoding="utf-8") as f:
            f.writelines('{')
            for k, v in self._options.items():
                f.writelines('\n')
                f.writelines(f'    "{k}": "{v}"')
            f.writelines('\n}')


if __name__ == '__main__':
    pass
