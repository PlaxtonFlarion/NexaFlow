import re
import json
from loguru import logger
from rich.table import Table
from rich.console import Console


class Deploy(object):

    _initial = {
        "target_size": (350, 700),
        "fps": 60,
        "step": 1,
        "block": 6,
        "threshold": 0.97,
        "offset": 3,
        "compress_rate": 0.5,
        "window_size": 1,
        "window_coefficient": 2
    }

    def __init__(
            self,
            target_size: tuple = (350, 700),
            fps: int = 60,
            step: int = 1,
            block: int = 6,
            threshold: int | float = 0.97,
            offset: int = 3,
            compress_rate: int | float = 0.5,
            window_size: int = 1,
            window_coefficient: int = 2
    ):

        self._initial["target_size"] = target_size
        self._initial["fps"] = fps
        self._initial["step"] = step
        self._initial["block"] = block
        self._initial["threshold"] = threshold
        self._initial["offset"] = offset
        self._initial["compress_rate"] = compress_rate
        self._initial["window_size"] = window_size
        self._initial["window_coefficient"] = window_coefficient

    @property
    def target_size(self):
        return self._initial["target_size"]

    @property
    def fps(self):
        return self._initial["fps"]

    @property
    def step(self):
        return self._initial["step"]

    @property
    def block(self):
        return self._initial["block"]

    @property
    def threshold(self):
        return self._initial["threshold"]

    @property
    def offset(self):
        return self._initial["offset"]

    @property
    def compress_rate(self):
        return self._initial["compress_rate"]

    @property
    def window_size(self):
        return self._initial["window_size"]

    @property
    def window_coefficient(self):
        return self._initial["window_coefficient"]

    def load_deploy(self, deploy_file: str) -> bool:
        is_load: bool = False
        try:
            with open(file=deploy_file, mode="r", encoding="utf-8") as f:
                data = json.loads(f.read())
                size = data.get("target_size", (350, 700))
                self._initial["target_size"] = tuple(
                    int(i) for i in re.findall(r"-?\d*\.?\d+", size)
                ) if isinstance(size, str) else size
                self._initial["fps"] = data.get("fps", 60)
                self._initial["step"] = data.get("step", 1)
                self._initial["block"] = data.get("block", 6)
                self._initial["threshold"] = data.get("threshold", 0.97)
                self._initial["offset"] = data.get("offset", 3)
                self._initial["compress_rate"] = data.get("compress_rate", 0.5)
                self._initial["window_size"] = data.get("window_size", 1)
                self._initial["window_coefficient"] = data.get("window_coefficient", 2)
        except FileNotFoundError:
            logger.debug("未找到部署文件,使用默认参数 ...")
        except json.decoder.JSONDecodeError:
            logger.debug("部署文件解析错误,文件格式不正确,使用默认参数 ...")
        else:
            logger.debug("读取部署文件,使用部署参数 ...")
            is_load = True
        finally:
            return is_load

    def dump_deploy(self, deploy_file: str) -> None:
        with open(file=deploy_file, mode="w", encoding="utf-8") as f:
            f.writelines('{')
            for k, v in self._initial.items():
                f.writelines('\n')
                if k == "target_size":
                    f.writelines(f'    "{k}": "{v}",')
                elif k == "window_coefficient":
                    f.writelines(f'    "{k}": {v}')
                else:
                    f.writelines(f'    "{k}": {v},')
            f.writelines('\n' + '}')

    def view_deploy(self, logo: bool = True) -> None:
        console = Console()

        title_color = "[bold #FFD7FF]"
        col_1_color = "[bold #FFFFAF]"
        col_2_color = "[bold #AFFFFF]"
        col_3_color = "[bold #FFAFAF]"

        table = Table(
            title=f"{title_color}Framix Analyzer Deploy",
            header_style=f"bold #FFD7FF", title_justify="center",
            show_header=True
        )
        table.add_column("Parametric")
        table.add_column("Value")
        table.add_column("Explanation")

        row1, row2 = [k for k in self._initial.keys()], [v for v in self._initial.values()]

        table.add_row(
            f"{col_1_color}{row1[0]}", f"{col_2_color}{row2[0]}",
            f"{col_3_color}",
        )
        table.add_row(
            f"{col_1_color}{row1[1]}", f"{col_2_color}{row2[1]}",
            f"{col_3_color}",
        )
        table.add_row(
            f"{col_1_color}{row1[2]}", f"{col_2_color}{row2[2]}",
            f"{col_3_color}",
        )
        table.add_row(
            f"{col_1_color}{row1[3]}", f"{col_2_color}{row2[3]}",
            f"{col_3_color}",
        )
        table.add_row(
            f"{col_1_color}{row1[4]}", f"{col_2_color}{row2[4]}",
            f"{col_3_color}",
        )
        table.add_row(
            f"{col_1_color}{row1[5]}", f"{col_2_color}{row2[5]}",
            f"{col_3_color}",
        )
        table.add_row(
            f"{col_1_color}{row1[6]}", f"{col_2_color}{row2[6]}",
            f"{col_3_color}",
        )
        table.add_row(
            f"{col_1_color}{row1[7]}", f"{col_2_color}{row2[7]}",
            f"{col_3_color}",
        )
        table.add_row(
            f"{col_1_color}{row1[8]}", f"{col_2_color}{row2[8]}",
            f"{col_3_color}",
        )

        framix_logo = """
                ███████╗██████╗  █████╗  ███╗   ███╗██╗██╗  ██╗
                ██╔════╝██╔══██╗██╔══██╗ ████╗ ████║██║╚██╗██╔╝
                █████╗  ██████╔╝███████║ ██╔████╔██║██║ ╚███╔╝ 
                ██╔══╝  ██╔══██╗██╔══██║ ██║╚██╔╝██║██║ ██╔██╗ 
                ██║     ██║  ██║██║  ██║ ██║ ╚═╝ ██║██║██╔╝ ██╗
                ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═╝     ╚═╝╚═╝╚═╝  ╚═╝
        """
        if logo:
            console.print(framix_logo)
        console.print(table)


if __name__ == '__main__':
    pass
