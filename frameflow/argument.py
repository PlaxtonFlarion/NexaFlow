"""argument"""


class Args(object):

    reconcilable: list[str] = []
    discriminate: list[str] = []
    
    # 核心操控
    GROUP_MAJOR = {
        "--video": {
            "args": {"action": "append"},
            "view": ["视频文件", "多次"],
            "help": "视频解析探索"
        },
        "--stack": {
            "args": {"action": "append"},
            "view": ["视频集合", "多次"],
            "help": "影像堆叠导航"
        },
        "--train": {
            "args": {"action": "append"},
            "view": ["视频文件", "多次"],
            "help": "模型训练大师"
        },
        "--build": {
            "args": {"action": "append"},
            "view": ["分类集合", "多次"],
            "help": "模型编译大师"
        },
        "--flick": {
            "args": {"action": "store_true"},
            "view": ["布尔", "一次"],
            "help": "循环节拍器"
        },
        "--carry": {
            "args": {"action": "append"},
            "view": ["脚本名称", "多次"],
            "help": "脚本驱动者"
        },
        "--fully": {
            "args": {"action": "append"},
            "view": ["脚本路径", "多次"],
            "help": "全域执行者"
        },
        "--paint": {
            "args": {"action": "store_true"},
            "view": ["布尔", "一次"],
            "help": "线迹创造者"
        },
        "--union": {
            "args": {"action": "append"},
            "view": ["报告集合", "多次"],
            "help": "时空纽带分析系统"
        },
        "--merge": {
            "args": {"action": "append"},
            "view": ["报告集合", "多次"],
            "help": "时序融合分析系统"
        }
    }

    # 辅助利器
    GROUP_MEANS = {
        "--speed": {
            "args": {"action": "store_true"},
            "view": ["布尔", "一次"],
            "help": "光速穿梭"
        },
        "--basic": {
            "args": {"action": "store_true"},
            "view": ["布尔", "一次"],
            "help": "基石阵地"
        },
        "--keras": {
            "args": {"action": "store_true"},
            "view": ["布尔", "一次"],
            "help": "思维导航"
        }
    }

    # 视控精灵
    GROUP_SPACE = {
        "--alone": {
            "args": {"action": "store_true"},
            "view": ["布尔", "一次"],
            "help": "独立驾驭"
        },
        "--whist": {
            "args": {"action": "store_true"},
            "view": ["布尔", "一次"],
            "help": "静默守护"
        }
    }

    # 像素工坊
    GROUP_MEDIA = {
        "--alike": {
            "args": {"action": "store_true"},
            "view": ["布尔", "一次"],
            "help": "均衡节奏"
        },
        "--shine": {
            "args": {"action": "store_true"},
            "view": ["布尔", "一次"],
            "help": "平行映射"
        }
    }

    # 数据智绘
    GROUP_ARRAY = {
        "--group": {
            "args": {"action": "store_true"},
            "view": ["布尔", "一次"],
            "help": "集群视图"
        },
        "--total": {
            "args": {"nargs": "?", "const": "", "type": str},
            "view": ["路径", "一次"],
            "help": "鉴证之书"
        }
    }

    # 图帧先行
    GROUP_FIRST = {
        "--shape": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["数值", "一次"],
            "help": "尺寸定制",
            "push": lambda x: [
                f"[bold #87AFD7]{list(x.shape) if x.shape else 'Auto'}",
                f"[bold][[bold #7FFFD4] ?  ? [/]]",
                f"[bold]宽高 [bold #FFD700]{x.shape[0]} x {x.shape[1]}" if x.shape else f"[bold #A4D3EE]自动"
            ]
        },
        "--scale": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["数值", "一次"],
            "help": "变幻缩放",
            "push": lambda x: [
                f"[bold #87AFD7]{x.scale if x.scale else 'Auto'}",
                f"[bold][[bold #7FFFD4] 0  1 [/]]",
                f"[bold]压缩 [bold #FFD700]{x.scale}[/]" if x.scale else f"[bold #A4D3EE]自动"
            ]
        },
        "--start": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["数值", "一次"],
            "help": "时刻启程",
            "push": lambda x, y: [
                f"[bold #87AFD7]{getattr(y, 'parse_mills')(x.start) if x.start else 'Auto'}",
                f"[bold][[bold #7FFFD4] 0  ? [/]]",
                f"[bold]开始 [bold #FFD700]{x.start}[/]" if x.start else f"[bold #A4D3EE]自动"
            ]
        },
        "--close": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["数值", "一次"],
            "help": "时光封印",
            "push": lambda x, y: [
                f"[bold #87AFD7]{getattr(y, 'parse_mills')(x.close) if x.close else 'Auto'}",
                f"[bold][[bold #7FFFD4] 0  ? [/]]",
                f"[bold]结束 [bold #FFD700]{x.close}[/]" if x.close else f"[bold #A4D3EE]自动"
            ]
        },
        "--limit": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["数值", "一次"],
            "help": "持续历程",
            "push": lambda x, y: [
                f"[bold #87AFD7]{getattr(y, 'parse_mills')(x.limit) if x.limit else 'Auto'}",
                f"[bold][[bold #7FFFD4] 0  ? [/]]",
                f"[bold]持续 [bold #FFD700]{x.limit}[/]" if x.limit else f"[bold #A4D3EE]自动"
            ]
        },
        "--gauss": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["数值", "一次"],
            "help": "朦胧幻界",
            "push": lambda x: [
                f"[bold #87AFD7]{x.gauss if x.gauss else 'Auto'}",
                f"[bold][[bold #7FFFD4] 0  10[/]]",
                f"[bold]模糊 [bold #FFD700]{x.gauss}[/]" if x.gauss else f"[bold #A4D3EE]自动"
            ]
        },
        "--grind": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["数值", "一次"],
            "help": "边缘觉醒",
            "push": lambda x: [
                f"[bold #87AFD7]{x.grind if x.grind else 'Auto'}",
                f"[bold][[bold #7FFFD4]-2  5 [/]]",
                f"[bold]锐化 [bold #FFD700]{x.grind}[/]" if x.grind else f"[bold #A4D3EE]自动"
            ]
        },
        "--frate": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["数值", "一次"],
            "help": "频率探测",
            "push": lambda x: [
                f"[bold #87AFD7]{x.frate}",
                f"[bold][[bold #7FFFD4] 1  60[/]]",
                f"[bold]帧率 [bold #FFD700]{x.frate}[/]" if x.frate else f"[bold #A4D3EE]自动"
            ]
        }
    }

    # 智析引擎
    GROUP_EXTRA = {
        "--boost": {
            "args": {"action": "store_true"},
            "view": ["布尔", "一次"],
            "help": "加速跳跃",
            "push": lambda x: [
                f"[bold #87AFD7]{x.boost}",
                f"[bold][[bold #7FFFD4] T  F [/]]",
                f"[bold #CAFF70]开启" if x.boost else f"[bold #FFB6C1]关闭"
            ]
        },
        "--color": {
            "args": {"action": "store_true"},
            "view": ["布尔", "一次"],
            "help": "彩绘世界",
            "push": lambda x: [
                f"[bold #87AFD7]{x.color}",
                f"[bold][[bold #7FFFD4] T  F [/]]",
                f"[bold #CAFF70]开启" if x.color else f"[bold #FFB6C1]关闭"
            ]
        },
        "--begin": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["数值", "一次"],
            "help": "序章开启",
            "push": lambda x: [
                f"[bold #87AFD7]{list(x.begin)}",
                f"[bold][[bold #7FFFD4] ?  ? [/]]",
                f"[bold]非稳定阶段 [bold #FFD700]{list(x.begin)}[/]"
            ]
        },
        "--final": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["数值", "一次"],
            "help": "终章落幕",
            "push": lambda x: [
                f"[bold #87AFD7]{list(x.final)}",
                f"[bold][[bold #7FFFD4] ?  ? [/]]",
                f"[bold]非稳定阶段 [bold #FFD700]{list(x.final)}[/]"
            ]
        },
        "--thres": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["数值", "一次"],
            "help": "稳定阈值",
            "push": lambda x: [
                f"[bold #87AFD7]{x.thres}",
                f"[bold][[bold #7FFFD4] 0  1 [/]]",
                f"[bold]阈值超过 [bold #FFD700]{x.thres}[/] 的帧为稳定帧"
            ]
        },
        "--shift": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["数值", "一次"],
            "help": "偏移调整",
            "push": lambda x: [
                f"[bold #87AFD7]{x.shift}",
                f"[bold][[bold #7FFFD4] 0  10[/]]",
                f"[bold]合并 [bold #FFD700]{x.shift}[/] 个变化不大的稳定区间"
            ]
        },
        "--block": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["数值", "一次"],
            "help": "矩阵分割",
            "push": lambda x: [
                f"[bold #87AFD7]{x.block}",
                f"[bold][[bold #7FFFD4] 1  10[/]]",
                f"[bold]每个图像分割 [bold #FFD700]{x.block * x.block}[/] 块"
            ]
        },
        "--crops": {
            "args": {"action": "append"},
            "view": ["坐标", "多次"],
            "help": "视界探索",
            "push": lambda x: [
                f"[bold #87AFD7]{['!' for _ in range(len(x.crops))]}",
                f"[bold][[bold #7FFFD4] 0  1 [/]]",
                f"[bold]探索 [bold #FFD700]{len(x.crops)}[/] 个区域的图像"
            ]
        },
        "--omits": {
            "args": {"action": "append"},
            "view": ["坐标", "多次"],
            "help": "视界忽略",
            "push": lambda x: [
                f"[bold #87AFD7]{['!' for _ in range(len(x.omits))]}",
                f"[bold][[bold #7FFFD4] 0  1 [/]]",
                f"[bold]忽略 [bold #FFD700]{len(x.omits)}[/] 个区域的图像"
            ]
        }
    }

    # 漏洞追踪
    GROUP_DEBUG = {
        "--debug": {
            "args": {"action": "store_true"},
            "view": ["布尔", "一次"],
            "help": "架构透镜"
        }
    }

    # Argument
    ARGUMENT = {
        "核心操控": GROUP_MAJOR,
        "辅助利器": GROUP_MEANS,
        "视控精灵": GROUP_SPACE,
        "像素工坊": GROUP_MEDIA,
        "数据智绘": GROUP_ARRAY,
        "图帧先行": GROUP_FIRST,
        "智析引擎": GROUP_EXTRA,
        "漏洞追踪": GROUP_DEBUG
    }


class Wind(object):

    SPEED_TEXT = """> >> >>> >>>> >>>>> >>>>>> >>>>>>> >>>>>>>> >>>>>> >>>>> >>>> >>> >> >
> > > > > > > > > > > > > > > > > > > >"""

    BASIC_TEXT = """[####][####][####][####][####][####]
[##################################]"""

    KERAS_TEXT = """|> * -> * -> * -> * -> * -> * -> * -> * -> * -> * 
|> * -> * -> * -> * -> * -> * -> * -> * -> * -> *"""

    SPEED = {
        "文本": {
            "style": "bold #FFFF00",
            "justify": "center",
        },
        "边框": {
            "title": "**<* 光速穿梭 *>**",
            "title_align": "center",
            "subtitle": None,
            "subtitle_align": "center",
            "border_style": "bold #87CEFA",
        }
    }

    BASIC = {
        "文本": {
            "style": "bold #FFFF00",
            "justify": "center",
        },
        "边框": {
            "title": "**<* 基石阵地 *>**",
            "title_align": "center",
            "subtitle": None,
            "subtitle_align": "center",
            "border_style": "bold #87CEFA",
        }
    }

    KERAS = {
        "文本": {
            "style": "bold #FFFF00",
            "justify": "center",
        },
        "边框": {
            "title": "**<* 思维导航 *>**",
            "title_align": "center",
            "subtitle": None,
            "subtitle_align": "center",
            "border_style": "bold #87CEFA",
        }
    }

    STANDARD = {
        "文本": {
            "style": "bold #87CEEB",
            "justify": "center",
        },
        "边框": {
            "title": "Balance Time Standard",
            "title_align": "center",
            "subtitle": None,
            "subtitle_align": "center",
            "border_style": "bold #0077B6",
        }
    }

    TAILOR = {
        "文本": {
            "style": "bold #FFD60A",
            "justify": "left",
        },
        "边框": {
            "title": "Tailor",
            "title_align": "center",
            "subtitle": None,
            "subtitle_align": "center",
            "border_style": "bold #0077B6",
        }
    }

    FILTER = {
        "文本": {
            "style": "bold #B0C4DE",
            "justify": "left",
        },
        "边框": {
            "title": "Filter",
            "title_align": "center",
            "subtitle": None,
            "subtitle_align": "center",
            "border_style": "bold #E0E0E0",
        }
    }

    METRIC = {
        "文本": {
            "style": "bold #CFCFCF",
            "justify": "left",
        },
        "边框": {
            "title": "Metric",
            "title_align": "center",
            "subtitle": None,
            "subtitle_align": "center",
            "border_style": "bold #B0C4DE",
        }
    }

    LOADER = {
        "文本": {
            "style": "bold #FFD39B",
            "justify": "left",
        },
        "边框": {
            "title": "Loader",
            "title_align": "center",
            "subtitle": None,
            "subtitle_align": "center",
            "border_style": "bold #2BC0E4",
        }
    }

    CUTTER = {
        "文本": {
            "style": "bold #D8BFD8",
            "justify": "left",
        },
        "边框": {
            "title": "Cutter",
            "title_align": "center",
            "subtitle": None,
            "subtitle_align": "center",
            "border_style": "bold #32CD32",
        }
    }

    FASTER = {
        "文本": {
            "style": "bold #FFE7BA",
            "justify": "left",
        },
        "边框": {
            "title": "Faster",
            "title_align": "center",
            "subtitle": None,
            "subtitle_align": "center",
            "border_style": "bold #C6E2FF",
        }
    }

    PROVIDER = {
        "文本": {
            "style": "bold #FCE38A",
            "justify": "left",
        },
        "边框": {
            "title": "Provider",
            "title_align": "center",
            "subtitle": None,
            "subtitle_align": "center",
            "border_style": "bold #F38181",
        }
    }

    DESIGNER = {
        "文本": {
            "style": "bold #8360c3",
            "justify": "left",
        },
        "边框": {
            "title": "Designer",
            "title_align": "center",
            "subtitle": None,
            "subtitle_align": "center",
            "border_style": "bold #2ebf91",
        }
    }

    REPORTER = {
        "文本": {
            "style": "bold #EE7AE9",
            "justify": "left",
        },
        "边框": {
            "title": "Reporter",
            "title_align": "center",
            "subtitle": None,
            "subtitle_align": "center",
            "border_style": "bold #008B8B",
        }
    }

    KEEPER = {
        "文本": {
            "style": "bold #FFFFFF on #FF6347",
            "justify": "left",
        },
        "边框": {
            "title": "<!> 警告 <!>",
            "title_align": "center",
            "subtitle": None,
            "subtitle_align": "center",
            "border_style": "bold #FFEDBC"
        }
    }

    DRAWER = {
        "文本": {
            "style": "bold #B5B5B5",
            "justify": "left",
        },
        "边框": {
            "title": "Drawer",
            "title_align": "center",
            "subtitle": None,
            "subtitle_align": "center",
            "border_style": "bold #FFC1C1",
        }
    }

    EXPLORER = {
        "文本": {
            "style": "bold #FFE47A",
            "justify": "left",
        },
        "边框": {
            "title": "Explorer",
            "title_align": "center",
            "subtitle": None,
            "subtitle_align": "center",
            "border_style": "bold #3D7EAA",
        }
    }


if __name__ == '__main__':
    pass
