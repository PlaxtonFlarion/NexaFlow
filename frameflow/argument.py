"""Argument"""

# 主要命令
GROUP_MAJOR = {
    "--video": {
        "args": {"action": "append"},
        "view": ["视频文件", "多次"],
        "help": "分析视频文件"
    },
    "--stack": {
        "args": {"action": "append"},
        "view": ["视频集合", "多次"],
        "help": "分析视频集合"
    },
    "--train": {
        "args": {"action": "append"},
        "view": ["视频文件", "多次"],
        "help": "训练模型"
    },
    "--build": {
        "args": {"action": "append"},
        "view": ["图片集合", "多次"],
        "help": "编译模型"
    },
    "--flick": {
        "args": {"action": "store_true"},
        "view": ["命令参数", "一次"],
        "help": "循环模式"
    },
    "--carry": {
        "args": {"action": "append"},
        "view": ["脚本名称", "多次"],
        "help": "运行指定脚本"
    },
    "--fully": {
        "args": {"action": "append"},
        "view": ["文件路径", "多次"],
        "help": "运行全部脚本"
    },
    "--paint": {
        "args": {"action": "store_true"},
        "view": ["命令参数", "一次"],
        "help": "绘制分割线条"
    },
    "--union": {
        "args": {"action": "append"},
        "view": ["报告集合", "多次"],
        "help": "聚合视频帧报告"
    },
    "--merge": {
        "args": {"action": "append"},
        "view": ["报告集合", "多次"],
        "help": "聚合时间戳报告"
    }
}

# 附加命令
GROUP_MEANS = {
    "--speed": {
        "args": {"action": "store_true"},
        "view": ["布尔", "一次"],
        "help": "快速模式"
    },
    "--basic": {
        "args": {"action": "store_true"},
        "view": ["布尔", "一次"],
        "help": "基础模式"
    },
    "--keras": {
        "args": {"action": "store_true"},
        "view": ["布尔", "一次"],
        "help": "智能模式"
    }
}

# 视频控制
GROUP_SPACE = {
    "--alone": {
        "args": {"action": "store_true"},
        "view": ["布尔", "一次"],
        "help": "独立控制"
    },
    "--whist": {
        "args": {"action": "store_true"},
        "view": ["布尔", "一次"],
        "help": "静默录制"
    }
}

# 视频配置
GROUP_MEDIA = {
    "--alike": {
        "args": {"action": "store_true"},
        "view": ["布尔", "一次"],
        "help": "平衡时间"
    }
}

# 报告配置
GROUP_ARRAY = {
    "--group": {
        "args": {"action": "store_true"},
        "view": ["布尔", "一次"],
        "help": "分组报告"
    }
}

# 分析配置
GROUP_EXTRA = {
    "--boost": {
        "args": {"action": "store_true"},
        "view": ["布尔", "一次"],
        "help": "跳帧模式"
    },
    "--color": {
        "args": {"action": "store_true"},
        "view": ["布尔", "一次"],
        "help": "彩色模式"
    },
    "--shape": {
        "args": {"nargs": "?", "const": None, "type": str},
        "view": ["数值", "一次"],
        "help": "图片尺寸"
    },
    "--scale": {
        "args": {"nargs": "?", "const": None, "type": str},
        "view": ["数值", "一次"],
        "help": "缩放比例"
    },
    "--start": {
        "args": {"nargs": "?", "const": None, "type": str},
        "view": ["数值", "一次"],
        "help": "开始时间"
    },
    "--close": {
        "args": {"nargs": "?", "const": None, "type": str},
        "view": ["数值", "一次"],
        "help": "结束时间"
    },
    "--limit": {
        "args": {"nargs": "?", "const": None, "type": str},
        "view": ["数值", "一次"],
        "help": "持续时间"
    },
    "--begin": {
        "args": {"nargs": "?", "const": None, "type": str},
        "view": ["数值", "一次"],
        "help": "开始阶段"
    },
    "--final": {
        "args": {"nargs": "?", "const": None, "type": str},
        "view": ["数值", "一次"],
        "help": "结束阶段"
    },
    "--frate": {
        "args": {"nargs": "?", "const": None, "type": str},
        "view": ["数值", "一次"],
        "help": "采样率"
    },
    "--thres": {
        "args": {"nargs": "?", "const": None, "type": str},
        "view": ["数值", "一次"],
        "help": "相似度"
    },
    "--shift": {
        "args": {"nargs": "?", "const": None, "type": str},
        "view": ["数值", "一次"],
        "help": "补偿值"
    },
    "--block": {
        "args": {"nargs": "?", "const": None, "type": str},
        "view": ["数值", "一次"],
        "help": "立方体"
    },
    "--crops": {
        "args": {"action": "append"},
        "view": ["坐标", "多次"],
        "help": "获取区域"
    },
    "--omits": {
        "args": {"action": "append"},
        "view": ["坐标", "多次"],
        "help": "忽略区域"
    }
}

# 调试配置
GROUP_DEBUG = {
    "--debug": {
        "args": {"action": "store_true"},
        "view": ["布尔", "一次"],
        "help": "调试模式"
    }
}

# Argument
ARGUMENT = {
    "主要命令": GROUP_MAJOR,
    "附加命令": GROUP_MEANS,
    "视频控制": GROUP_SPACE,
    "视频配置": GROUP_MEDIA,
    "报告配置": GROUP_ARRAY,
    "分析配置": GROUP_EXTRA,
    "调试配置": GROUP_DEBUG
}


if __name__ == '__main__':
    pass
