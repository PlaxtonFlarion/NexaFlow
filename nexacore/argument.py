#      _                                         _
#     / \   _ __ __ _ _   _ _ __ ___   ___ _ __ | |_
#    / _ \ | '__/ _` | | | | '_ ` _ \ / _ \ '_ \| __|
#   / ___ \| | | (_| | |_| | | | | | |  __/ | | | |_
#  /_/   \_\_|  \__, |\__,_|_| |_| |_|\___|_| |_|\__|
#               |___/
#
# ==== Notes: License ====
# Copyright (c) 2024  Framix :: 画帧秀
# This file is licensed under the Framix :: 画帧秀 License. See the LICENSE.md file for more details.


class Args(object):
    """Args Style"""

    discriminate: list[str] = [
        "核心操控", "辅助利器", "视控精灵"
    ]
    reconcilable: list[str] = [
        "显示布局", "时序调控", "像素工坊", "数据智绘", "图帧先行", "智析引擎", "漏洞追踪"
    ]

    # 核心操控
    GROUP_MAJOR = {
        "--video": {
            "args": {"action": "append"},
            "view": ["多次", " /path/to/file"],
            "help": "视频解析探索",
            "func": """- 启用对单个或批量视频文件的多维度内容解析，涵盖结构化帧处理、视觉特征提取与音频质量扫描等功能模块。
                - 支持自动识别视频分段、分析关键事件并提取元信息，为后续的机器学习、内容审核或质量控制提供基础数据支持。
                - 该命令适用于构建高性能视频分析流程，助力智能化内容识别与全流程可视化评估。"""
        },
        "--stack": {
            "args": {"action": "append"},
            "view": ["多次", " /path/to/file_data/"],
            "help": "影像堆叠导航",
            "func": """- 用于组织和分析由多个视频构成的序列或集合，支持统一加载、同步处理以及帧级别的比对和汇总分析。
                - 适用于批量视频管理与结构化解构场景，尤其针对多源拍摄、教学课程章节、系列剧集等提供稳定高效的处理方案。
                - 该功能旨在提升内容一致性检验、格式统一性评估和跨片段逻辑链路的构建效率。"""
        },
        "--train": {
            "args": {"action": "append"},
            "view": ["多次", " /path/to/file"],
            "help": "模型训练大师",
            "func": """- 设计用于对视频数据进行深入分析和处理，准备数据以符合机器学习模型的训练需求。
                - 这包括特征提取、数据归一化和训练测试集的划分等关键步骤，确保输入数据能够有效地支持后续的训练过程。"""
        },
        "--build": {
            "args": {"action": "append"},
            "view": ["多次", " /path/to/file_data/"],
            "help": "模型编译大师",
            "func": """- 提供从准备好的数据集到机器学习模型文件的转换服务。
                - 支持定义模型结构、初始化模型参数并进行编译优化，确保生成的模型既能满足性能要求又具备生产级的可用性。"""
        },
        "--flick": {
            "args": {"action": "store_true"},
            "view": ["一次", " "],
            "help": "循环节拍器",
            "func": """- 在设定的时间周期内自动进行多轮屏幕录制，并对捕获的视频帧进行持续性分析。
                - 支持集成多种智能分析模型，包括但不限于动作识别、内容筛查和情绪解析，适用于长时间监控、行为检测和交互行为建模等复杂场景。"""
        },
        "--carry": {
            "args": {"action": "append"},
            "view": ["多次", " /path/to/file,KEY"],
            "help": "脚本驱动者",
            "func": """- 加载并执行指定的单个自动化脚本文件，支持多次传递以覆盖多任务执行场景。
                - 每个脚本可通过`key`分隔形式传参，适用于精细化控制单元测试或特定环境配置下的执行逻辑。"""
        },
        "--fully": {
            "args": {"action": "append"},
            "view": ["多次", " /path/to/file"],
            "help": "全域执行者",
            "func": """- 加载一个包含多个自动化脚本的脚本集合并依序执行所有任务，支持并发或序列化控制策略。
                - 常用于批量任务执行、回归测试与全流程验证，适合需要高度可扩展与可配置执行机制的系统部署场景。"""
        },
        "--paint": {
            "args": {"action": "store_true"},
            "view": ["一次", " "],
            "help": "线迹创造者",
            "func": """- 在所有设备捕获的屏幕截图上自动绘制横向和竖向的线条。
                - 这一过程可以帮助忽略或专注于图像的特定区域，提高分析的精确度和效率。"""
        },
        "--union": {
            "args": {"action": "append"},
            "view": ["多次", " /path/to/file_data/"],
            "help": "时空纽带分析系统",
            "func": """- 对无时间戳的视频帧数据进行空间逻辑聚合，构建统一的分析视图以提升帧级结构的一致性。
                - 适用于来源不明或无序采样的帧数据整合，有助于支持后续的聚类、标注与建模等流程。"""
        },
        "--merge": {
            "args": {"action": "append"},
            "view": ["多次", " /path/to/file_data/"],
            "help": "时序融合分析系统",
            "func": """- 对附带时间戳的视频帧报告进行时序对齐与结构整合，实现多源数据的时间轴同步与动态重构。
                - 适用于需要高精度时间分析的场景，如事件回溯、交叉验证或多模态融合。"""
        },
        "--talks": {
            "args": {"action": "store_true"},
            "view": ["一次", " "],
            "help": "对话协调器",
            "func": """- 读取配置源，动态注入或修改参数后重新回写并解析，形成统一配置视图。
                - 适合复杂自动化场景中对任务参数进行多轮协商、联动触发和格式化展示，构建灵活的配置执行闭环。
                - 可用于快速调试、远程参数对话、配置集成与策略推演等复杂流程。"""
        },
        "--rings": {
            "args": {"action": "store_true"},
            "view": ["一次", " "],
            "help": "回环注入器",
            "func": """- 预设运行所需的关键配置参数，包括报告路径、模型路径、彩色与灰度模型名称等。
                - 用于在主流程执行前注入标准化运行环境，确保任务执行上下文一致且可复现。
                - 与其他功能配合使用时，可作为基础参数注入器，完成运行逻辑与资源配置的解耦。"""
        },
        "--apply": {
            "args": {"type": str},
            "view": ["一次", " ACTIVATION CODE"],
            "help": "序列通行证",
            "func": """- 使用激活码向远程授权中心发起请求，获取签名后的授权数据。
                - 授权数据将绑定当前设备指纹，并以 LIC 文件形式存储在本地。
                - 一旦激活成功，系统将进入已授权状态，具备完全运行权限。
                - 适用于首次部署、跨环境迁移、临时激活等场景。"""
        }
    }

    # 辅助利器
    GROUP_MEANS = {
        "--speed": {
            "args": {"action": "store_true"},
            "view": ["一次", " "],
            "help": "光速穿梭",
            "func": """- 快速执行视频帧的结构化解构，绕过时间戳计算步骤，直接生成帧级报告，大幅提升处理效率。
                - 适用于对时序精度要求不高但对性能敏感的场景，如预处理任务、粗筛分析或快速回溯。"""
        },
        "--basic": {
            "args": {"action": "store_true"},
            "view": ["一次", " "],
            "help": "基石阵地",
            "func": """- 提供视频拆帧与精确时间戳注入的集成化流程，构建结构化帧级数据基础。
                - 该模式适用于需要时间对齐的动态场景分析，有助于后续事件定位、节奏识别及时序建模等任务。"""
        },
        "--keras": {
            "args": {"action": "store_true"},
            "view": ["一次", " "],
            "help": "思维导航",
            "func": """- 基于深度学习模型对视频帧进行智能解构分析，融合帧级拆解与时序识别。
                - 自动识别关键帧区间（起始与终止帧），适用于内容聚焦、事件检测与智能视频摘要等场景。
                - 支持时间戳注入与特征感知，显著提升视频分析的自动化水平与语义理解能力。"""
        },
        "--infer": {
            "args": {"action": "store_true"},
            "view": ["一次", " "],
            "help": "感知链接",
            "func": """- 启用远程推理模式，连接云端 GPU 容器，将视频帧数据发送至在线模型进行推理计算。
                - 该模式适用于需要更强算力或云端模型的场景。"""
        }
    }

    # 视控精灵
    GROUP_SPACE = {
        "--alone": {
            "args": {"action": "store_true"},
            "view": ["一次", " "],
            "help": "独立驾驭",
            "func": """- 在多任务并行录制环境中，独立控制指定设备的录屏终止流程。
                - 支持多设备并发时的定向控制，有效提升设备管理的粒度与任务调度的灵活性。"""
        },
        "--whist": {
            "args": {"action": "store_true"},
            "view": ["一次", " "],
            "help": "静默守护",
            "func": """- 启动无提示录制模式，确保录制过程在用户界面完全静默，屏蔽所有系统投屏与录屏提示。
                - 适用于数据采集、隐私保护或干扰最小化场景，保障录制过程的隐蔽性与完整性。"""
        }
    }

    # 显示布局
    GROUP_ORDER = {
        "--alter": {
            "args": {"default": 0, "type": int},
            "view": ["一次", "=POSITIVE INT"],
            "help": "映界之门",
            "func": """- 指定切换至目标显示器，灵活迁移录制或投屏窗口，适配多屏环境操作。
                - 通过明确的显示器编号，控制录制/投屏在不同显示器间的布局，实现多场景、多任务自由切换。"""
        }
    }

    # 时序调控
    GROUP_TIMER = {
        "--delay": {
            "args": {"default": 0, "type": int},
            "view": ["一次", "=POSITIVE INT"],
            "help": "凝滞核心",
            "func": """- 启动 **凝滞核心机制**，在执行主流程前注入指定的延迟时长（单位：秒）。
                - 用于模拟冷启动等待、手动干预窗口、环境初始化缓冲等场景。
                - 保证操作节奏一致性，为后续任务铺垫稳定执行时机。"""
        }
    }

    # 像素工坊
    GROUP_MEDIA = {
        "--alike": {
            "args": {"action": "store_true"},
            "view": ["一次", " "],
            "help": "均衡节奏",
            "func": """- 通过统一化视频长度来提升内容的一致性和观看体验。
                - 自动裁剪每一组视频至最短录制时间，确保所有视频段落的时间长度一致，从而在多视频项目中实现视觉和时间的均衡。"""
        },
        "--shine": {
            "args": {"action": "store_true"},
            "view": ["一次", " "],
            "help": "平行映射",
            "func": """- 允许用户在多个视频分析任务之间创建并行处理通道，从而提高视频处理效率。
                - 通过开启该功能，系统可以同时对多个视频进行并行分析，在减少总处理时间的同时，确保分析结果的一致性和同步性。
                - 该功能特别适合处理大规模的视频数据集或需要实时分析的视频流。"""
        }
    }

    # 数据智绘
    GROUP_ARRAY = {
        "--group": {
            "args": {"action": "store_true"},
            "view": ["一次", " "],
            "help": "集群视图",
            "func": """- 针对多设备协同执行的任务场景，为每个设备单独生成详尽的分析报告，实现设备级的数据隔离与可追溯性。
                - 此功能适用于集群化部署、远程设备控制与分布式数据采集场景，有助于定位性能瓶颈、数据偏差与系统异常，提升系统的可观测性与可维护性。
                - 支持动态扩展设备并行度，确保在异构设备或高并发分析中保持准确性与稳定性。"""
        },
        "--total": {
            "args": {"nargs": "?", "const": "", "type": str},
            "view": ["一次", " /path/to/file_data"],
            "help": "鉴证之书",
            "func": """- 用于指定最终分析报告的输出路径，系统将在该位置生成包含各类分析结果、图表与统计信息的综合性报告文件。
                - 此功能便于统一存档与后期审阅，可结合自动化流水线，将分析结果同步至远程存储、报告系统或质量评估平台，提升流程集成度与可追踪性。"""
        }
    }

    # 图帧先行
    GROUP_FIRST = {
        "--shape": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["一次", "=WIDTH,HEIGHT"],
            "help": "尺寸定制",
            "func": """- 支持自定义输出视频的尺寸（宽度 × 高度），以适应不同分发平台或终端设备的分辨率要求。
                - 在确保视觉质量的前提下，有效优化处理效率与传输带宽，占用更少存储空间。
                - 适用于批量压缩、格式转换或模型推理场景中的尺寸归一化处理。""",
            "push": lambda x, y: [
                f"[bold #87AFD7]{x.shape or 'Auto'}",
                f"[bold][[bold #7FFFD4]   ?  ?   [/]]",
                f"[bold]宽高 [bold #FFD700]{x.shape[0]} x {x.shape[1]}" if x.shape else f"[bold #A4D3EE]自动"
            ]
        },
        "--scale": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["一次", "=FLOAT"],
            "help": "变幻缩放",
            "func": """- 支持对视频帧进行等比或非等比缩放处理，以调整整体画面比例或适配特定视觉需求。
                - 可用于降低分辨率以提高处理速度，或放大图像以增强细节观察。
                - 缩放算法可与后续图像分析或模型推理任务协同优化，提高整体流程效率。
                - 在保持画面清晰度与比例合理性的基础上，有效控制资源消耗和输出体积。""",
            "push": lambda x, y: [
                f"[bold #87AFD7]{x.scale or 'Auto'}",
                f"[bold][[bold #7FFFD4] 0.1  1.0 [/]]",
                f"[bold]压缩 [bold #FFD700]{x.scale}[/]" if x.scale else f"[bold #A4D3EE]自动"
            ]
        },
        "--start": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["一次", "=INT | FLOAT | 00:00:00"],
            "help": "时刻启程",
            "func": """- 指定视频片段的起始时间点，仅分析或处理该时间点之后的内容。
                - 可用于跳过片头、定位关键时间段，适配大文件裁剪或任务加速。""",
            "push": lambda x, y: [
                f"[bold #87AFD7]{getattr(y, 'parse_mills')(x.start) or 'Auto'}",
                f"[bold][[bold #7FFFD4]   0  ?   [/]]",
                f"[bold]开始 [bold #FFD700]{x.start}[/]" if x.start else f"[bold #A4D3EE]自动"
            ]
        },
        "--close": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["一次", "=INT | FLOAT | 00:00:00"],
            "help": "时光封印",
            "func": """- 设置视频片段的终止时间点，仅处理该时间点之前的内容。
                - 常用于跳过片尾、控制分析范围或预留系统资源。""",
            "push": lambda x, y: [
                f"[bold #87AFD7]{getattr(y, 'parse_mills')(x.close) or 'Auto'}",
                f"[bold][[bold #7FFFD4]   0  ?   [/]]",
                f"[bold]结束 [bold #FFD700]{x.close}[/]" if x.close else f"[bold #A4D3EE]自动"
            ]
        },
        "--limit": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["一次", "=INT | FLOAT | 00:00:00"],
            "help": "持续历程",
            "func": """- 指定从起始时间起要分析的持续时长，自动截取对应长度的视频段落。
                - 与 `--start` 搭配使用时，可精准控制处理窗口，避免冗余计算。""",
            "push": lambda x, y: [
                f"[bold #87AFD7]{getattr(y, 'parse_mills')(x.limit) or 'Auto'}",
                f"[bold][[bold #7FFFD4]   0  ?   [/]]",
                f"[bold]持续 [bold #FFD700]{x.limit}[/]" if x.limit else f"[bold #A4D3EE]自动"
            ]
        },
        "--gauss": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["一次", "=POSITIVE INT"],
            "help": "朦胧幻界",
            "func": """- 使用高斯模糊算法对图像进行卷积平滑处理，有效削弱图像噪声与边缘锐度。
                - 常用于背景虚化、艺术滤镜、前处理降噪等场景，营造柔和梦幻的视觉氛围。""",
            "push": lambda x, y: [
                f"[bold #87AFD7]{x.gauss or 'Auto'}",
                f"[bold][[bold #7FFFD4] 0.0  2.0 [/]]",
                f"[bold]模糊 [bold #FFD700]{x.gauss}[/]" if x.gauss else f"[bold #A4D3EE]自动"
            ]
        },
        "--grind": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["一次", "=POSITIVE INT"],
            "help": "边缘觉醒",
            "func": """- 基于边缘增强或锐化算法（如拉普拉斯或高通滤波），提高图像边缘与纹理的清晰度。
                - 适用于修复模糊图像、强化结构轮廓，或在特定分析任务中凸显细节信息。""",
            "push": lambda x, y: [
                f"[bold #87AFD7]{x.grind or 'Auto'}",
                f"[bold][[bold #7FFFD4] 0.0  2.0 [/]]",
                f"[bold]锐化 [bold #FFD700]{x.grind}[/]" if x.grind else f"[bold #A4D3EE]自动"
            ]
        },
        "--frate": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["一次", "=POSITIVE INT"],
            "help": "频率探测",
            "func": """- 控制视频处理过程中的帧率采样频率，用于平衡图像精度与计算负载。
                - 常用于视频加速预览、动作识别、性能优化等任务中，尤其适合需要高帧细节的分析场景。""",
            "push": lambda x, y: [
                f"[bold #87AFD7]{x.frate or 'Auto'}",
                f"[bold][[bold #7FFFD4]  30  60  [/]]",
                f"[bold]帧率 [bold #FFD700]{x.frate}[/]" if x.frate else f"[bold #A4D3EE]自动"
            ]
        }
    }

    # 智析引擎
    GROUP_EXTRA = {
        "--boost": {
            "args": {"action": "store_true"},
            "view": ["一次", " "],
            "help": "加速跳跃",
            "func": """- 控制视频帧的抽样与处理节奏，特别是在选取关键帧和处理这些帧时如何优化和加速这一过程。
                - 通过智能跳帧与关键帧优选机制，加速分析流程，提升处理效率，尤其适用于长视频或资源受限场景。""",
            "push": lambda x, y: [
                f"[bold #87AFD7]{x.boost}",
                f"[bold][[bold #7FFFD4]   T  F   [/]]",
                f"[bold #CAFF70]开启" if x.boost else f"[bold #FFB6C1]关闭"
            ]
        },
        "--color": {
            "args": {"action": "store_true"},
            "view": ["一次", " "],
            "help": "彩绘世界",
            "func": """- 可以根据具体的应用需求控制是否启用彩色通道进行处理。
                - 在无需色彩信息的场景下可选择灰度模式，以降低计算负担并提升处理速度，实现性能与视觉需求之间的灵活权衡。""",
            "push": lambda x, y: [
                f"[bold #87AFD7]{x.color}",
                f"[bold][[bold #7FFFD4]   T  F   [/]]",
                f"[bold #CAFF70]开启" if x.color else f"[bold #FFB6C1]关闭"
            ]
        },
        "--begin": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["一次", "=STAGE_INDEX,FRAME_INDEX"],
            "help": "序章开启",
            "func": """- 用于定位并选取视频中初始的不稳定阶段帧，作为分析或剪辑的起始点。
                - 该参数可实现对内容开头的精细控制，适用于关键时刻的标注与后期处理。""",
            "push": lambda x, y: [
                f"[bold #87AFD7]{x.begin}",
                f"[bold][[bold #7FFFD4]   ?  ?   [/]]",
                f"[bold]非稳定阶段 [bold #FFD700]{list(x.begin)}[/]"
            ]
        },
        "--final": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["一次", "=STAGE_INDEX,FRAME_INDEX"],
            "help": "终章落幕",
            "func": """- 用于定义视频处理的终止点，支持通过 `final_stage` 或 `final_frame` 指定结束阶段或具体帧位置。
                - 当参数设为 `-1` 时，表示自动选取视频的最后一个阶段或最后一帧，便于快速截取完整片段的末尾部分。""",
            "push": lambda x, y: [
                f"[bold #87AFD7]{x.final}",
                f"[bold][[bold #7FFFD4]   ?  ?   [/]]",
                f"[bold]非稳定阶段 [bold #FFD700]{list(x.final)}[/]"
            ]
        },
        "--thres": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["一次", "=FLOAT"],
            "help": "稳定阈值",
            "func": """- 用于设定图像帧之间结构相似性（SSIM）的判定阈值。
                - 该值越高，对图像稳定状态的判定越严格，在视频剪辑、关键帧提取、稳定性检测等任务中提升图像变化识别的敏感度与精度。""",
            "push": lambda x, y: [
                f"[bold #87AFD7]{x.thres}",
                f"[bold][[bold #7FFFD4]0.85  1.00[/]]",
                f"[bold]帧间相似度大于 [bold #FFD700]{x.thres}[/] 视为稳定"
            ]
        },
        "--shift": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["一次", "=POSITIVE INT"],
            "help": "偏移调整",
            "func": """- 定义视频区间匹配时的最大允许偏移容差，适用于对比两个片段之间的起止时间或帧编号差异。
                - 该参数广泛用于多段拼接、场景变化识别及视频同步过程中，对容忍度进行灵活调控。""",
            "push": lambda x, y: [
                f"[bold #87AFD7]{x.shift}",
                f"[bold][[bold #7FFFD4]   0  15  [/]]",
                f"[bold]允许合并 [bold #FFD700]{x.shift}[/] 个间隔帧"
            ]
        },
        "--slide": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["一次", "=POSITIVE INT"],
            "help": "星线踏步",
            "func": """- 控制滑动窗口在视频帧序列中的推进步幅。
                - 步幅越小，窗口间重叠度越高，有助于提升检测精度；步幅越大，推进速度更快，有利于大规模分析。
                - 配合 `--scope`（序维穿梭）与 `--grade`（相位律动）使用，可精准调节分析节奏与资源消耗之间的平衡。""",
            "push": lambda x, y: [
                f"[bold #87AFD7]{x.slide}",
                f"[bold][[bold #7FFFD4]   1  10  [/]]",
                f"[bold]滑动窗口在帧序列中每次前进 [bold #FFD700]{x.slide}[/] 步"
            ]
        },
        "--block": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["一次", "=POSITIVE INT"],
            "help": "矩阵分割",
            "func": """- 指定图像在水平和垂直方向上的分块数量，用于将整帧图像划分为更小的子区域。
                - 提升图像处理、运动检测、局部特征提取等操作中的计算效率与空间分辨能力，适用于高分辨率视频的并行分析与局部建模。""",
            "push": lambda x, y: [
                f"[bold #87AFD7]{x.block}",
                f"[bold][[bold #7FFFD4]   1  10  [/]]",
                f"[bold]每帧分块数 [bold #FFD700]{x.block} * {x.block} = {x.block * x.block}"
            ]
        },
        "--scope": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["一次", "=POSITIVE INT"],
            "help": "序维穿梭",
            "func": """- 控制滑动窗口的分析跨度，即每次运算所覆盖的帧数区间。
                - 决定了算法对“时间”维度的理解深度，影响稳定性判断与细节捕捉能力。
                - 本质上是时间感知的“感受器尺寸”，范围越大，感知越长；范围越小，感知越敏捷。""",
            "push": lambda x, y: [
                f"[bold #87AFD7]{x.scope}",
                f"[bold][[bold #7FFFD4]   1  20  [/]]",
                f"[bold]滑动窗口中包含 [bold #FFD700]{x.scope}[/] 帧"
            ]
        },
        "--grade": {
            "args": {"nargs": "?", "const": None, "type": str},
            "view": ["一次", "=POSITIVE INT"],
            "help": "相位律动",
            "func": """- 控制滑动窗口内各帧的加权分布方式，用于平滑值（如 SSIM、MSE、PSNR）的加权计算。
                - 权重分布以幂函数方式衰减：靠近当前帧的权重更高，远离当前帧的影响力降低。
                - 本质上模拟“时序惯性”与“当前性偏好”，可理解为时序分析中的“物理驱动感”。""",
            "push": lambda x, y: [
                f"[bold #87AFD7]{x.grade}",
                f"[bold][[bold #7FFFD4]   1  5   [/]]",
                f"[bold]滑动窗口中加权系数的幂指数 [bold #FFD700]{x.grade}[/]"
            ]
        },
        "--crops": {
            "args": {"action": "append"},
            "view": ["多次", " X,Y,X_SIZE,Y_SIZE"],
            "help": "视界探索",
            "func": """- 该参数通过定义区域坐标和尺寸，允许用户灵活聚焦图像片段。
                - 指定裁剪区域的左上角坐标 `(x, y)` 和该区域的宽度 `x_size` 及高度 `y_size`。""",
            "push": lambda x, y: [
                f"[bold #87AFD7]{['!' for _ in range(len(x.crops))]}",
                f"[bold][[bold #7FFFD4]0.00  1.00[/]]",
                f"[bold]探索 [bold #FFD700]{len(x.crops)}[/] 个区域的图像"
            ]
        },
        "--omits": {
            "args": {"action": "append"},
            "view": ["多次", " X,Y,X_SIZE,Y_SIZE"],
            "help": "视界忽略",
            "func": """- 定义图像中应排除处理的区域。
                - 与 `--crops` 相反，此参数用于标记无需参与处理或分析的图像区域。""",
            "push": lambda x, y: [
                f"[bold #87AFD7]{['!' for _ in range(len(x.omits))]}",
                f"[bold][[bold #7FFFD4]0.00  1.00[/]]",
                f"[bold]忽略 [bold #FFD700]{len(x.omits)}[/] 个区域的图像"
            ]
        }
    }

    # 漏洞追踪
    GROUP_DEBUG = {
        "--debug": {
            "args": {"action": "store_true"},
            "view": ["一次", " "],
            "help": "架构透镜",
            "func": """- 激活调试模式后，系统将进入高透明度的运行追踪状态，输出底层架构的完整执行日志。
                - 该功能专为开发和测试阶段设计，能够深入揭示每个处理阶段的内部行为和关键变量状态。"""
        }
    }

    # Argument
    ARGUMENT = {
        "核心操控": GROUP_MAJOR,
        "辅助利器": GROUP_MEANS,
        "视控精灵": GROUP_SPACE,
        "显示布局": GROUP_ORDER,
        "时序调控": GROUP_TIMER,
        "像素工坊": GROUP_MEDIA,
        "数据智绘": GROUP_ARRAY,
        "图帧先行": GROUP_FIRST,
        "智析引擎": GROUP_EXTRA,
        "漏洞追踪": GROUP_DEBUG,
    }


class Wind(object):
    """Wind style"""

    SPEED_TEXT = """> >> >>> >>>> >>>>> >>>>>> >>>>>>> >>>>>>>> >>>>>> >>>>> >>>> >>> >> >
> > > > > > > > > > > > > > > > > > > >"""

    BASIC_TEXT = """[####][####][####][####][####][####]
[##################################]"""

    KERAS_TEXT = """|> * -> * -> * -> * -> * -> * -> * -> * -> * -> *
|> * -> * -> * -> * -> * -> * -> * -> * -> * -> *"""

    INFER_TEXT = """==>>==>>==>>==>>==>>==>>==>>==>>==>>==>>==>>==>>==>>==>>==>>
<<==<<==<<==<<==<<==<<==<<==<<==<<==<<==<<==<<==<<==<<==<<=="""

    MOVIE_TEXT = """▌▒░▐▌▒░▐▌▒░▐▌▒░▐▌▒░▐▌▒░▐▌▒░▐▌▒░▐▌▒░▐▌▒░▐▌▒░▐▌▒░▐
▒▐▌░▐▌▒░▐▌▒░▐▌▒░▐▌▒░▐▌▒░▒▐▌░▐▌▒░▐▌▒░▐▌▒░▐▌▒░▐▌▒░"""

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

    INFER = {
        "文本": {
            "style": "bold #00FF7F",
            "justify": "center",
        },
        "边框": {
            "title": "**<* 感知链接 *>**",
            "title_align": "center",
            "subtitle": None,
            "subtitle_align": "center",
            "border_style": "bold #00CED1",
        }
    }

    MOVIE = {
        "文本": {
            "style": "bold #FFFF00",
            "justify": "center",
        },
        "边框": {
            "title": "**<* 影像捕手 *>**",
            "title_align": "center",
            "subtitle": None,
            "subtitle_align": "center",
            "border_style": "bold #87CEFA",
        }
    }

    PREDICT = {
        "文本": {
            "style": "bold #A6E3E9",
            "justify": "left",
        },
        "边框": {
            "title": "**<* 感知链接 *>**",
            "title_align": "center",
            "subtitle": None,
            "subtitle_align": "center",
            "border_style": "bold #5EEAD4",
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
