import re
import datetime
import argparse


class Parser(object):

    @staticmethod
    def parse_shape(dim_str):
        if type(dim_str) is tuple:
            return dim_str
        elif type(dim_str) is list:
            if len(dim_str_list := [dim for dim in dim_str if type(dim) is int]) < 2:
                return None
            return tuple(dim_str_list)[:2]
        elif type(dim_str) is str:
            match_size_list = re.findall(r"-?\d*\.?\d+", dim_str)
            if len(match_size_list) >= 2:
                converted = []
                for num in match_size_list:
                    try:
                        converted_num = int(num)
                    except ValueError:
                        converted_num = float(num)
                    converted.append(converted_num)
                return tuple(converted[:2])
        return None

    @staticmethod
    def parse_aisle(dim_str):
        try:
            return value if (value := int(dim_str)) in [1, 3] else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def parse_scale(dim_str):
        try:
            value = float(dim_str) if dim_str else None
        except (ValueError, TypeError):
            return None
        return round(max(0.1, min(1.0, value)), 1) if value else None

    @staticmethod
    def parse_times(dim_str):
        hour_scope, second_scope = 24, 86400
        if type(dim_str) is int or type(dim_str) is float:
            if dim_str >= second_scope:
                return None
            return str(datetime.timedelta(seconds=dim_str))
        elif type(dim_str) is str:
            time_pattern = re.compile(r"(?:(\d+):)?(\d+):(\d+)(?:\.(\d+))?|^\d*(?:\.\d+)?$")
            if match := time_pattern.match(dim_str):
                hours = int(match.group(1)) if match.group(1) else 0
                minutes = int(match.group(2)) if match.group(2) else 0
                seconds = int(match.group(3)) if match.group(3) else 0
                milliseconds = int(float("0." + match.group(4)) * 1000) if match.group(4) else 0
                if match.group(0) and '.' in match.group(0):
                    seconds = float(match.group(0))
                    milliseconds = int((seconds - int(seconds)) * 1000)
                    seconds = int(seconds)
                time_str = datetime.timedelta(
                    hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds
                )
                return str(time_str) if hours <= hour_scope else None
        return None

    @staticmethod
    def parse_mills(dim_str):
        if type(dim_str) is int or type(dim_str) is float:
            return float(dim_str)
        if type(dim_str) is str:
            seconds_pattern = re.compile(r"^\d+(\.\d+)?$")
            full_pattern = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})(\.\d+)?")
            if match := full_pattern.match(dim_str):
                hours, minutes, seconds, milliseconds = match.groups()
                total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
                if milliseconds:
                    total_seconds += float(milliseconds)
                return total_seconds
            elif seconds_pattern.match(dim_str):
                return float(dim_str)
        return None

    @staticmethod
    def parse_stage(dim_str):
        if type(dim_str) is tuple:
            return dim_str
        elif type(dim_str) is list:
            if len(dim_str_list := [dim for dim in dim_str if type(dim) is int]) < 2:
                return None
            return tuple(dim_str_list)[:2]
        elif type(dim_str) is str:
            stage_parts = []
            parts = re.split(r"[.,;:\s]+", dim_str)
            match_parts = [part for part in parts if re.match(r"-?\d+(\.\d+)?", part)]
            for number in match_parts:
                try:
                    stage_parts.append(int(number))
                except ValueError:
                    stage_parts = []
                    break
            return tuple(stage_parts[:2]) if len(stage_parts) >= 2 else None
        else:
            return None

    @staticmethod
    def parse_hooks(dim_str):
        hooks_list, effective = dim_str, []
        for hook in hooks_list:
            if type(hook) is dict:
                data_list = [
                    v for v in hook.values() if type(v) is int or type(v) is float
                ]
                if len(data_list) == 4 and sum(data_list) > 0:
                    effective.append(hook)
        unique_hooks = {tuple(i.items()) for i in effective}
        return [dict(i) for i in unique_hooks]

    @staticmethod
    def parse_frate(dim_str):
        try:
            value = int(dim_str) if dim_str else None
        except (ValueError, TypeError):
            return None
        return round(max(1, min(60, int(value))), 1) if value else None

    @staticmethod
    def parse_thres(dim_str):
        try:
            value = float(dim_str) if dim_str else None
        except (ValueError, TypeError):
            return None
        return round(max(0.1, min(1.0, value)), 2) if value else None

    @staticmethod
    def parse_other(dim_str):
        try:
            value = int(dim_str) if dim_str else None
        except (ValueError, TypeError):
            return None
        return round(max(1, value), 0) if value else None

    @staticmethod
    def parse_cmd() -> argparse.Namespace:
        parser = argparse.ArgumentParser('framix', None, 'Command Line Arguments Framix')

        group_major_cmd = parser.add_argument_group(title='主要命令', description='参数互斥')
        group_major = group_major_cmd.add_mutually_exclusive_group()
        group_major.add_argument('--video', action='append', help='分析视频文件')
        group_major.add_argument('--stack', action='append', help='分析视频集合')
        group_major.add_argument('--train', action='append', help='训练模型')
        group_major.add_argument('--build', action='append', help='编译模型')
        group_major.add_argument('--flick', action='store_true', help='循环运行模式')
        group_major.add_argument('--carry', action='append', help='运行指定脚本')
        group_major.add_argument('--fully', action='append', help='运行全部脚本')
        group_major.add_argument('--paint', action='store_true', help='绘制分割线条')
        group_major.add_argument('--union', action='append', help='聚合视频帧报告')
        group_major.add_argument('--merge', action='append', help='聚合时间戳报告')

        group_means_cmd = parser.add_argument_group(title='附加命令', description='参数互斥')
        group_means = group_means_cmd.add_mutually_exclusive_group()
        group_means.add_argument('--quick', action='store_true', help='快速模式')
        group_means.add_argument('--basic', action='store_true', help='基础模式')
        group_means.add_argument('--keras', action='store_true', help='智能模式')

        group_extra = parser.add_argument_group(title='分析配置', description='参数兼容')
        group_extra.add_argument('--alone', action='store_true', help='独立控制')
        group_extra.add_argument('--group', action='store_true', help='分组报告')
        group_extra.add_argument('--boost', action='store_true', help='跳帧模式')
        group_extra.add_argument('--color', action='store_true', help='彩色模式')
        group_extra.add_argument('--shape', nargs='?', const=None, type=Parser.parse_shape, help='图片尺寸')
        group_extra.add_argument('--scale', nargs='?', const=None, type=Parser.parse_scale, help='缩放比例')
        group_extra.add_argument('--start', nargs='?', const=None, type=Parser.parse_mills, help='开始时间')
        group_extra.add_argument('--close', nargs='?', const=None, type=Parser.parse_mills, help='结束时间')
        group_extra.add_argument('--limit', nargs='?', const=None, type=Parser.parse_mills, help='持续时间')
        group_extra.add_argument('--begin', nargs='?', const=None, type=Parser.parse_stage, help='开始阶段')
        group_extra.add_argument('--final', nargs='?', const=None, type=Parser.parse_stage, help='结束阶段')
        group_extra.add_argument('--frate', nargs='?', const=None, type=Parser.parse_frate, help='帧采样率')
        group_extra.add_argument('--thres', nargs='?', const=None, type=Parser.parse_thres, help='相似度')
        group_extra.add_argument('--shift', nargs='?', const=None, type=Parser.parse_other, help='补偿值')
        group_extra.add_argument('--block', nargs='?', const=None, type=Parser.parse_other, help='立方体')
        group_extra.add_argument('--crops', action='append', help='获取区域')
        group_extra.add_argument('--omits', action='append', help='忽略区域')

        group_debug = parser.add_argument_group(title='调试配置', description='参数兼容')
        group_debug.add_argument('--debug', action='store_true', help='调试模式')

        return parser.parse_args()


if __name__ == '__main__':
    pass