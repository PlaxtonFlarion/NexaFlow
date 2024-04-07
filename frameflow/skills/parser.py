import re
from argparse import ArgumentParser


class Parser(object):

    @staticmethod
    def parse_scale(dim_str):
        try:
            float_val = float(dim_str) if dim_str else None
        except ValueError:
            return None
        return round(max(0.1, min(1.0, float_val)), 2) if float_val else None

    @staticmethod
    def parse_sizes(dim_str):
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
    def parse_mills(dim_str):
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

    @staticmethod
    def parse_cmd():
        parser = ArgumentParser(description="Command Line Arguments Framix")

        parser.add_argument('--flick', action='store_true', help='循环分析视频帧')
        parser.add_argument('--paint', action='store_true', help='绘制图片分割线条')
        parser.add_argument('--video', action='append', help='分析视频')
        parser.add_argument('--stack', action='append', help='分析视频文件集合')
        parser.add_argument('--union', action='append', help='聚合视频帧报告')
        parser.add_argument('--merge', action='append', help='聚合时间戳报告')
        parser.add_argument('--train', action='append', help='归类图片文件')
        parser.add_argument('--build', action='append', help='训练模型文件')

        parser.add_argument('--carry', action='append', help='指定执行')
        parser.add_argument('--fully', action='store_true', help='自动执行')
        parser.add_argument('--alone', action='store_true', help='独立控制')
        parser.add_argument('--group', action='store_true', help='分组报告')
        parser.add_argument('--quick', action='store_true', help='快速模式')
        parser.add_argument('--basic', action='store_true', help='基础模式')
        parser.add_argument('--keras', action='store_true', help='智能模式')

        parser.add_argument('--boost', action='store_true', help='跳帧模式')
        parser.add_argument('--color', action='store_true', help='彩色模式')
        parser.add_argument('--shape', nargs='?', const=None, type=Parser.parse_sizes, help='图片尺寸')
        parser.add_argument('--scale', nargs='?', const=None, type=Parser.parse_scale, help='缩放比例')
        parser.add_argument('--start', nargs='?', const=None, type=Parser.parse_mills, help='开始时间')
        parser.add_argument('--close', nargs='?', const=None, type=Parser.parse_mills, help='结束时间')
        parser.add_argument('--limit', nargs='?', const=None, type=Parser.parse_mills, help='持续时间')
        parser.add_argument('--begin', nargs='?', const=None, type=Parser.parse_stage, help='开始帧')
        parser.add_argument('--final', nargs='?', const=None, type=Parser.parse_stage, help='结束帧')
        parser.add_argument('--crops', action='append', help='获取区域')
        parser.add_argument('--omits', action='append', help='忽略区域')

        # 调试模式
        parser.add_argument('--debug', action='store_true', help='调试模式')

        return parser.parse_args()


if __name__ == '__main__':
    pass
