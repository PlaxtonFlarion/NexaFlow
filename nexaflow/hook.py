#
#   _   _             _
#  | | | | ___   ___ | | __
#  | |_| |/ _ \ / _ \| |/ /
#  |  _  | (_) | (_) |   <
#  |_| |_|\___/ \___/|_|\_\
#

import os
import cv2
import typing
# from loguru import logger
from nexaflow import toolbox
from nexaflow.video import VideoFrame


class BaseHook(object):

    def __init__(self, *_, **__):
        # logger.debug(f"start initialing: {self.__class__.__name__} ...")
        self.result = dict()

    def do(self, frame: VideoFrame, *_, **__) -> typing.Optional[VideoFrame]:
        # info = f"execute hook: {self.__class__.__name__}"

        frame_id = frame.frame_id
        if frame_id != -1:
            # logger.debug(f"{info}, frame id: {frame_id}")
            pass
        return frame


class ExampleHook(BaseHook):

    def __init__(self, *_, **__):
        super().__init__(*_, **__)

    def do(self, frame: VideoFrame, *_, **__) -> typing.Optional[VideoFrame]:
        super().do(frame, *_, **__)
        frame.data = toolbox.turn_grey(frame.data)
        self.result[frame.frame_id] = frame.data.shape

        return frame


class FrameSizeHook(BaseHook):

    def __init__(
        self,
        compress_rate: float = None,
        target_size: typing.Tuple[int, int] = None,
        not_grey: bool = None,
        *_,
        **__,
    ):
        super().__init__(*_, **__)
        self.compress_rate = compress_rate
        self.target_size = target_size
        self.not_grey = not_grey
        # logger.debug(f"compress rate: {compress_rate}")
        # logger.debug(f"target size: {target_size}")

    def do(self, frame: VideoFrame, *_, **__) -> typing.Optional[VideoFrame]:
        super().do(frame, *_, **__)
        frame.data = toolbox.compress_frame(
            frame.data, self.compress_rate, self.target_size, self.not_grey
        )
        return frame


class GreyHook(BaseHook):

    def do(self, frame: VideoFrame, *_, **__) -> typing.Optional[VideoFrame]:
        super().do(frame, *_, **__)
        frame.data = toolbox.turn_grey(frame.data)
        return frame


class RefineHook(BaseHook):

    def do(self, frame: VideoFrame, *_, **__) -> typing.Optional[VideoFrame]:
        super().do(frame, *_, **__)
        frame.data = toolbox.sharpen_frame(frame.data)
        return frame


class _AreaBaseHook(BaseHook):

    def __init__(
        self,
        size: typing.Tuple[typing.Union[int, float], typing.Union[int, float]],
        offset: typing.Tuple[typing.Union[int, float], typing.Union[int, float]] = None,
        *_,
        **__,
    ):
        super().__init__(*_, **__)
        self.size = size
        self.offset = offset or (0, 0)
        # logger.debug(f"size: {self.size}")
        # logger.debug(f"offset: {self.offset}")

    @staticmethod
    def is_proportion(
        target: typing.Tuple[typing.Union[int, float], typing.Union[int, float]]
    ) -> bool:
        return len([i for i in target if 0.0 <= i <= 1.0]) == 2

    @staticmethod
    def convert(
        origin_h: int,
        origin_w: int,
        input_h: typing.Union[float, int],
        input_w: typing.Union[float, int],
    ) -> typing.Tuple[typing.Union[int, float], typing.Union[int, float]]:
        if _AreaBaseHook.is_proportion((input_h, input_w)):
            return origin_h * input_h, origin_w * input_w
        return input_h, input_w

    def convert_size_and_offset(
        self, *origin_size
    ) -> typing.Tuple[typing.Tuple, typing.Tuple]:

        size_h, size_w = self.convert(*origin_size, *self.size)
        # logger.debug(f"size: ({size_h}, {size_w})")
        offset_h, offset_w = self.convert(*origin_size, *self.offset)
        # logger.debug(f"offset: {offset_h}, {offset_w}")
        height_range, width_range = (
            (int(offset_h), int(offset_h + size_h)),
            (int(offset_w), int(offset_w + size_w)),
        )
        # logger.debug(f"final range h: {height_range}, w: {width_range}")
        return height_range, width_range


class CropHook(_AreaBaseHook):

    def do(self, frame: "VideoFrame", *_, **__) -> typing.Optional["VideoFrame"]:
        super().do(frame, *_, **__)

        height_range, width_range = self.convert_size_and_offset(*frame.data.shape)
        frame.data[: height_range[0], :] = 0
        frame.data[height_range[1]:, :] = 0
        frame.data[:, : width_range[0]] = 0
        frame.data[:, width_range[1]:] = 0
        return frame


class OmitHook(_AreaBaseHook):

    def do(self, frame: "VideoFrame", *_, **__) -> typing.Optional["VideoFrame"]:
        super().do(frame, *_, **__)

        height_range, width_range = self.convert_size_and_offset(*frame.data.shape)
        frame.data[
            height_range[0]: height_range[1], width_range[0]: width_range[1]
        ] = 0
        return frame


class PaintCropHook(_AreaBaseHook):

    def do(self, frame: "VideoFrame", *_, **__) -> typing.Optional["VideoFrame"]:
        super().do(frame, *_, **__)

        height_range, width_range = self.convert_size_and_offset(*frame.data.shape[:2])
        frame.data[: height_range[0], :] = 0
        frame.data[height_range[1]:, :] = 0
        frame.data[:, : width_range[0]] = 0
        frame.data[:, width_range[1]:] = 0
        return frame


class PaintOmitHook(_AreaBaseHook):

    def do(self, frame: "VideoFrame", *_, **__) -> typing.Optional["VideoFrame"]:
        super().do(frame, *_, **__)

        height_range, width_range = self.convert_size_and_offset(*frame.data.shape[:2])
        frame.data[
            height_range[0]: height_range[1], width_range[0]: width_range[1]
        ] = 0
        return frame


class FrameSaveHook(BaseHook):

    def __init__(self, target_dir: str, *_, **__):
        super().__init__(*_, **__)

        self.target_dir = target_dir
        os.makedirs(target_dir, exist_ok=True)
        # logger.debug(f"target dir: {target_dir}")

    def do(self, frame: VideoFrame, *_, **__) -> typing.Optional[VideoFrame]:
        super().do(frame, *_, **__)

        safe_timestamp = str(frame.timestamp).replace(".", "_")
        frame_name = f"{frame.frame_id}({safe_timestamp}).png"
        target_path = os.path.join(self.target_dir, frame_name)

        # 不能保存中文路径
        # cv2.imwrite(target_path, frame.data)
        # logger.debug(f"frame saved to {target_path}")

        # 保存中文路径
        cv2.imencode(".png", frame.data)[1].tofile(target_path)

        return frame


if __name__ == '__main__':
    pass
