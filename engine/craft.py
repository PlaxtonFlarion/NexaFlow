import os
import cv2
import typing
import aiofiles
from nexaflow import const
from nexaflow.classifier.base import SingleClassifierResult


class Craft(object):

    @staticmethod
    async def achieve(
            template: typing.Union[str, "os.PathLike"]
    ) -> typing.Union[str, "Exception"]:

        try:
            async with aiofiles.open(template, "r", encoding=const.CHARSET) as f:
                template_file = await f.read()
        except FileNotFoundError as e:
            return e
        return template_file

    @staticmethod
    async def frame_forge(
            frame: "SingleClassifierResult", frame_path: typing.Union[str, "os.PathLike"]
    ) -> typing.Union[dict, "Exception"]:

        try:
            picture = f"{frame.frame_id}_{format(round(frame.timestamp, 5), '.5f')}.png"
            _, codec = cv2.imencode(".png", frame.data)
            async with aiofiles.open(os.path.join(frame_path, picture), "wb") as f:
                await f.write(codec.tobytes())
        except Exception as e:
            return e
        return {"id": frame.frame_id, "picture": os.path.join(os.path.basename(frame_path), picture)}


if __name__ == '__main__':
    pass
