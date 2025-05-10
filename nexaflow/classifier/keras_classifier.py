#   _  __                     ____ _               _  __ _
#  | |/ /___ _ __ __ _ ___   / ___| | __ _ ___ ___(_)/ _(_) ___ _ __
#  | ' // _ \ '__/ _` / __| | |   | |/ _` / __/ __| | |_| |/ _ \ '__|
#  | . \  __/ | | (_| \__ \ | |___| | (_| \__ \__ \ |  _| |  __/ |
#  |_|\_\___|_|  \__,_|___/  \____|_|\__,_|___/___/_|_| |_|\___|_|
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

import os
import cv2
import numpy
import typing
import pathlib

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow
except ImportError:
    raise ImportError("KerasClassifier requires tensorflow. install it first.")

from loguru import logger
from tensorflow import keras
from nexaflow import (
    toolbox, const
)
from nexaflow.video import VideoFrame
from nexaflow.classifier.base import BaseModelClassifier


class KerasStruct(BaseModelClassifier):
    """
    使用 Keras 实现的卷积神经网络结构分类器。

    该分类器支持模型的构建、训练、保存、加载与推理，基于 TensorFlow/Keras 提供的深度学习框架，
    适用于视频帧图像的阶段分类任务。内部模型结构包含多层卷积池化、Dropout 和全连接层，
    可在训练数据集上进行拟合并进行高效推理。

    支持：
        - 动态构建模型结构并设置训练参数
        - 基于图像目录数据进行训练和验证
        - 推理阶段支持路径或对象图像输入
        - 模型保存为 TensorFlow 格式

    默认支持的类别数由 `MODEL_DENSE` 决定。
    """

    MODEL_DENSE = 6

    def __init__(self, *_, **kwargs):
        super(KerasStruct, self).__init__(*_, **kwargs)

        # Model
        self.model: typing.Optional["keras.Sequential"] = None
        # Model Config
        self.score_threshold: float = kwargs.get("score_threshold", 0.0)
        self.nb_train_samples: int = kwargs.get("nb_train_samples", 64)
        self.nb_validation_samples: int = kwargs.get("nb_validation_samples", 64)
        self.epochs: int = kwargs.get("epochs", 20)
        self.batch_size: int = kwargs.get("batch_size", 4)

        # logger.debug(f"score threshold: {self.score_threshold}")
        # logger.debug(f"nb train samples: {self.nb_train_samples}")
        # logger.debug(f"nb validation samples: {self.nb_validation_samples}")
        # logger.debug(f"epochs: {self.epochs}")
        # logger.debug(f"batch size: {self.batch_size}")

    @property
    def follow_cv_size(self) -> tuple:
        """
        获取当前加载模型的输入图像尺寸（宽度，高度）。

        本属性返回模型期望的输入图像空间维度（不包含通道数与 batch size），
        用于后续图像预处理、输入格式转换等操作。

        Returns
        -------
        tuple
            二元组 (width, height)，表示模型输入图像的空间尺寸。

        Raises
        ------
        AssertionError
            如果模型尚未加载（`self.model` 为 None），则抛出异常。

        Notes
        -----
        - OpenCV 与 TensorFlow/Keras 的尺寸格式顺序不同，传递图像尺寸时需特别注意：
        - OpenCV 使用 (width, height)
        - TensorFlow/Keras 使用 (height, width)
        - 转换示例
            cv_size = (640, 480)  # OpenCV 尺寸
            tf_size = (cv_size[1], cv_size[0])  # 转为 TensorFlow 尺寸
        - 该尺寸取自模型的 input shape（去除 batch 和通道信息）；
        - 仅在模型已加载后可访问。
        """
        assert self.model, f"Keras sequence model not loaded"
        return self.model.input_shape[1], self.model.input_shape[2]

    def load_model(self, model_path: str, overwrite: bool = None) -> None:
        """
        加载 Keras 模型文件并初始化模型结构。

        使用 `keras.models.load_model()` 从指定路径加载模型，并绑定到当前对象。
        支持 TF SavedModel 格式。

        Parameters
        ----------
        model_path : str
            模型所在的目录路径，必须为有效的 TF SavedModel 文件夹。
        overwrite : bool, optional
            是否覆盖现有模型（当前未使用，仅保留参数占位）。

        Raises
        ------
        AssertionError
            如果指定路径不存在或不是有效目录，将抛出异常。

        Notes
        -----
        - 日志会记录加载路径与模型结构；
        - 模型加载后，可使用 `follow_cv_size` 获取输入尺寸；
        - 推荐使用 `.save(..., save_format='tf')` 生成的模型目录进行加载。
        """
        logger.debug(f"Keras sequence model load from {model_path}")

        assert os.path.isdir(model_path), f"model file {model_path} not existed"

        self.model = keras.models.load_model(model_path)
        logger.debug(f"Keras sequence model load data {self.model.input_shape}")

    def create_model(self, follow_tf_size: tuple, model_aisle: int) -> "keras.Sequential":
        """
        创建一个基于卷积神经网络（CNN）的 Keras 序列模型。

        本方法构建一个多层卷积网络结构，适用于多分类图像识别任务。模型由 3 层卷积块（Conv2D + MaxPooling + Dropout）
        和 2 层全连接层（Flatten + Dense）构成，输出层使用 softmax 激活函数以适配多类标签的概率预测。

        Parameters
        ----------
        follow_tf_size : tuple
            图像输入尺寸，格式为 (height, width)，不包含通道维度。

        model_aisle : int
            输入图像的通道数，通常为 1（灰度图）或 3（RGB 彩色图）。

        Returns
        -------
        keras.Sequential
            构建好的 Keras `Sequential` 模型对象，可直接用于训练、预测或保存。

        Notes
        -----
        - 本模型设计用于轻量级图像分类任务，适合在资源受限设备（如移动端或嵌入式）上进行部署。
        - 网络层数较浅，能够在小规模样本集（~几百张）上完成过拟合控制良好的训练。
        - Dropout 层用于防止过拟合，在每个卷积块后均加入一定比例的随机失活单元。
        - 输出层使用 softmax 适配稀疏标签格式，配合 `sparse_categorical_crossentropy`，可避免将标签转换为独热编码。
        - 可通过调整 `self.MODEL_DENSE` 改变最终分类数量（默认为 6 类）。
        - 输入图像需归一化到 [0, 1] 区间，建议在预处理阶段进行像素值缩放。
        - 对于彩色图像建议统一转为灰度后处理（可降低模型复杂度）。
        - 若训练数据较小，可尝试增大 Dropout 概率、增加图像增强策略（如旋转、平移、裁剪等）。

        Workflow
        --------
        1. 检查当前 Keras 后端图像格式，构造输入形状 `input_shape`；
        2. 创建顺序模型 `Sequential` 实例；
        3. 添加第 1 个卷积块：
            - Conv2D(32 filters, kernel size 3x3, padding="same")
            - MaxPooling2D(pool size=2x2)
            - Dropout(rate=0.25)
        4. 添加第 2 个卷积块：
            - Conv2D(64 filters, kernel size 3x3, padding="same")
            - MaxPooling2D(pool size=2x2)
            - Dropout(rate=0.25)
        5. 添加第 3 个卷积块：
            - Conv2D(128 filters, kernel size 3x3, padding="same")
            - MaxPooling2D(pool size=2x2)
            - Dropout(rate=0.25)
        6. 添加全连接部分：
            - Flatten
            - Dense(256 units, activation="relu")
            - Dropout(rate=0.5)
            - Dense(self.MODEL_DENSE units, activation="softmax")
        7. 编译模型：使用 Adam 优化器，损失函数为 sparse_categorical_crossentropy，评估指标为 accuracy。
        8. 返回已构建并编译完成的模型对象。
        """
        logger.debug(f"Keras sequence model is being created")

        if keras.backend.image_data_format() == "channels_first":
            input_shape = (model_aisle, *follow_tf_size)
        else:
            input_shape = (*follow_tf_size, model_aisle)

        model = keras.Sequential()

        model.add(keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Conv2D(64, (3, 3), padding="same"))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Conv2D(128, (3, 3), padding="same"))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256, activation="relu"))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(self.MODEL_DENSE, activation="softmax"))

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        logger.debug("Keras sequence model is created")
        return model

    def train(self, data_path: typing.Optional[str] = None, *args, **kwargs) -> None:
        """
        使用指定目录中的图像数据训练 Keras 模型。

        本方法读取文件夹结构格式的数据集，自动划分训练集与验证集，构建图像增强流水线，并执行模型训练。
        若模型尚未初始化，则会自动调用 `create_model()` 创建新模型。

        Parameters
        ----------
        data_path : str
            数据集根目录路径。该路径下应包含多个子目录，每个子目录对应一个类别，内部包含该类的图像文件。
        *args :
            包含以下位置参数（按顺序）：
                - model_color (str): 图像颜色模式，常用值为 'grayscale' 或 'rgb'。
                - follow_tf_size (tuple): 模型期望输入的图像尺寸 (height, width)。
                - model_aisle (int): 图像通道数，如 1 表示灰度图，3 表示彩色图。
        **kwargs :
            其他可选关键字参数（未使用）。

        Raises
        ------
        AssertionError
            若 `data_path` 无效、类别数小于 2 或大于模型支持上限，则抛出异常。

        Notes
        -----
        - 本方法内部使用 Keras 的 `ImageDataGenerator` 构建训练/验证数据生成器，支持自动增强与数据分割。
        - `rescale=1/16` 可将原始图像像素值（0~255）缩放到 0~16 区间，需与模型设计保持一致。
        - `class_mode="sparse"` 可避免对标签做独热编码，适用于使用 `sparse_categorical_crossentropy` 损失函数的多分类任务。
        - 默认划分比例为 2:1，即验证集占比约为 33%。

        Workflow
        --------
        1. 执行 `check()` 函数校验数据路径：
            - 确保目录存在；
            - 至少包含两个类别子文件夹；
            - 类别数不超过 `self.MODEL_DENSE` 设置的最大类数；
        2. 解包模型参数：
            - 图像颜色模式；
            - 图像输入尺寸；
            - 图像通道数；
        3. 若尚未加载模型，则调用 `create_model()` 创建新模型；
        4. 初始化 `ImageDataGenerator`：
            - 配置增强参数（剪切、缩放、水平翻转）；
            - 设置验证集划分比例；
        5. 构建训练集与验证集数据生成器：
            - 从 `data_path` 加载图像；
            - 使用指定目标尺寸、批次大小和颜色模式；
        6. 调用 `model.fit()` 开始训练模型；
        7. 训练完成后输出调试日志。
        """

        def check(p: str):
            p = pathlib.Path(p)
            assert p.is_dir(), f"{p} is not a valid directory"

            number_of_dir = len([each for each in os.listdir(p) if (p / each).is_dir()])
            assert (
                number_of_dir > 1
            ), f"dataset only contains one class. maybe some path errors happened: {p}?"

            assert number_of_dir <= self.MODEL_DENSE, (
                f"dataset has {number_of_dir} classes (more than " + str(self.MODEL_DENSE) + ")"
            )

        check(data_path)

        model_color, follow_tf_size, model_aisle, *_ = args

        if not self.model:
            self.model = self.create_model(follow_tf_size, model_aisle)

        datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 16,
            shear_range=0.2,
            zoom_range=0.2,
            validation_split=0.33,
            horizontal_flip=True
        )

        train_generator = datagen.flow_from_directory(
            data_path,
            target_size=follow_tf_size,
            batch_size=self.batch_size,
            color_mode=model_color,
            class_mode="sparse",
            subset="training",
        )

        validation_generator = datagen.flow_from_directory(
            data_path,
            target_size=follow_tf_size,
            batch_size=self.batch_size,
            color_mode=model_color,
            class_mode="sparse",
            subset="validation",
        )

        self.model.fit(
            train_generator,
            epochs=self.epochs,
            validation_data=validation_generator,
        )

        logger.debug("Model train finished")

    def build(self, model_color: str, model_shape: tuple, model_aisle: int, *args) -> typing.Optional[str]:
        """
        构建并保存训练完成的 Keras 模型。

        该方法首先根据指定的数据集路径进行模型训练，随后将训练好的模型保存为 TensorFlow 格式文件，并返回模型保存路径。
        若训练阶段出现异常（如数据错误），则捕获并记录错误信息。

        Parameters
        ----------
        model_color : str
            图像颜色模式，如 "grayscale" 或 "rgb"。影响训练数据的通道数。
        model_shape : tuple
            图像尺寸信息，格式为 (width, height)，将用于调整输入图像的目标大小。
        model_aisle : int
            图像通道数，通常为 1（灰度）或 3（彩色）。
        *args :
            依次包含以下三个位置参数：
                - src_model_path (str): 训练数据所在的目录路径。
                - new_model_path (str): 训练完成后保存模型的目标目录。
                - new_model_name (str): 保存的模型文件名称（不含后缀）。

        Returns
        -------
        Optional[str]
            若模型训练和保存成功，返回保存后的模型文件路径；否则返回 None，并通过日志输出错误信息。

        Notes
        -----
        - 模型保存使用 TensorFlow 原生格式（SavedModel），兼容后续部署与推理。
        - 若目录不存在则自动创建；支持跨平台。
        - 若模型尚未初始化，则会自动构建新的网络结构并进行训练。
        - 如果训练阶段抛出异常（如路径无效或类别不足），则返回 None。

        Workflow
        --------
        1. 解构位置参数，提取训练路径、保存路径、模型名称；
        2. 将模型输入尺寸 `model_shape` 转换为符合 TensorFlow 格式的 (height, width)；
        3. 调用 `train()` 方法进行模型训练：
            - 若训练失败，则记录错误并返回；
        4. 创建模型保存路径（如不存在）；
        5. 使用 `.save()` 保存模型至指定路径；
        6. 输出模型结构概要（`model.summary()`）以供审查；
        7. 返回模型保存的绝对路径。
        """
        src_model_path, new_model_path, new_model_name = args

        follow_tf_size = model_shape[1], model_shape[0]

        try:
            self.train(
                src_model_path, model_color, follow_tf_size, model_aisle
            )
        except AssertionError as e:
            return logger.error(e)

        final_model: str = os.path.join(new_model_path, new_model_name).format()
        os.makedirs(new_model_path, exist_ok=True)

        # self.model.save_weights(final_model, save_format="h5")
        self.model.save(final_model, save_format="tf")
        self.model.summary()

        return final_model

    def predict(self, pic_path: str, *args, **kwargs) -> str:
        """
        使用模型对图像路径指定的图像进行预测。

        本方法从给定路径加载图像，将其转换为 `VideoFrame` 对象，并应用模型前处理钩子，随后调用 `predict_with_object` 进行推理。

        Parameters
        ----------
        pic_path : str
            图像文件的路径，应为本地可读取的图片。
        *args, **kwargs :
            可选参数，将传递给钩子处理函数 `_apply_hook`。

        Returns
        -------
        str
            预测结果标签（字符串形式），如果置信度过低，则返回 `const.UNKNOWN_STAGE_FLAG`。

        Notes
        -----
        - 使用 `toolbox.imread()` 读取图像并确保为 OpenCV 格式；
        - 构造伪 `VideoFrame` 对象用于兼容钩子机制；
        - 支持模型前自定义图像变换处理流程；
        - 实际预测由 `predict_with_object` 执行。
        """
        picture = toolbox.imread(pic_path)
        fake_frame = VideoFrame(0, 0.0, picture)
        fake_frame = self._apply_hook(fake_frame, *args, **kwargs)
        return self.predict_with_object(fake_frame.data)

    def predict_with_object(self, frame: "numpy.ndarray") -> str:
        """
        使用模型对输入的图像数据进行预测。

        该方法接收一个已处理的图像数组，自动调整尺寸并扩展维度后输入模型进行预测，返回最可能的类别标签。

        Parameters
        ----------
        frame : numpy.ndarray
            图像数组，要求为单通道（灰度）或多通道格式，类型为 `np.ndarray`。

        Returns
        -------
        str
            预测的类别标签，若置信度低于设定阈值 `self.score_threshold`，则返回 `const.UNKNOWN_STAGE_FLAG`。

        Notes
        -----
        - 图像将会缩放至模型的输入尺寸（`follow_cv_size`）；
        - 自动添加 batch 维度和通道维度，以满足 Keras 输入格式；
        - 返回标签类型为字符串；
        - 若预测分数不足，将标记为未知分类，适用于低置信度过滤。
        """
        frame = cv2.resize(frame, dsize=self.follow_cv_size)
        frame = numpy.expand_dims(frame, axis=[0, -1])
        frame_result = self.model.predict(frame, verbose=0)
        frame_tag = str(numpy.argmax(frame_result, axis=1)[0])
        frame_confidence = frame_result.max()

        if frame_confidence < self.score_threshold:
            logger.warning(
                f"max score is lower than {self.score_threshold}, unknown class"
            )
            return const.UNKNOWN_STAGE_FLAG
        return frame_tag

    def _classify_frame(self, frame: "VideoFrame", *_, **__) -> str:
        """
        内部方法：对指定帧对象进行分类。

        Parameters
        ----------
        frame : VideoFrame
            待分类的帧对象，需包含图像数据（`frame.data`）。

        Returns
        -------
        str
            预测的阶段标签。

        Notes
        -----
        - 用于主分类逻辑中，包装调用 `predict_with_object`。
        """
        return self.predict_with_object(frame.data)


if __name__ == '__main__':
    pass
