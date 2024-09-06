#
#   _  __                     ____ _               _  __ _
#  | |/ /___ _ __ __ _ ___   / ___| | __ _ ___ ___(_)/ _(_) ___ _ __
#  | ' // _ \ '__/ _` / __| | |   | |/ _` / __/ __| | |_| |/ _ \ '__|
#  | . \  __/ | | (_| \__ \ | |___| | (_| \__ \__ \ |  _| |  __/ |
#  |_|\_\___|_|  \__,_|___/  \____|_|\__,_|___/___/_|_| |_|\___|_|
#

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
from nexaflow import toolbox, const
from nexaflow.video import VideoFrame
from nexaflow.classifier.base import BaseModelClassifier


class KerasStruct(BaseModelClassifier):

    MODEL_DENSE = 6

    def __init__(self, *_, **kwargs):
        """
        初始化方法: `__init__`

        功能:
            初始化 `KerasStruct` 类的实例，并设置模型训练的配置参数。

        参数:
            *_ (tuple): 忽略位置参数。
            **kwargs (dict): 关键字参数，用于配置模型训练的相关参数。

        操作流程:
            1. 调用父类的初始化方法，确保基础功能的初始化。
            2. 初始化私有属性 `self.__model` 为 `None`，用于存储Keras模型实例。
            3. 从 `kwargs` 中获取并设置模型训练的配置参数:
               - `self.score_threshold`：分数阈值，默认为 0.0。
               - `self.nb_train_samples`：训练样本数量，默认为 64。
               - `self.nb_validation_samples`：验证样本数量，默认为 64。
               - `self.epochs`：训练的迭代次数，默认为 20。
               - `self.batch_size`：每个批次的样本数量，默认为 4。

        备注:
            - 已注释的 `logger.debug` 语句可用于调试，输出各个配置参数的值。
        """

        super(KerasStruct, self).__init__(*_, **kwargs)

        # Model
        self.__model: typing.Optional["keras.Sequential"] = None
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
    def model(self) -> typing.Optional["keras.Sequential"]:
        return self.__model

    @model.setter
    def model(self, value):
        self.__model = value

    @property
    def follow_cv_size(self) -> tuple:
        """
        属性方法: `follow_cv_size`

        功能:
            返回模型输入层的尺寸（宽度和高度）。

        返回:
            tuple: 包含模型输入层的宽度和高度，格式为 `(width, height)`。

        操作流程:
            1. 通过 `self.model.input_shape` 获取模型输入的形状信息。
            2. 返回输入形状的第 1 和第 2 维度，分别表示宽度和高度。

        备注:
            - 假设 `self.model` 是一个 Keras 模型对象。
            - 输入形状的第 0 维通常是批次大小，因此跳过。
        """

        return self.model.input_shape[1], self.model.input_shape[2]

    def load_model(self, model_path: str, overwrite: bool = None) -> None:
        """
        方法: `load_model`

        功能:
            加载指定路径下的 Keras 序列模型，并将其赋值给实例的 `model` 属性。

        参数:
            model_path (str): 要加载的模型所在的文件路径。
            overwrite (bool, 可选): 决定是否覆盖已有模型，默认值为 `None`。此参数目前未被使用。

        操作流程:
            1. 记录日志，显示模型加载路径。
            2. 断言检查 `model_path` 是否为一个有效的目录路径，如果不是则抛出异常。
            3. 使用 `keras.models.load_model` 函数从指定路径加载模型。
            4. 将加载的模型赋值给实例的 `self.model` 属性。
            5. 记录日志，显示加载的模型的输入形状信息。

        异常处理:
            - 如果 `model_path` 不是有效目录路径，抛出 `AssertionError` 异常并提示模型文件不存在。

        返回:
            None

        备注:
            - 该方法假设模型路径是有效的，并且路径中包含可以加载的 Keras 模型。
            - `overwrite` 参数目前没有实际作用。
        """

        logger.debug(f"Keras sequence model load from {model_path}")

        assert os.path.isdir(model_path), f"model file {model_path} not existed"

        self.model = keras.models.load_model(model_path)
        logger.debug(f"Keras sequence model load data {self.model.input_shape}")

    def create_model(self, follow_tf_size: tuple, model_aisle: int) -> "keras.Sequential":
        """
        方法: `create_model`

        功能:
            创建一个 Keras 序列模型（`Sequential`），该模型包含多个卷积层、池化层、全连接层和 Dropout 层，用于图像分类任务。

        参数:
            follow_tf_size (tuple): 输入图像的尺寸，通常为宽和高的组合（例如 `(224, 224)`）。
            model_aisle (int): 输入图像的通道数，例如 3 表示 RGB 图像，1 表示灰度图像。

        操作流程:
            1. 记录日志，指示开始创建 Keras 序列模型。
            2. 根据 `keras.backend.image_data_format()` 的结果，确定输入图像的形状 (`input_shape`)：
               - 如果数据格式为 `channels_first`，则形状为 `(channels, height, width)`。
               - 否则形状为 `(height, width, channels)`。
            3. 创建一个 `Sequential` 模型实例。
            4. 按顺序向模型添加以下层：
               - 卷积层 (`Conv2D`)：过滤器数为 32，核大小为 (3, 3)，填充方式为 "same"。
               - 最大池化层 (`MaxPooling2D`)：池化窗口大小为 (2, 2)。
               - Dropout 层 (`Dropout`)：Dropout 率为 0.25。
               - 第二个卷积层：过滤器数为 64，同样的核大小和填充方式。
               - 第二个最大池化层和 Dropout 层。
               - 第三个卷积层：过滤器数为 128，同样的核大小和填充方式。
               - 第三个最大池化层和 Dropout 层。
               - 展平层 (`Flatten`)：将多维特征图展平为一维。
               - 全连接层 (`Dense`)：神经元数为 256，激活函数为 ReLU。
               - Dropout 层：Dropout 率为 0.5。
               - 输出层 (`Dense`)：神经元数由 `self.MODEL_DENSE` 决定，激活函数为 Softmax。
            5. 编译模型，使用 Adam 优化器、稀疏分类交叉熵 (`sparse_categorical_crossentropy`) 作为损失函数，评估指标为准确率。
            6. 记录日志，指示模型创建完成。
            7. 返回创建的模型实例。

        返回:
            keras.Sequential: 创建好的 Keras 序列模型实例。

        异常处理:
            - 该方法没有显式的异常处理，假定所有操作都能成功执行。

        备注:
            - 此模型主要用于图像分类任务，适用于小型到中型的数据集。
            - `self.MODEL_DENSE` 应包含最终分类类别的数量。
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

    def train(self, data_path: str = None, *args, **kwargs) -> None:
        """
        方法: `train`

        功能:
            训练 Keras 序列模型，使用图像数据生成器从目录中加载训练和验证数据集，并进行模型训练。

        参数:
            data_path (str): 图像数据集所在的目录路径。该目录应包含多个子文件夹，每个子文件夹代表一个类别。
            *args: 额外的参数，通常用于指定模型的颜色模式、输入图像尺寸和通道数。
            **kwargs: 额外的关键字参数，允许向模型训练过程中传递其他配置选项。

        操作流程:
            1. 定义 `check` 内部函数用于验证数据集路径的有效性：
               - 将路径转换为 `pathlib.Path` 对象。
               - 验证路径是否为目录，否则抛出断言错误。
               - 统计目录中子文件夹的数量，并验证是否大于 1（表示有多个类别）。
               - 验证类别数量不超过模型输出层的神经元数量（`self.MODEL_DENSE`），否则抛出断言错误。
            2. 调用 `check` 函数检查 `data_path` 的有效性。
            3. 从 `args` 中解包出模型的颜色模式、输入图像尺寸、通道数等信息。
            4. 如果尚未创建模型，则调用 `create_model` 方法创建一个新的模型。
            5. 使用 `ImageDataGenerator` 定义图像数据生成器 (`datagen`)，包括数据增强和归一化操作：
               - `rescale`: 对图像进行缩放。
               - `shear_range`: 应用剪切变换。
               - `zoom_range`: 应用缩放变换。
               - `validation_split`: 指定训练集和验证集的分割比例。
               - `horizontal_flip`: 随机水平翻转图像。
            6. 使用 `flow_from_directory` 方法从 `data_path` 目录加载训练数据 (`train_generator`) 和验证数据 (`validation_generator`)：
               - `target_size`: 目标图像尺寸。
               - `batch_size`: 批量大小。
               - `color_mode`: 图像颜色模式（如 RGB 或灰度）。
               - `class_mode`: 分类模式（`sparse` 表示稀疏分类）。
               - `subset`: 指定使用数据集的哪个部分（训练或验证）。
            7. 调用模型的 `fit` 方法，使用生成器进行模型训练：
               - `epochs`: 训练的轮数。
               - `validation_data`: 验证数据生成器。
            8. 记录日志，指示模型训练已完成。

        返回:
            None

        异常处理:
            - 该方法通过断言确保数据路径的有效性，抛出 `AssertionError` 用于不符合条件的路径。
            - 其他异常未显式捕获，假定由外部调用者处理。

        备注:
            - 此方法假定数据集目录中每个子文件夹对应一个类别，并要求类别数量符合模型输出层的配置。
            - 图像数据生成器用于在训练过程中对数据进行增强，以提高模型的泛化能力。
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
        方法: `build`

        功能:
            构建和训练 Keras 模型，并将训练后的模型保存到指定路径。

        参数:
            model_color (str): 模型的颜色模式，例如 "rgb" 或 "grayscale"。
            model_shape (tuple): 模型的输入图像尺寸，格式为 (宽度, 高度)。
            model_aisle (int): 输入图像的通道数，例如 3 对于 RGB 图像。
            *args: 额外的参数，包括源模型路径 (`src_model_path`)、新模型保存路径 (`new_model_path`)、新模型名称 (`new_model_name`)。

        操作流程:
            1. 将 `model_shape` 转换为 TensorFlow 兼容的尺寸 `follow_tf_size`。
            2. 调用 `train` 方法以训练模型，传入源模型路径、颜色模式、输入图像尺寸和通道数。
               - 如果训练过程中抛出 `AssertionError`，捕获异常并记录错误信息，方法返回 `None`。
            3. 定义模型的最终保存路径 `final_model`，使用 `os.makedirs` 确保保存目录存在。
            4. 保存训练后的模型：
               - `self.model.save(final_model, save_format="tf")`: 将模型保存为 TensorFlow 格式。
            5. 调用 `self.model.summary()` 打印模型结构的摘要信息。
            6. 返回最终模型的保存路径 `final_model`。

        返回:
            typing.Optional[str]: 返回保存模型的路径 `final_model`，如果训练失败，返回 `None`。

        异常处理:
            - 捕获并处理训练过程中的 `AssertionError`，记录错误并终止方法执行。

        备注:
            - 此方法假定在训练数据准备充分的情况下进行模型构建和保存。
            - `save_format="tf"` 用于保存模型为 TensorFlow 格式文件。
            - 模型结构信息通过 `summary()` 方法输出到控制台。
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
        方法: `predict`

        功能:
            该方法接收图像路径，将图像加载为数据对象后，对其应用模型预测，返回分类结果。

        参数:
            - pic_path (str): 图像文件的路径。
            - *args: 额外的参数，可用于定制图像预处理或钩子函数。
            - **kwargs: 关键字参数，用于进一步自定义图像处理过程。

        操作流程:
            1. 调用 `toolbox.imread` 方法读取图像文件，将其转换为矩阵表示的图像数据，存储在 `picture` 变量中。
            2. 创建一个 `VideoFrame` 对象 `fake_frame`，模拟视频帧数据，其中包括帧 ID 为 `0`、时间戳为 `0.0`，以及加载的图像数据。
            3. 调用 `_apply_hook` 方法，将 `fake_frame` 对象传入，以便根据传入的 `*args` 和 `**kwargs` 对图像进行自定义处理。
            4. 使用 `predict_with_object` 方法，将预处理后的图像数据传入模型，进行预测。
            5. 返回预测结果字符串。

        返回:
            str: 返回模型预测结果的字符串形式。

        异常处理:
            - 如果读取图像或模型预测过程中出现问题，需确保异常处理合理并返回有意义的错误信息。
        """

        picture = toolbox.imread(pic_path)
        fake_frame = VideoFrame(0, 0.0, picture)
        fake_frame = self._apply_hook(fake_frame, *args, **kwargs)
        return self.predict_with_object(fake_frame.data)

    def predict_with_object(self, frame: numpy.ndarray) -> str:
        """
        方法: `predict_with_object`

        功能:
            该方法接收一个图像帧对象，并对其进行预处理和模型预测，返回预测的类别标签字符串。

        参数:
            - frame (numpy.ndarray): 输入的图像帧数据，以 NumPy 数组的形式表示。

        操作流程:
            1. 使用 `cv2.resize` 方法将输入图像帧 `frame` 调整为模型的输入尺寸 (`self.follow_cv_size`)。
            2. 调用 `numpy.expand_dims` 方法为图像数据增加额外的维度，以符合模型的输入要求。
            3. 使用 `self.model.predict` 方法进行预测，得到每个类别的概率分布，存储在 `frame_result` 变量中。
            4. 通过 `numpy.argmax` 方法获取概率最高的类别标签，转换为字符串形式存储在 `frame_tag` 变量中。
            5. 获取最高概率值 `frame_confidence`，并与阈值 `self.score_threshold` 进行比较。
            6. 如果最高概率值低于阈值，则记录警告日志并返回表示未知类别的标志 (`const.UNKNOWN_STAGE_FLAG`)。
            7. 否则，返回预测的类别标签 `frame_tag`。

        返回:
            str: 返回预测的类别标签。如果置信度低于阈值，返回 `UNKNOWN_STAGE_FLAG`。

        异常处理:
            - 在处理图像或进行模型预测时，如果发生错误，需确保异常处理合理并返回有意义的错误信息。
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
        方法: `_classify_frame`

        功能:
            该方法接收一个 `VideoFrame` 对象，并对其进行分类操作，返回预测的类别标签字符串。

        参数:
            - frame (VideoFrame): 输入的视频帧对象，包含图像数据及其相关信息。
            - *_, **__: 忽略位置参数和关键字参数。

        操作流程:
            1. 调用 `predict_with_object` 方法，将视频帧对象中的图像数据 `frame.data` 传递给模型进行分类预测。
            2. 获取并返回预测的类别标签字符串。

        返回:
            str: 返回预测的类别标签。

        异常处理:
            - 确保传入的 `VideoFrame` 对象是有效的，并包含必要的图像数据以供预测。
            - 在处理图像数据或进行模型预测时，如果发生错误，需确保异常处理合理并返回有意义的错误信息。
        """

        return self.predict_with_object(frame.data)


if __name__ == '__main__':
    pass
