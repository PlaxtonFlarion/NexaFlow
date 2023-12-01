import os
import cv2
import typing
import pathlib
import numpy as np
from loguru import logger
from nexaflow import toolbox, constants
from nexaflow.video import VideoFrame
from nexaflow.classifier.base import BaseModelClassifier

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    import tensorflow
    tensorflow.get_logger().setLevel("ERROR")
except ImportError:
    raise ImportError("KerasClassifier requires tensorflow. install it first.")

from keras import backend
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.layers import BatchNormalization, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class KerasClassifier(BaseModelClassifier):

    UNKNOWN_STAGE_NAME = constants.UNKNOWN_STAGE_FLAG
    MODEL_DENSE = 6

    def __init__(
        self,
        score_threshold: float = None,
        data_size: typing.Sequence[int] = None,
        nb_train_samples: int = None,
        nb_validation_samples: int = None,
        epochs: int = None,
        batch_size: int = None,
        *_,
        **__,
    ):
        super(KerasClassifier, self).__init__(*_, **__)

        # 模型
        self._model: typing.Optional[Sequential] = None
        # 配置
        self.score_threshold: float = score_threshold or 0.0
        self.data_size: typing.Sequence[int] = data_size or (200, 200)
        self.nb_train_samples: int = nb_train_samples or 64
        self.nb_validation_samples: int = nb_validation_samples or 64
        self.epochs: int = epochs or 20
        self.batch_size: int = batch_size or 4

        # logger.debug(f"score threshold: {self.score_threshold}")
        # logger.debug(f"data size: {self.data_size}")
        # logger.debug(f"nb train samples: {self.nb_train_samples}")
        # logger.debug(f"nb validation samples: {self.nb_validation_samples}")
        # logger.debug(f"epochs: {self.epochs}")
        # logger.debug(f"batch size: {self.batch_size}")

    def clean_model(self):
        self._model = None

    def save_model(self, model_path: str, overwrite: bool = None):
        logger.debug(f"save model to {model_path}")
        # assert model file
        if os.path.isfile(model_path) and not overwrite:
            raise FileExistsError(
                f"model file {model_path} already existed, you can set `overwrite` True to cover it"
            )
        # assert model data is not empty
        assert self._model, "model is empty"
        print(self._model.summary())
        self._model.save_weights(model_path)

    def load_model(self, model_path: str, overwrite: bool = None):
        # logger.debug(f"load model from {model_path}")
        logger.info(f"加载Keras神经网络引擎 ...")
        # assert model file
        assert os.path.isfile(model_path), f"model file {model_path} not existed"
        # assert model data is empty
        if self._model and not overwrite:
            raise RuntimeError(
                f"model is not empty, you can set `overwrite` True to cover it"
            )
        self._model = self.create_model()
        self._model.load_weights(model_path)

    def create_model(self) -> Sequential:
        # logger.info(f"creating Keras sequential model")
        logger.info("Keras神经网络引擎创建图像分析模型 ...")
        if backend.image_data_format() == "channels_first":
            input_shape = (1, *self.data_size)
        else:
            input_shape = (*self.data_size, 1)

        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.5))
        model.add(Dense(self.MODEL_DENSE, activation='softmax'))

        model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # logger.info("Keras model created")
        logger.info("Keras神经网络引擎加载完成，开始分析图像 ...")
        return model

    def train(self, data_path: str = None, final_model_path: str = None, *_, **__):

        def _data_verify(p: str):
            p = pathlib.Path(p)
            assert p.is_dir(), f"{p} is not a valid directory"

            number_of_dir = len([each for each in os.listdir(p) if (p / each).is_dir()])
            assert (
                number_of_dir > 1
            ), f"dataset only contains one class. maybe some path errors happened: {p}?"

            assert number_of_dir <= self.MODEL_DENSE, (
                f"dataset has {number_of_dir} classes (more than " + str(self.MODEL_DENSE) + ")"
            )

        _data_verify(data_path)

        if not self._model:
            self._model = self.create_model()

        datagen = ImageDataGenerator(
            rescale=1.0 / 16,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,  # 水平翻转增强
            validation_split=0.33
        )

        train_generator = datagen.flow_from_directory(
            data_path,
            target_size=self.data_size,
            batch_size=self.batch_size,
            color_mode="grayscale",
            class_mode="sparse",
            subset="training",
        )

        validation_generator = datagen.flow_from_directory(
            data_path,
            target_size=self.data_size,
            batch_size=self.batch_size,
            color_mode="grayscale",
            class_mode="sparse",
            subset="validation",
        )

        # 早停
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1
        )

        # 模型检查点
        model_checkpoint = ModelCheckpoint(
            final_model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )

        # 动态学习率调整
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-5,
            verbose=1
        )

        self._model.fit(
            train_generator,
            epochs=self.epochs,
            validation_data=validation_generator,
            callbacks=[early_stopping, model_checkpoint, reduce_lr]  # 新增：回调列表
        )

        print(self._model.summary())
        logger.debug("train finished")

    def predict(self, pic_path: str, *args, **kwargs) -> str:
        pic_object = toolbox.imread(pic_path)
        # fake VideoFrame for apply_hook
        fake_frame = VideoFrame(0, 0.0, pic_object)
        fake_frame = self._apply_hook(fake_frame, *args, **kwargs)
        return self.predict_with_object(fake_frame.data)

    def predict_with_object(self, frame: np.ndarray) -> str:
        # resize for model
        frame = cv2.resize(frame, dsize=self.data_size)
        frame = np.expand_dims(frame, axis=[0, -1])
        # verbose = 0, 静默Keras分类显示
        result = self._model.predict(frame, verbose=0)
        tag = str(np.argmax(result, axis=1)[0])
        confidence = result.max()
        # logger.debug(f"confidence: {confidence}")
        if confidence < self.score_threshold:
            logger.warning(
                f"max score is lower than {self.score_threshold}, unknown class"
            )
            return self.UNKNOWN_STAGE_NAME
        return tag

    def _classify_frame(self, frame: VideoFrame, *_, **__) -> str:
        return self.predict_with_object(frame.data)


if __name__ == '__main__':
    pass
