import os
import cv2
import typing
import pathlib
import numpy as np
from loguru import logger
from nexaflow import toolbox, const
from nexaflow.video import VideoFrame
from nexaflow.classifier.base import BaseModelClassifier

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow
except ImportError:
    raise ImportError("KerasClassifier requires tensorflow. install it first.")

from tensorflow import keras


class KerasClassifier(BaseModelClassifier):

    UNKNOWN_STAGE_NAME = const.UNKNOWN_STAGE_FLAG
    MODEL_DENSE = 6

    def __init__(self, *_, **kwargs):
        super(KerasClassifier, self).__init__(*_, **kwargs)

        # 模型
        self._model: typing.Optional[keras.Sequential] = None
        # 配置
        self.aisle: int = kwargs.get("aisle", 1)
        self.score_threshold: float = kwargs.get("score_threshold", 0.0)
        self.data_size: typing.Sequence[int] = kwargs.get("data_size", (256, 256))
        self.nb_train_samples: int = kwargs.get("nb_train_samples", 64)
        self.nb_validation_samples: int = kwargs.get("nb_validation_samples", 64)
        self.epochs: int = kwargs.get("epochs", 20)
        self.batch_size: int = kwargs.get("batch_size", 4)

        # logger.debug(f"score threshold: {self.score_threshold}")
        # logger.debug(f"data size: {self.data_size}")
        # logger.debug(f"nb train samples: {self.nb_train_samples}")
        # logger.debug(f"nb validation samples: {self.nb_validation_samples}")
        # logger.debug(f"epochs: {self.epochs}")
        # logger.debug(f"batch size: {self.batch_size}")

    @property
    def follow_tf_size(self):
        return self.data_size[1], self.data_size[0]

    @property
    def follow_cv_size(self):
        return self.data_size[0], self.data_size[1]

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

    def create_model(self) -> keras.Sequential:
        # logger.info(f"creating Keras sequential model")
        logger.info("Keras神经网络引擎创建图像分析模型 ...")
        if keras.backend.image_data_format() == "channels_first":
            input_shape = (self.aisle, *self.follow_tf_size)
        else:
            input_shape = (*self.follow_tf_size, self.aisle)

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

        # logger.info("Keras model created")
        logger.info("Keras神经网络引擎加载完成，开始分析图像 ...")
        return model

    def train(self, data_path: str = None, *_, **__):

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

        datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 16,
            shear_range=0.2,
            zoom_range=0.2,
            validation_split=0.33,
            horizontal_flip=True  # 水平翻转增强
        )

        train_generator = datagen.flow_from_directory(
            data_path,
            target_size=self.follow_tf_size,
            batch_size=self.batch_size,
            color_mode="grayscale",
            class_mode="sparse",
            subset="training",
        )

        validation_generator = datagen.flow_from_directory(
            data_path,
            target_size=self.follow_tf_size,
            batch_size=self.batch_size,
            color_mode="grayscale",
            class_mode="sparse",
            subset="validation",
        )

        self._model.fit(
            train_generator,
            epochs=self.epochs,
            validation_data=validation_generator,
        )

        logger.debug("train finished")

    def predict(self, pic_path: str, *args, **kwargs) -> str:
        pic_object = toolbox.imread(pic_path)
        # fake VideoFrame for apply_hook
        fake_frame = VideoFrame(0, 0.0, pic_object)
        fake_frame = self._apply_hook(fake_frame, *args, **kwargs)
        return self.predict_with_object(fake_frame.data)

    def predict_with_object(self, frame: np.ndarray) -> str:
        # resize for model
        frame = cv2.resize(frame, dsize=self.follow_cv_size)
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
