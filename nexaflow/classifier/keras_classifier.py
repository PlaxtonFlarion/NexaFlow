import os
import cv2
import numpy
import typing
import pathlib

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
    def model(self):
        return self.__model

    @model.setter
    def model(self, value):
        self.__model = value

    @property
    def follow_cv_size(self):
        return self.model.input_shape[1], self.model.input_shape[2]

    def load_model(self, model_path: str, overwrite: bool = None):
        logger.debug(f"Keras sequence model load from {model_path}")

        assert os.path.isdir(model_path), f"model file {model_path} not existed"

        self.model = keras.models.load_model(model_path)
        logger.debug(f"Keras sequence model load data {self.model.input_shape}")

    def create_model(self, follow_tf_size: tuple, model_aisle: int) -> "keras.Sequential":
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

    def train(self, data_path: str = None, *args, **kwargs):

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
        src_model_path, new_model_path, new_model_name = args

        follow_tf_size = model_shape[1], model_shape[0]

        try:
            self.train(
                src_model_path, model_color, follow_tf_size, model_aisle
            )
        except AssertionError as e:
            return logger.error(e)

        final_model: str = os.path.join(new_model_path, new_model_name)
        os.makedirs(new_model_path, exist_ok=True)

        # self.model.save_weights(final_model, save_format="h5")
        self.model.save(final_model, save_format="tf")
        self.model.summary()

        return final_model

    def predict(self, pic_path: str, *args, **kwargs) -> str:
        picture = toolbox.imread(pic_path)
        fake_frame = VideoFrame(0, 0.0, picture)
        fake_frame = self._apply_hook(fake_frame, *args, **kwargs)
        return self.predict_with_object(fake_frame.data)

    def predict_with_object(self, frame: numpy.ndarray) -> str:
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
        return self.predict_with_object(frame.data)


if __name__ == '__main__':
    pass
