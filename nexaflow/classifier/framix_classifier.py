import os
import pathlib
from loguru import logger
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FramixClassifier(object):

    MODEL_DENSE = 6

    def __init__(
            self,
            color: str = None,
            aisle: int = None,
            data_size: tuple = None,
            batch_size: int = None,
            epochs: int = None
    ):

        self.color: str = color or "grayscale"
        self.aisle: int = aisle or 1
        self.data_size: tuple = data_size or (200, 200)
        self.batch_size: int = batch_size or 4
        self.epochs: int = epochs or 20

    @property
    def follow_tf_size(self):
        return self.data_size[1], self.data_size[0]

    @property
    def follow_cv_size(self):
        return self.data_size[0], self.data_size[1]

    def create_model(self) -> keras.Sequential:

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

        return model

    def train(self, data_path: str = None):

        p = pathlib.Path(data_path)
        assert p.is_dir(), f"{p} is not a valid directory"

        number_of_dir = len([each for each in os.listdir(p) if (p / each).is_dir()])
        assert (
            number_of_dir > 1
        ), f"dataset only contains one class. maybe some path errors happened: {p}?"

        assert number_of_dir <= self.MODEL_DENSE, (
            f"dataset has {number_of_dir} classes (more than " + str(self.MODEL_DENSE) + ")"
        )

        model = self.create_model()

        logger.info("开始训练模型 ...")

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
            color_mode=self.color,
            class_mode="sparse",
            subset="training",
        )

        validation_generator = datagen.flow_from_directory(
            data_path,
            target_size=self.follow_tf_size,
            batch_size=self.batch_size,
            color_mode=self.color,
            class_mode="sparse",
            subset="validation",
        )

        model.fit(
            train_generator,
            epochs=self.epochs,
            validation_data=validation_generator
        )

        logger.info("模型训练完成 ...")
        return model

    def build(self, *args):
        src, new_model_path, new_model_name = args
        try:
            model = self.train(src)
        except AssertionError as e:
            logger.error(e)
        else:
            final_model = os.path.join(new_model_path, new_model_name)
            os.makedirs(new_model_path, exist_ok=True)
            model.save_weights(final_model, save_format="h5")
            model.summary()
            logger.info(f"模型保存完成 {final_model}")


if __name__ == '__main__':
    pass
