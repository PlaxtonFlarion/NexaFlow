import os
import sys
import pathlib
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FramixClassifier(object):

    MODEL_DENSE = 6

    def __init__(self):
        self.model = None
        self.data_size = (100, 100)
        self.batch_size = 4
        self.epochs = 20

    def create_model(self) -> keras.Sequential:

        if keras.backend.image_data_format() == "channels_first":
            input_shape = (1, *self.data_size)
        else:
            input_shape = (*self.data_size, 1)

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

        if not self.model:
            self.model = self.create_model()

        datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 16,
            shear_range=0.2,
            zoom_range=0.2,
            validation_split=0.33,
            horizontal_flip=True  # 水平翻转增强
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

        self.model.fit(
            train_generator,
            epochs=self.epochs,
            validation_data=validation_generator
        )

    def save_model(self, model_path: str):
        assert self.model, "model is empty"
        self.model.save_weights(model_path, save_format="h5")
        self.model.summary()

    def load_model(self, model_path: str, overwrite: bool = None):
        if self.model and not overwrite:
            raise RuntimeError(
                f"model is not empty, you can set `overwrite` True to cover it"
            )
        self.model = self.create_model()
        self.model.load_weights(model_path)

    def __call__(self, *args, **kwargs):
        src, dst = args
        self.train(src)
        self.save_model(dst)


if __name__ == '__main__':
    fc = FramixClassifier()
    fc(sys.argv[1], sys.argv[2])
