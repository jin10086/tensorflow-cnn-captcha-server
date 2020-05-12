import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalMaxPool2D,
    Input,
    MaxPool2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from config import (
    C,
    H,
    W,
    checkpoint_dir,
    checkpoint_path,
    D,
    N_LABELS,
    str_charts,
    model_file_name,
    accuracy_rate,
)

from process_data import get_train_data


def get_data_generator(df, indices, for_training, batch_size=16):
    images, labels = [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, label = r["file"], r["label"]
            im = cv2.imread(file, cv2.IMREAD_UNCHANGED)

            # 图片统一大小
            if im.shape != (H, W, C):
                im = cv2.resize(im, (W, H), cv2.INTER_AREA)
                print("resize image", label)
                cv2.imwrite(file, im)

            im = im / 255.0
            images.append(im)
            labels.append(
                np.array(
                    [
                        np.array(to_categorical(str_charts.find(i), N_LABELS))
                        for i in label
                    ]
                )
            )
            if len(images) >= batch_size:
                #                 print(np.array(images), np.array(labels))
                yield np.array(images), np.array(labels)
                images, labels = [], []
        if not for_training:
            break


cp_callback = ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1
)


def get_models():
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    input_layer = tf.keras.Input(shape=(H, W, C))
    x = layers.Conv2D(32, 3, activation="relu")(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(D * N_LABELS, activation="softmax")(x)
    x = layers.Reshape((D, N_LABELS))(x)

    model = models.Model(inputs=input_layer, outputs=x)

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    if latest:
        model.load_weights(checkpoint_path)
    print(model.summary())
    return model


def train(df, train_idx, valid_idx, model):
    batch_size = 64
    valid_batch_size = 64
    train_gen = get_data_generator(
        df, train_idx, for_training=True, batch_size=batch_size
    )
    valid_gen = get_data_generator(
        df, valid_idx, for_training=True, batch_size=valid_batch_size
    )

    accuracy = 0
    while True:
        if accuracy > accuracy_rate:
            model.save(model_file_name)
            break
        # with tf.device("/device:GPU:0"):
        history = model.fit(
            train_gen,
            steps_per_epoch=len(train_idx) // batch_size,
            epochs=5,
            callbacks=[cp_callback],
            validation_data=valid_gen,
            validation_steps=len(valid_idx) // valid_batch_size,
        )
        # 拿到本次训练结束后的准确度
        accuracy = history.history["accuracy"][-1]
        # 保存模型
        model.save(model_file_name)


def main():
    train_idx, valid_idx, df = get_train_data()
    model = get_models()
    train(df, train_idx, valid_idx, model)


if __name__ == "__main__":
    main()
