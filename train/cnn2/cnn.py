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
    model_file_name,
    accuracy_rate,
)

from process_data import get_train_data


def get_data_generator(df, indices, str_charts, N_LABELS, for_training, batch_size=16):
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
            labels.append(to_categorical(str_charts.index(label), N_LABELS))
            if len(images) >= batch_size:
                #                 print(np.array(images), np.array(labels))
                yield np.array(images), np.array(labels)
                images, labels = [], []
        if not for_training:
            break


cp_callback = ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1
)


def get_models(N_LABELS):
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(H, W, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(N_LABELS, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=["accuracy"],
    )

    if latest:
        model.load_weights(checkpoint_path)
    print(model.summary())
    return model


def train(df, train_idx, valid_idx, model, str_charts, N_LABELS):
    batch_size = 64
    valid_batch_size = 64
    train_gen = get_data_generator(
        df, train_idx, str_charts, N_LABELS, for_training=True, batch_size=batch_size
    )
    valid_gen = get_data_generator(
        df,
        valid_idx,
        str_charts,
        N_LABELS,
        for_training=True,
        batch_size=valid_batch_size,
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
    # 拿到总的分类
    str_charts = list(set(df.label))
    str_charts.sort()
    N_LABELS = len(str_charts)
    model = get_models(N_LABELS)
    train(df, train_idx, valid_idx, model, str_charts, N_LABELS)


if __name__ == "__main__":
    main()
