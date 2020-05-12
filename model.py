from tensorflow import keras
import cv2
import numpy as np
import tensorflow as tf
import string

new_model = keras.models.load_model("mymodel.h5")

str_charts = string.digits + string.ascii_letters  # 验证码里的所有字符


def getchat(index):
    return str_charts[index]


def test_train(file):
    im = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    im = im / 255.0
    img = np.expand_dims(im, 0)
    predictions_single = new_model.predict(img)
    _label = tf.math.argmax(predictions_single, axis=-1)
    label = "".join(map(getchat, _label[0]))
    return label


if __name__ == "__main__":
    pass
