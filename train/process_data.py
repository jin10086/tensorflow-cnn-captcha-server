import glob
import os
import numpy as np
import pandas as pd

from config import DATA_DIR


def parse_filepath(filepath):
    try:
        path, filename = os.path.split(filepath)
        filename, ext = os.path.splitext(filename)
        label, _ = filename.split("_")
        return label
    except Exception as e:
        print("error to parse %s. %s" % (filepath, e))
        return None, None


def get_images():
    files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))  # 目录下所有的.jpg文件
    attributes = list(map(parse_filepath, files))
    df = pd.DataFrame(attributes)
    df["file"] = files
    df.columns = ["label", "file"]
    df = df.dropna()
    return df


def getchat(index):
    return str_charts[index]


def get_train_data():
    df = get_images()
    p = np.random.permutation(len(df))
    train_up_to = int(len(df) * 0.7)
    train_idx = p[:train_up_to]
    test_idx = p[train_up_to:]

    # split train_idx further into training and validation set
    train_up_to = int(train_up_to * 0.7)
    train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]

    print(
        "train count: %s, valid count: %s, test count: %s"
        % (len(train_idx), len(valid_idx), len(test_idx))
    )
    return train_idx, valid_idx, df
