import pandas as pd
import os
import numpy as np
from .dataset_h import fetch_from_one_row
from ..config import DIR_TRAIN_IMG
from sklearn.utils import shuffle


def get_channel_mean(name_csv):
    df = pd.read_csv(os.path.join(DIR_TRAIN_IMG, name_csv))
    imgs = []
    df = shuffle(df)
    for i in range(df.shape[0]):
        img, label = fetch_from_one_row(df.iloc[i, :])
        imgs.append(img)
        if i > 10000:
            break
    imgs = np.array(imgs)
    print("mean: ", np.mean(img))
    print("mean: ", np.std(img))
    return img, label