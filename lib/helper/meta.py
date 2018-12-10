import os
from ..config import DIR_TRAIN_IMG_RAW


def get_file_names(my_dir):
    files = os.listdir(my_dir)
    files = [f for f in files if ".csv" in f]
    return files


def create_label_mapping(reverse=False):
    files = get_file_names(my_dir=DIR_TRAIN_IMG_RAW)
    labels = [f[:-4] for f in files]
    labels.sort()
    mapping = dict([(l, i) for i, l in enumerate(labels)])
    if reverse:
        mapping = dict([(i, l) for i, l in enumerate(labels)])
    return mapping
