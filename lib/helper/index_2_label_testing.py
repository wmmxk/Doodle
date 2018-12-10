import os
import pandas as pd
import numpy as np
from .meta import create_label_mapping
from ..config import LOG_NAME
from ..config import DIR_TEST_IMG
from ..config import FILE_TEST
from ..config import SUB_TEMPLATE
from ..config import DIR_SUB


def make_submission():
    path_test_file = os.path.join(DIR_TEST_IMG, FILE_TEST)
    file_test = pd.read_csv(path_test_file)
    file_sub_template = pd.read_csv(os.path.join(DIR_TEST_IMG, SUB_TEMPLATE))
    #os.remove(path_test_file)
    pred = pd.read_csv(os.path.join(DIR_SUB, LOG_NAME + "_prob.csv"))
    labels = np.argsort(-pred.values, axis=1)[:, :3]
    labels = pd.DataFrame(labels)
    mapping = create_label_mapping(reverse=True)
    labels = labels.apply(lambda col: col.map(mapping), axis=0)
    labels = labels.replace(' ', '_', regex=True)
    label_series = labels.apply(lambda row: " ".join(row), axis=1)

    file_sub_template['word'] = label_series
    file_sub_template.to_csv(os.path.join(DIR_SUB, LOG_NAME + "_sub.csv"), index=False)
    print(label_series[:3])
    print(file_sub_template.head())

    #  TODO: Nov 20 reformat according to the submission template
