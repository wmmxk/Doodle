import numpy as np
import cv2
import ast
import matplotlib.pyplot as plt
from ..config import SIZE_CANVAS
from ..config import SIZE_INPUT
from ..config import MARGIN
from ..config import SHIFT
from ..config import MEAN, STD


def visualize_img(img):
    plt.imshow(img)
    plt.show()


def _adjust_boundary(left, right, top, bottom):
    """
    :param left: int
    :param right: int
    :param top: int
    :param bottom: int
    :return: a tuple of 4 int
    """
    return left + SHIFT - MARGIN, right + SHIFT + MARGIN, top + SHIFT - MARGIN, bottom + +SHIFT + MARGIN


def pad_arr(arr):
    h, w = arr.shape
    margin_total = (max(h, w) - min(h, w))
    margin_1 = margin_total // 2
    margin_2 = margin_total - margin_1
    if h > w:
        out = np.pad(arr, ((0, 0), (margin_1, margin_2)), 'constant', constant_values=(0, 0))
    else:
        out = np.pad(arr, ((margin_1, margin_2), (0, 0)), 'constant', constant_values=(0, 0))
    return out


def retrieve_one_image(strokes, lw=6, time_color=True):
    """
    :param strokes: [ [[],[]], [[],[]] ]
    :param lw: int
    :param time_color: boolean
    :return: a numpy array
    """
    img = np.zeros((SIZE_CANVAS, SIZE_CANVAS), np.uint8)
    left, top = SIZE_CANVAS, SIZE_CANVAS
    right, bottom = 0, 0
    for t, stroke in enumerate(strokes):
        left = min(left, min(stroke[0]))
        right = max(right, max(stroke[0]))
        top = min(top, min(stroke[1]))
        bottom = max(bottom, max(stroke[1]))
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img,
                         (stroke[0][i] + SHIFT, stroke[1][i] + SHIFT),
                         (stroke[0][i+1] + SHIFT, stroke[1][i+1] + SHIFT),
                         color,
                         lw)
    # check whether the figure is larger than the canvas
    assert right < SIZE_CANVAS and bottom < SIZE_CANVAS
    left, right, top, bottom = _adjust_boundary(left, right, top, bottom)
    img = img[top:bottom, left:right]
    img = pad_arr(img)
    img = cv2.resize(img, (SIZE_INPUT, SIZE_INPUT))/255.0
    return np.repeat(img[:, :, np.newaxis], 3, axis=2)


def fetch_from_one_row(one_row, mapping=None, train=True):
    strokes = ast.literal_eval(one_row['drawing'])
    if train:
        try:
            img = retrieve_one_image(strokes)
            # when you compute the channel mean, you don't care about the label
            if mapping is not None:
                label = mapping[one_row['word']]
            else:
                label = -1
        except:
            img = np.zeros((SIZE_INPUT, SIZE_INPUT), np.uint8)
            label = -1 # if fail at retrieving, return -1 and filter this out when training
    else:
        img = retrieve_one_image(strokes)
        label = one_row['key_id']
        img = img.astype(np.float32)
    return (img-MEAN)/STD, label