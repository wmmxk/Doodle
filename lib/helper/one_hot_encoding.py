import torch
import keras

def one_hot_encoding_tensor(y, num_class):
    batch_size = y.shape[0]
    y_onehot = torch.FloatTensor(batch_size, num_class)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot


def one_hot_keras():
    label = keras.utils.to_categorical(1, num_classes=340)
    print(label)