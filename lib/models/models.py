from keras.applications import resnet50
from keras.applications.mobilenet import MobileNet
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense
from ..config import SIZE_INPUT


img_rows, img_cols, img_channel = SIZE_INPUT, SIZE_INPUT, 3


def create_resnet50():
    base_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))
    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dense(340, activation='softmax'))

    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    return model


def create_mobilenet():
    model = MobileNet(input_shape=(img_rows, img_cols, 3), alpha=1., weights=None, classes=340)


    return model

