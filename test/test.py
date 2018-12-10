import pytest
import torch
import numpy as np
from torch.autograd import Variable
from lib.helper.dataset_h import visualize_img
from lib.dataloader.my_dataset import MyDataset
from train import Train
from torch.utils.data import DataLoader
from lib.config import POOLING_SIZE
from lib.helper.index_2_label_testing import make_submission
from lib.dataloader.custom_transform import MultiCompose
from lib.config import FILE_TEST
from lib.dataloader.my_dataset import create_loader
from lib.dataloader.my_dataset import create_generator
from lib.config import CONFIG
from lib.config import FILE_PYTEST


@pytest.fixture(scope="module")
def create_dataset():
    dataset = MyDataset(FILE_PYTEST, transform=MultiCompose())
    return dataset


def test_mydataset(create_dataset):
    dataset = create_dataset
    for i, (img, label) in enumerate(dataset):
        print("max: ", torch.mean(img))
        print(i, img.shape)
        visualize_img(img)
        print(label)
        if i > 4:
            break


def test_transform(create_dataset, transform=MultiCompose):
    dataset = create_dataset
    for i, (img, label) in enumerate(dataset):
        visualize_img(img)
        print(img.shape)
        out = transform()(img.numpy())
        visualize_img(out)
        print(out.shape)
        if i > 2:
            break


def test_test_dataset():
    dataset = MyDataset(FILE_TEST, transform=None)
    for i, (img, label) in enumerate(dataset):
        assert 3 == len(img.shape)
        break
    dataloader = DataLoader(dataset, shuffle=False, num_workers=4, pin_memory=True, batch_size=64)
    for i, img in enumerate(dataloader):
        print(i, " ", img[0].shape)
        break


def test_generator(name_csv=FILE_PYTEST):
    generator_tr = create_generator(name_csv, batch_size=CONFIG.batch_size)
    imgs, labels = next(generator_tr)
    print(imgs.shape)
    print(labels.shape)
    print(np.argmax(labels, axis=1))
    print(labels)
    for i, img in enumerate(imgs):
        visualize_img(img)
        if i > 3:
            break


def test_dataloader(file_name=FILE_PYTEST):
    loader = create_loader(file_name, batch_size=CONFIG.batch_size)
    for i, (imgs, labels) in enumerate(loader):
        print(i, imgs.shape)
        print(labels.shape)
        print(labels[0])
        visualize_img(imgs[0])
        if i > 2:
            break


def test_model(class_name):
    model = class_name(num_classes=340, pool_size=POOLING_SIZE)
    inp = Variable(torch.randn(3, 3, 64*POOLING_SIZE//2, 64*POOLING_SIZE//2))
    out = model(inp)
    print(inp.shape)
    print(out.shape)


def test_train():
    train = Train()
    train.one_epoch()
    preds = train.predict_test()
    print(preds.shape)
    train.load_model()
    model = train.model
    generator_tr = create_generator("aa.csv", batch_size=CONFIG.batch_size)
    imgs, labels = next(generator_tr)

    preds = model.predict(imgs)
    preds = np.argmax(preds, axis=1)
    print(preds)
    print(labels)


def test_make_submission():
    make_submission()

