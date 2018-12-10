
import os
import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from keras.utils.np_utils import to_categorical
import torch
from ..config import DIR_TRAIN_IMG
from ..config import DIR_TEST_IMG
from ..helper.dataset_h import fetch_from_one_row
from ..helper.meta import create_label_mapping
from ..config import CONFIG
from ..helper.random_adjust_mask import random_adjust_mask
from .custom_transform import MultiCompose
from ..config import RANDOM_SEED
from ..config import NUM_CLASS


class MyDataset(Dataset):

    def __init__(self, name_csv, transform=MultiCompose(), cutoff=0.5, train=True):
        if train and CONFIG.transform:
            self.transform = transform
        else:
            self.transform = None
        self.train = train
        if self.train:
            self.df = pd.read_csv(os.path.join(DIR_TRAIN_IMG, name_csv))
            if CONFIG.recognized:
                self.df = self.df[self.df.recognized]
            else:
                mask = random_adjust_mask(self.df.recognized, cutoff=cutoff)
                print(" total rows--- %d; selected rows %d" % (self.df.shape[0], sum(mask)))
                self.df = self.df[mask]

            self.df = shuffle(self.df, random_state=RANDOM_SEED)
            # no shuffle for sanity check
            if CONFIG.sanity:
                self.df = self.df.iloc[:5000, :]

        #  sanity recognized, shuffle, and sanity do not apply to the testing dataset
        else:
            self.df = pd.read_csv(os.path.join(DIR_TEST_IMG, name_csv))
        self.len = self.df.shape[0]
        self.mapping = create_label_mapping()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img, label = fetch_from_one_row(self.df.iloc[index, :], self.mapping, self.train)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def create_loader(name_csv, batch_size=32, mode=True, cutoff=0.5):
    """
    swap the dimension and filter the bad samples when using this loader
    the unrecognized samples are gradually included in a Train object.
    """
    dataset = MyDataset(name_csv, cutoff=cutoff, train=mode)
    return DataLoader(
        dataset=dataset,
        shuffle=mode,
        num_workers=0,
        pin_memory=True,
        batch_size=batch_size)


class CreateGenerator:

    def __init__(self, name_csv, batch_size=32, mode=True, cutoff=0.5):
        self.dataset = MyDataset(name_csv, cutoff=cutoff, train=mode)
        self.batch_size = batch_size
        self.steps = (len(self.dataset) + batch_size - 1)//batch_size
        self.mode = mode
        self.cutoff = cutoff

    def create_generator(self):
        dataloader = DataLoader(dataset=self.dataset,
                                shuffle=self.mode,
                                num_workers=4,
                                pin_memory=True,
                                batch_size=self.batch_size)
        while True:
            for imgs, labels in dataloader:
                mask = labels != -1
                #labels = to_categorical(labels[mask].numpy(), num_classes=NUM_CLASS)
                yield imgs[mask].numpy(), labels[mask].numpy()

def create_generator(name_csv, batch_size=32, mode=True, cutoff=0.5):
    dataset = MyDataset(name_csv, cutoff=cutoff, train=mode)
    dataloader = DataLoader(dataset=dataset,
                            shuffle=mode,
                            num_workers=4,
                            pin_memory=True,
                            batch_size=batch_size)
    while True:
        for imgs, labels in dataloader:
            mask = labels != -1
            #labels = to_categorical(labels[mask].numpy(), num_classes=NUM_CLASS)
            yield imgs[mask].numpy(), labels[mask].numpy()
