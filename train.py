import os
import pandas as pd
import keras.backend as K
from lib.config import DIR_MODEL
import lib.models.models as models
from keras.metrics import sparse_categorical_accuracy
from lib.config import FILE_TEST
from lib.helper.meta import get_file_names
from lib.config import DIR_TRAIN_IMG
from lib.config import LOG_NAME
from lib.config import DIR_OUTPUT
from lib.config import CONFIG
from lib.config import DIR_SUB
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from lib.dataloader.my_dataset import CreateGenerator
from keras.models import load_model
from lib.helper.top_3_accuracy import top_3_accuracy


class Train(object):

    def __init__(self):
        self.epoch = 0
        self.chunk = 0
        self.name_csvs = get_file_names(my_dir=DIR_TRAIN_IMG)
        self.csv_valid = self.name_csvs[0]
        self.name_csvs = self.name_csvs[1:]
        self.current_csv = self.name_csvs[0]
        self.generator_train = None
        self.generator_valid = None
        self.generator_test = None
        self.model = None
        self.loss_fun = None
        self.lr = 0.01
        self.create_model_loss()

    def create_model_loss(self):
        self.model = getattr(models, CONFIG.type_model)()

        checkpoint_path = os.path.join(DIR_MODEL, LOG_NAME + "_best.h5")
        exist = os.path.isfile(checkpoint_path)
        if exist:
            self.model = load_model(checkpoint_path, custom_objects={"top_3_accuracy": top_3_accuracy})
        self.model.compile(optimizer=Adam(lr=0.01), loss='sparse_categorical_crossentropy',
                           metrics=[sparse_categorical_accuracy, top_3_accuracy])

    def create_generator(self, train=True):
        # train vs test is controlled by CONFIG.train;
        # train vs validation is controlled by train
        if train:
            self.generator_train = CreateGenerator(self.current_csv, batch_size=CONFIG.batch_size, mode=True,
                                                   cutoff=(self.chunk+1)*0.02)
            if self.chunk == 0:
                self.generator_valid = CreateGenerator(self.csv_valid, batch_size=CONFIG.batch_size, mode=True,
                                                       cutoff=(self.chunk+1)*0.02)
        else:
            self.generator_test = CreateGenerator(FILE_TEST, batch_size=CONFIG.batch_size, mode=False, cutoff=1.1)

    def train_current_generator(self):
        df = pd.read_csv(os.path.join(DIR_TRAIN_IMG, self.current_csv))
        path_history = os.path.join(DIR_OUTPUT, LOG_NAME+"log.csv")
        csv_logger = CSVLogger(path_history, append=True, separator=';')
        adjust_lr = ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy', factor=0.7, patience=5,
                                      min_delta=0.005, mode='max', cooldown=3, verbose=1)

        path_model = os.path.join(DIR_MODEL, LOG_NAME + "_best.h5")
        checkpoint = ModelCheckpoint(path_model, verbose=1, monitor='val_sparse_categorical_accuracy', save_best_only=True)

        callbacks = [adjust_lr, checkpoint, csv_logger]

        hist = self.model.fit_generator(self.generator_train.create_generator(),
                                        steps_per_epoch=self.generator_train.steps,
                                        epochs=1,
                                        verbose=1,
                                        validation_data=self.generator_valid.create_generator(),
                                        validation_steps=self.generator_valid.steps,
                                        callbacks=callbacks)
        return hist

    def train_one_chunk(self):
        self.create_generator()
        self.train_current_generator()

    def one_epoch(self):
        for name_csv in self.name_csvs:
            self.current_csv = name_csv
            self.create_generator()
            hist = self.train_current_generator()
            lr = float(K.get_value(self.model.optimizer.lr))
            print("Chunk %d: %s, Learning rate: %f" % (self.chunk, self.current_csv, lr))
            if hist.history['val_top_3_accuracy'][0] > 0.92:
                self.model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy',
                                   metrics=[sparse_categorical_accuracy, top_3_accuracy])
                self.predict_test()
            elif hist.history['val_top_3_accuracy'][0] > 0.91:
                self.model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy',
                                   metrics=[sparse_categorical_accuracy, top_3_accuracy])
            elif hist.history['val_top_3_accuracy'][0] > 0.90:
                self.model.compile(optimizer=Adam(lr=0.005), loss='sparse_categorical_crossentropy',
                                   metrics=[sparse_categorical_accuracy, top_3_accuracy])
            self.chunk += 1

    def save_model(self):
        path_model = os.path.join(DIR_MODEL, LOG_NAME + "_best.h5")
        self.model.save(path_model)

    def load_model(self):
        path_model = os.path.join(DIR_MODEL, LOG_NAME + "_best.h5")
        self.model = load_model(path_model)

    def evaluate_valid(self):
        predictions = model.predict_generator(self.generator_valid)
        #predictions = np.argmax(predictions, axis=-1)
        return predictions

    def predict_test(self):
        print(" predict test---")
        self.create_generator(train=False)
        preds = self.model.predict_generator(self.generator_test.create_generator(), steps=self.generator_test.steps)
        preds = pd.DataFrame(preds)
        preds.to_csv(os.path.join(DIR_SUB, LOG_NAME+"_prob.csv"), index=False)
        return preds
