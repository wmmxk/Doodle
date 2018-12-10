import os
from collections import namedtuple
from .helper.settings import settings
import sys

#TODO remove this when running
setting_id = 0
#setting_id = int(sys.argv[1])

NAME_PROJECT = "Doodle"
RANDOM_SEED = 2018
DIR_DATA = "/home/wxk/Data/Kaggle/%s"%NAME_PROJECT
DIR_TRAIN_IMG = os.path.join(DIR_DATA, "chunks")
DIR_TEST_IMG = os.path.join(DIR_DATA, "test")
DIR_TRAIN_IMG_RAW = os.path.join(DIR_DATA, "train_simplified")
FILE_TEST = "test_simplified.csv"
SUB_TEMPLATE = "sample_submission.csv"
FILE_PYTEST = "aa.csv"

SIZE_INPUT = 64
MARGIN = 40  # add margin to the final plot
SHIFT = 50  # shift before plot the figure on the canvas
SIZE_CANVAS = 255 + MARGIN + SHIFT  # the size of the canvas for painting
POOLING_SIZE = 2 # 2*(SIZE_INPUT//64)

MEAN = 0.11
STD = 0.29

DIR_PROJECT = "/home/wxk/Data_Science/Kaggle"
DIR_MODEL = os.path.join(DIR_PROJECT, "Model", NAME_PROJECT)
DIR_SUB = os.path.join(DIR_PROJECT, "Submission", NAME_PROJECT)
DIR_OUTPUT = os.path.join(DIR_PROJECT, "Output", NAME_PROJECT)

#  Notes for Config: recognized: include the unrecognized rows or not
#                    train: training or testing.

Config = namedtuple('Config', 'type_model, type_loss, epoch, batch_size, recognized, transform, sanity')
setting = settings[setting_id]
CONFIG = Config(*setting)
#  LOG_NAME: name for model and log file
LOG_NAME = "-".join(["%s:%s" % (k, v) for k, v in zip(CONFIG._fields, CONFIG)][:-1] + ["input_size:%d"%(SIZE_INPUT)])   # not all setting are useful
print(LOG_NAME)

NUM_SKIM = 400
NUM_TRAIN = 500
NUM_VALID = 40
NUM_CLASS = 340

# TO DO refactor this code

