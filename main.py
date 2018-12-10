from train import Train
from lib.helper.create_log import create_log


def main():
    create_log()
    train = Train()
    train.one_epoch()
    train.predict_test()

main()
