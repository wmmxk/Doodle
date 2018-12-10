import matplotlib.pyplot as plt


def look_into_model(train):
    num = len(train.acc_1_valid)
    plt.scatter(range(num), train.acc_1_valid)
    print(train.acc_3_valid)
    plt.scatter(range(num), train.acc_3_valid)
    plt.show()
