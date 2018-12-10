from keras.metrics import sparse_top_k_categorical_accuracy


def top_3_accuracy(y_true, y_pred):
    return sparse_top_k_categorical_accuracy(y_true, y_pred, k=3)
