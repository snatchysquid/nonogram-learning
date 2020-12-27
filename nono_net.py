import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import сreate_nono
import pickle

TRAIN_PERCENT = 0.9

# TODO NORMALIZE DATA SO YOU DONT GET HUGE LOSSES


def create_model(checkpoint_path=""):
    """ create a model and load checkpoint if specified"""
    # model = keras.Sequential([
    #     keras.layers.Dense(80, input_shape=(100,), activation='relu'),
    #     keras.layers.Dense(60, activation='relu'),
    #     keras.layers.Dense(80, activation='relu'),
    #     keras.layers.Dense(100),
    #     keras.layers.Activation(activation='sigmoid')
    # ])
    model = keras.Sequential([
        keras.layers.Dense(16, input_shape=(12,), activation='elu'),
        keras.layers.Dense(24, activation='elu'),
        keras.layers.Dense(64, activation='elu'),
        keras.layers.Dense(24, activation='elu'),
        keras.layers.Dense(16, activation='elu'),
        keras.layers.Dense(9),
        keras.layers.Activation(activation='sigmoid')
    ])
    if checkpoint_path:
        model.load_weights(checkpoint_path)

    optimizer = keras.optimizers.Adam(lr=0.12)
    model.compile(optimizer=optimizer, loss="mse")
    return model


def get_and_save_data(n=10, size=1000000, path="training_data.pickle"):
    """ get `size` training instances for a board of nXn and save them to path as a pickle """
    X_set, Y_set = сreate_nono.get_set(n=n, size=size)

    # should also add validation
    sep = int(len(X_set) * TRAIN_PERCENT)
    X_train, Y_train = X_set[:sep], Y_set[:sep]
    X_test, Y_test = X_set[sep:], Y_set[sep:]

    with open("size" + str(n) + "_" + path, 'wb') as handle:
        pickle.dump(((X_train, Y_train), (X_test, Y_test)), handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


def load_data(n=10, path="training_data.pickle"):
    """ load data of nXn board instances from a pickle """
    with open("size" + str(n) + "_" + path, "rb") as f:
        ((X_train, Y_train), (X_test, Y_test)) = pickle.load(f)
    return (X_train, Y_train), (X_test, Y_test)


def train_model(model, X_train, Y_train, epochs=200, weights_path="nono_weights"):
    """ train the model and save the weights to `weights_path`, also plot the loss at the end """
    earlystop_callback = keras.callbacks.EarlyStopping(min_delta=0.00001,
                                                       patience=1,
                                                       monitor='loss')
    hist = model.fit(X_train, Y_train, batch_size=2, epochs=epochs
#              , callbacks=[earlystop_callback]
              )
    model.save_weights(weights_path)

    plt.plot(hist.history['loss'])
    plt.show()

    return model


def main():
    n = 3  # board size
    data_size = 10000

    start_train = True  # set true if you want to train the model
    start_test = True  # set true if you want to test the model
    get_data = False  # set true if you want to save new data

    if get_data:
        get_and_save_data(n, data_size, "small_data")

    # each instance in X trains is a flat list of len [2 * ceil(n/2) * n]
    # which is the maximum length possible for a constraint, for example for a
    # 10X10 board the longest constraint would [1,1,1,1,1]
    ((X_train, Y_train), (X_test, Y_test)) = load_data(n, "small_data")

    model = create_model()
    if start_train:
        model = train_model(model, X_train, Y_train, epochs=200)
    if start_test:
        ev = model.evaluate(X_test, Y_test)
        print(ev)


if __name__ == '__main__':
    main()
