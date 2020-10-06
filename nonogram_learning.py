from tensorflow import keras
import tensorflow as tf
import nonogram_data
import pickle


TRAIN_PERCENT = 0.9


def create_model(checkpoint_path=""):
    # autoencoder style, just to see if this works
    model = keras.Sequential([
        keras.layers.Dense(80, input_shape=(100,), activation='relu'),
        keras.layers.Dense(60, activation='relu'),
        keras.layers.Dense(80, activation='relu'),
        keras.layers.Dense(100),
        keras.layers.Activation(activation='sigmoid')
    ])
    if checkpoint_path:
        model.load_weights(checkpoint_path)


    optimizer = keras.optimizers.Adam(lr=0.12)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy")
    return model


def get_and_save_data(N=1000000, path="training_data.pickle"):
    X_set, Y_set = nonogram_data.get_set(N)
    sep = int(len(X_set) * TRAIN_PERCENT)
    X_train, Y_train = X_set[:sep], Y_set[:sep]
    X_test, Y_test = X_set[sep:], Y_set[sep:]

    with open(path, 'wb') as handle:
        pickle.dump(((X_train, Y_train), (X_test, Y_test)), handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


def load_data(path="training_data.pickle"):
    with open(path, "rb") as f:
        ((X_train, Y_train), (X_test, Y_test)) = pickle.load(f)
    return ((X_train, Y_train), (X_test, Y_test))




def train_model(model, epochs=200, weights_path="nono_weights"):
    earlystop_callback = keras.callbacks.EarlyStopping(min_delta=0.00001, patience=1, monitor='loss')
    model.fit(X_train, Y_train, batch_size=2, epochs=epochs
              , callbacks=[earlystop_callback]
              )
    model.save_weights(weights_path)


if __name__ == '__main__':
    #get_and_save_data(2, "small_data")
    ((X_train, Y_train), (X_test, Y_test)) = load_data("small_data")
    model = create_model("nono_weights")
    #train_model(model, 1)  # uncomment to train
    ev = model.evaluate(X_test, Y_test)
    print(ev)
