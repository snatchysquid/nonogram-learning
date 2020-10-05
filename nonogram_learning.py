from tensorflow import keras
import tensorflow as tf
import nonogram_data
import pickle

TRAIN_PERCENT = 0.9

model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(100,), activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(100),
    keras.layers.Activation(activation='sigmoid')
])

earlystop_callback = keras.callbacks.EarlyStopping(min_delta=0.000001, patience=1
                                                   , monitor='accuracy'
                                                   )

optimizer = keras.optimizers.Adam(lr=0.4)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

def create_data(N=1000000):
    """
    WARNING: the default N produces a file of size 1.2GB!
    """
    X_set, Y_set = nonogram_data.get_set(N)
    sep = int(len(X_set)*TRAIN_PERCENT)
    X_train, Y_train = X_set[:sep], Y_set[:sep]
    X_test, Y_test = X_set[sep:], Y_set[sep:]

    with open('training_data.pickle', 'wb') as handle:
        pickle.dump(((X_train, Y_train), (X_test, Y_test)), handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("training_data.pickle", "rb") as f:
    ((X_train, Y_train), (X_test, Y_test)) = pickle.load(f)


#model.fit(X_train, Y_train, batch_size=8, epochs=200, callbacks=[earlystop_callback])
model.fit(X_train, Y_train, batch_size=2, epochs=200
#          , callbacks=[earlystop_callback]
          )
ev = model.evaluate(X_test, Y_test)
print(ev)
