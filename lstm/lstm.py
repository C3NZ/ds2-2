import keras
import numpy as np
from keras.datasets import mnist
from keras.layers import LSTM, Activation, Dense, Dropout
from keras.models import Sequential

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/np.max(x_train)
x_test = x_test/np.max(x_test)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# print(x_train[1])
x_train = x_train.reshape(x_train.shape[0], 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28)
print(len(x_train[0]))
nb_units = 50

model = Sequential()
# model.add(LSTM(256, input_dim=1, input_length=5))
model.add(LSTM(nb_units, input_shape=(28, 28)))
model.add(Dense(units=10, activation='softmax'))
# 2.5 Compile the model.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 2.6 Print out model.summary
epochs = 3

model.summary()

#history = model.fit(x_train,
#                    y_train,
#                    epochs=epochs,
#                    batch_size=128,
#                    verbose=1,
#                    validation_split=0.2)
#
#scores = model.evaluate(x_test, y_test, verbose=2)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
