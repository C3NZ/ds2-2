import numpy as np
from keras.layers import LSTM, Activation, Dense, Dropout, Embedding
from keras.models import Sequential

input_array = np.array([[[0], [1], [2], [3], [4]], [[5], [1], [2], [3], [6]]])
print(input_array.shape)
model = Sequential()
# model.add(LSTM(256, input_dim=1, input_length=5))
model.add(LSTM(10, input_shape=(5, 1), return_sequences=False))
model.summary()
print(input_array)
model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
print(output_array)
# the number of parameters of a LSTM layer in Keras equals to
# params = 4 * ((size_of_input + 1) * size_of_output + size_of_output^2)
n_params = 4 * ((1 + 1) * 10 + 10**2)
print(n_params)
print(model.summary())
