import numpy as np
from keras.layers import Input, Lambda
from keras.models import Model

input = Input(shape=(3,))
double = Lambda(lambda x: 2 * x)(input)

model = Model(input=input, output=double)
model.compile(optimizer="sgd", loss="mse")

data = np.array([[5, 12, 1]])
print(model.predict(data))
