from keras.layers.core import Activation, Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

y_train_one_hot = np_utils.to_categorical(y_train)
y_test_one_hot = np_utils.to_categorical(y_test)

# print(y_one_hot)

model = Sequential()
model.add(Dense(16, input_shape=(4,)))
model.add(Activation("sigmoid"))
model.add(Dense(3))
model.add(Activation("softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train_one_hot, epochs=100, batch_size=1, verbose=0)
loss, accuracy = model.evaluate(X_test, y_test_one_hot, verbose=0)
print("Accuracy = {:.2f}".format(accuracy))
