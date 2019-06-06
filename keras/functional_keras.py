from keras.layers import Dense, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

iris = datasets.load_iris()
X, y = iris.data, iris.target
# print(y_one_hot)


def create_nn() -> Model:
    """
        Create the model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    y_train_one_hot = np_utils.to_categorical(y_train)
    y_test_one_hot = np_utils.to_categorical(y_test)

    inp = Input(shape=(4,))
    x = Dense(16, activation="sigmoid")(inp)
    out = Dense(3, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(X_train, y_train_one_hot, epochs=100, batch_size=1, verbose=0)
    loss, accuracy = model.evaluate(X_test, y_test_one_hot, verbose=0)
    print("Accuracy = {:.2f}".format(accuracy))
    return model


if __name__ == "__main__":
    create_nn()
