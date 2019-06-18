from keras import applications, optimizers
from keras.layers import (LSTM, Activation, Dense, Dropout, Embedding,
                          GlobalAveragePooling2D)
from keras.models import Model, Sequential

base_model = applications.vgg16.VGG16(include_top=False, weights='imagenet')



i=0
for layer in base_model.layers:
    layer.trainable = False
    i = i+1
    print(i,layer.name)

def create_sequential():

    new_base_model  = Model(inputs=base_model.input, outputs=base_model.output)
    model = Sequential()
    model.add(new_base_model)
    model.add(Dense(128, input_dim=base_model.output))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.2))

    model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.001, momentum=0.9),metrics=["accuracy"])
    return model

def create_functional():
    x = base_model.output
    x = Dense(128, activation='sigmoid')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(10, activation='softmax')(x)



    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.001, momentum=0.9),metrics=["accuracy"])


model = create_sequential()
model.summary()
