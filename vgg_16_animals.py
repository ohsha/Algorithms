
from tensorflow.keras.layers import  Conv2D, Activation, Flatten, Dense, MaxPool2D
from tensorflow.keras.models import  Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np


class VGG16:
    def __init__(self):
        pass


    def _block(self, model, layers, filters):
        for l in range(layers):
            model.add(Conv2D(filters=filters, kernel_size=(3,3), padding='same'))
            model.add(Activation('relu'))

        model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

        return model


    def build(self, classes):

        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(224,224,3)))
        model.add(Activation('relu'))

        model = self._block(model, layers=1, filters=64)
        model = self._block(model, layers=2, filters=128)
        model = self._block(model, layers=3,  filters=256)
        model = self._block(model, layers=3,  filters=512)
        model = self._block(model, layers=3,  filters=512) # output > 7X7X512

        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dense(classes, activation='softmax'))

        model.summary()

        return model



if __name__ == '__main__':

    # downloaded from Kaggle:
    # https://www.kaggle.com/ashishsaxena2209/animal-image-datasetdog-cat-and-panda
    data_path = r'C:\Users\MY\Desktop\AI\Primrose\bootcamp\hands_on\datasets\images\animals'

    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2,
                                       zoom_range=0.3, rotation_range=25 ,horizontal_flip=True, validation_split=0.2)
    # test_datagen = ImageDataGenerator(rescale=1./255)

    batch_size = 32
    train_generator = train_datagen.flow_from_directory(data_path, batch_size=batch_size, target_size=(224,224),
                                                        shuffle=True, seed=42, subset='training')
    val_generator = train_datagen.flow_from_directory(data_path , batch_size=batch_size, target_size=(224,224),
                                                      shuffle=True, seed=42, subset='validation')
    # test_generator = test_datagen.flow_from_directory(test_data_path , batch_size=batch_size, target_size=(224,224),
    #                                                   shuffle=True, seed=42)

    epochs = 50
    step_per_train = train_generator.n // batch_size
    step_per_val = val_generator.n // batch_size

    opt = Adam(0.001)
    model = VGG16().build(classes=3)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_generator, validation_data=val_generator,
              steps_per_epoch=step_per_train, validation_steps=step_per_val,
                        epochs=epochs, verbose=1)


    preds = model.predict(val_generator, batch_size, verbose=1)
    predictions = preds.argmax(axis=1)

    score = model.evaluate(test_generator)
    print(f'loss : {score[0]:.5f}   accuracy: {100 * score[1]:.2f}%')
