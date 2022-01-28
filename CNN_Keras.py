import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#parametri
num_classes = 10
input_shape = (28, 28, 1)

#dati di train e dati di test
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#scaliamo i pixel delle immagini tra 0/1
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

#aggiungiamo la dimensione del colore
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# one hot ancoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""COSTRUIAMO IL MODELLO"""

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(3, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(6, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

"""plottiamo la shape dell modello"""
model.summary()

"""TRAINING"""

batch_size = 128
epochs = 3

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

"""VALUTAZIONE"""

score = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

plt.plot(history.history["accuracy"])  #history1.history
plt.show()



fig = plt.figure(figsize=(10, 10)) #impostiamo le dimensioni della figura

y_pred = model.predict(x_test) #y_pred ha 10000 righe (che sono i samples)  e 10 colonne che sono le feauture

"""il metodo np. argmax returns the indices of the maximum values along an axis"""
Y_pred = np.argmax(y_pred, 1)
Y_test = np.argmax(y_test, 1) # Decode labels

mat = confusion_matrix(Y_test, Y_pred) # Confusion matrix

"""
# Plot Confusion matrix
sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues)
plt.xlabel('Predicted Values')
plt.ylabel('True Values');
plt.show() """