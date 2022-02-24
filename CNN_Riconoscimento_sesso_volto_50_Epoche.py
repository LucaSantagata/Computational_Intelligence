import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATASET_DIR =r"C:\Users\lucas\OneDrive\Desktop\faces"
epochs= 50
BATCH_SIZE = 16

#istanziamo il generatore
datagen = ImageDataGenerator(
    validation_split = 0.3,   #Frazione dei dati di addestramento da utilizzare come dati di convalida (vedere screenshot)
    rescale = 1./255, #normalizziamo le immagini
    horizontal_flip = True,
    zoom_range = 0.2,
    brightness_range = [1,2]
)
train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(200,200), #ridimensioniamo tutte le immagini ad una dimensione comune
    batch_size = BATCH_SIZE,
    class_mode = "binary",
    subset = "training"
)
test_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(200,200), #ridimensioniamo tutte le immagini ad una dimensione comune
    batch_size = BATCH_SIZE,
    class_mode = "binary",
    subset = "validation"
)
print(train_generator.class_indices) #sapere a quale classe, il nostro gen, ha associato un determinato label

model = Sequential()
model.add(Conv2D(filters=64, kernel_size = 4, padding ="same", activation = "relu", input_shape=(200,200,3)))
model.add(MaxPooling2D(pool_size=4, strides=4))
model.add(Dropout(0.1)) #ad ogni iterazione del GD spegniamo il 10% dei nodi
model.add(Conv2D(filters=32, kernel_size=4, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=4, strides=4))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics= "accuracy")

history_CNN = model.fit(train_generator,batch_size=BATCH_SIZE,epochs=epochs,verbose=1,validation_data=test_generator)

metrics_train = model.evaluate(train_generator)
metrics_test = model.evaluate(test_generator)

plt.plot(history_CNN.history['accuracy'])
plt.plot(history_CNN.history['loss'])
plt.title('Accuracy vs Loss 50 epoche')
plt.xlabel('epochs')
plt.ylabel('accuracy\loss')
plt.legend(['accuracy','loss'],loc='upper left')
plt.show()


model.save("CNN_faces_50_epoche.h5")

print("TEST ACCURACY=",metrics_test[1],"TEST LOSS",metrics_test[0])