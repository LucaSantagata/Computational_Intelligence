from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

DATASET_DIR=r"C:\Users\lucas\OneDrive\Desktop\faces"
BATCH_SIZE= 16

datagen = ImageDataGenerator(
    validation_split = 0.3,   #numero di immagini per il test (ne usiamo il 70%
    rescale = 1./255, #normalizziamo le immagini
    horizontal_flip = True,
    zoom_range = 0.2,
    brightness_range = [1,2]
)

#importiamo solo il test_set, in quanto è a questo set che la confusion matrix fa riferimento
test_generator = datagen.flow_from_directory(
    DATASET_DIR,
    shuffle=False,
    target_size=(200,200), #ridimensioniamo tutte le immagini ad una dimensione comune
    batch_size = BATCH_SIZE,
    class_mode = "binary",
    subset = "validation"
)

model= tf.keras.models.load_model("CNN_faces_50_epoche.h5")#importiamo il modello della CNN (CAMBIARE PATH)

y_pred = model.predict(test_generator,batch_size=17)


"""PREDIZIONI"""
predizioni=[] #sarà una mtrice

for i in y_pred:
    if i<0.50:
        predizioni.append([1,0])
    else:
        predizioni.append([0,1])

y_pred_classe = np.argmax(predizioni, axis=1)



confusion_mtx = confusion_matrix(test_generator.classes,y_pred_classe)

fig, ax = plt.subplots(figsize=(2,2))
ax=sns.heatmap(confusion_mtx, annot=True, fmt="d", ax=ax, cmap="Blues")
ax.set_xlabel("DONNA                                                                                                                   UOMO\n VALORE PREDETTO ")
ax.set_ylabel("VALORE CORRETTO\n    UOMO                                                                      DONNA")
ax.set_title("MATRICE DI CONFUSIONE")
plt.show()
