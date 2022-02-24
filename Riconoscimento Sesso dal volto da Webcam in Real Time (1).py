import cv2
from tensorflow.keras.models import load_model

#DEFINIAMO LA DIMENSIONE DELLE IMMAGINI
SCALE = (200, 200)

model = load_model("CNN_faces_50_epoche.h5")

cap = cv2.VideoCapture(0) #apriamo la webcam

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(cap.isOpened()):
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = face_cascade.detectMultiScale(gray, 1.1, 15) #cerchiamo eventuali volti nell'immagine, 1.1 è lo scale factor

    #iteriamo su tutti i volti trovati, ed estraiamo il volto dall'immagine
    for rect in rects:
        img = frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]  #lavoriamo con l'immagine a colori
        small_img = cv2.resize(img, SCALE)
        x = small_img.astype(float) #convertiamo tutto in float
        x/=255.


        x = x.reshape(1, SCALE[0], SCALE[1], 3)
        y= model.predict(x)

        y = y[0][0] #la predizione è una matrice con una riga e una colonna. In questo modo ne estraiamo l'unicio valore.

        label = "Donna" if y>0.5 else "Uomo"
        percentage = y if y>0.5 else 1.0-y
        percentage = round(percentage*100,1)

        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0]+rect[2],rect[1]+rect[3]),(0,255,0), 2)
        cv2.rectangle(frame, (rect[0], rect[1]-20), (rect[0]+170, rect[1]),(0,255,0), cv2.FILLED)
        cv2.putText(frame, label+ "("+str(percentage)+"%)", (rect[0]+5, rect[1]), cv2.FONT_HERSHEY_PLAIN, 1.4, (255,255,255), 2) #1.4 è il font scale, mentre 2 è lo spessore

    cv2.imshow("Gender Recognition",frame)

    if (cv2.waitKey(1) == ord("q")):
        break











