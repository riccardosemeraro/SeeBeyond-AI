#Dataset contiene le immagini degli oggetti da riconoscere, divisi per (train, val, test) e classe (car, cat)
#training contiene tutti gli script per il training
#model contiene il modello addestrato salvato in un file .h5


#apparentemente questa parte di codice è completa, prende il modello e lo usa per il riconoscimento degli oggetti
#bisogna implementare meglio la parte di creazione del modello
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import time

# Classi 'Automobile' e 'Gatto'
classi = ['Automobile', 'Gatto']

# Carica il modello addestrato
model = load_model('./model/SeeBeyond.h5') 

# Dimensioni dell'immagine che il modello si aspetta
img_size = (64, 64)

# Funzione per elaborare l'immagine prima di effettuare la predizione
def preprocess_image(image):
    img = cv2.resize(image, img_size)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Avvia il flusso video dalla webcam
cap = cv2.VideoCapture(0)  # 0 indica la webcam predefinita, potrebbe variare a seconda del tuo sistema

while True:
    # Leggi un frame dalla webcam
    ret, frame = cap.read()

    # Effettua la predizione solo ogni tot frame (può essere regolato)
    if time.time() % 5 == 0:
        # Preprocessa l'immagine
        processed_frame = preprocess_image(frame)

        # Effettua la predizione
        prediction = model.predict(processed_frame)

        # Trova l'indice della classe con la probabilità massima
        predicted_class = np.argmax(prediction)

        # Ottieni l'etichetta della classe
        label = classi[predicted_class]

        # Disegna il risultato sul frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Visualizza il frame
    cv2.imshow('Webcam', frame)

    # Esci dal loop se viene premuto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia la risorsa della webcam e chiudi le finestre
cap.release()
cv2.destroyAllWindows()