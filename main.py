#import librerie
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import time

# Classi 'Automobile' e 'Gatto'
classi = ['Automobile', 'Gatto']

# Carica il modello addestrato
model = load_model('./model/SeeBeyond.h5') 

# Dimensioni dell'immagine che il modello si aspetta
img_size = (128, 128)

# Funzione per elaborare l'immagine prima di effettuare la predizione
#def preprocess_image(image):
 #   img = cv2.resize(image, img_size)
  #  img = img_to_array(img) / 255
   # img = np.expand_dims(img, axis=0)
    #return img

def preprocess_image(frame):
    # Converti l'immagine in scala di grigi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Ridimensiona l'immagine
    img = cv2.resize(gray, img_size)

    # Converte l'immagine in un array e normalizza i valori dei pixel a un range tra 0 e 1
    img = img_to_array(img) / 255.0

    # Espandi le dimensioni dell'array per includere la dimensione del batch
    img = np.expand_dims(img, axis=0)

    # Espandi le dimensioni dell'array per includere la dimensione del canale
    img = np.expand_dims(img, axis=-1)

    return img

# Avvia il flusso video dalla webcam
cap = cv2.VideoCapture(0)  # 0 indica la webcam predefinita, potrebbe variare a seconda del tuo sistema

while True:
    # Leggi un frame dalla webcam
    ret, frame = cap.read()

    # Preprocessa l'immagine
    processed_frame = preprocess_image(frame)

    # Effettua la predizione
    prediction = model.predict(processed_frame)

    # Trova l'indice della classe con la probabilità massima
    cv2.putText(frame, 'automobile: '+str(prediction[0][0]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, 'gatto: '+str(prediction[0][1]), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if np.max(prediction) > 0.65:
        predicted_class = np.argmax(prediction)

        # Ottieni l'etichetta della classe
        label = classi[predicted_class]

        # Disegna il risultato sul frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    print(np.argmax(prediction))

    # Visualizza il frame
    cv2.imshow('Webcam', frame)

    # Esci dal loop se viene premuto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia la risorsa della webcam e chiudi le finestre
cap.release()
cv2.destroyAllWindows()