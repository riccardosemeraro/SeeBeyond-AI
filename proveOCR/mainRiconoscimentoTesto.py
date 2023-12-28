#import
import cv2
from PIL import Image
import pytesseract
import numpy as np

from proveOCR.riconoscimento_Testo import TextRecognition









#Inizio main -----------------

oggetto = TextRecognition()

# Inizializza l'acquisizione video da una fotocamera (0 è solitamente la fotocamera predefinita)
# video_capture = cv2.VideoCapture(0)
frame = cv2.imread('./immaginiOCR/cartello2 copia.jpg')

testo_estratto = oggetto.riconoscimento_testo_da_frame(frame)

print("Testo Riconosciuto:")
print(testo_estratto)

#Controlla se ha riconosciuto o meno un testo
#if (testo_estratto.count("\n") == 1):
#    print("Non è stata riconosciuta alcuna stringa")

'''
while True:
    # Leggi il frame dalla fotocamera
        #_, frame = video_capture.read()

    # Riconosci il testo nel frame
    testo_estratto = oggetto.riconoscimento_testo_da_frame(frame)

    # Visualizza il frame con il testo riconosciuto
    #cv2.imshow('Frame', frame)
    print("Testo riconosciuto:")
    print(testo_estratto)

    #Controlla se ha riconosciuto o meno un testo
    if (testo_estratto.count("\n") == 1):
    print("Non è stata riconosciuta alcuna stringa")

    # Esci dal loop quando viene premuto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''
        
# Rilascia la fotocamera e chiudi le finestre
#video_capture.release()
cv2.destroyAllWindows()