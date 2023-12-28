#from riconoscimento_Testo import ___

#import
import cv2
from PIL import Image
import pytesseract

# Imposta il percorso del programma Tesseract OCR (assicurati che sia nel tuo PATH)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

class TextRecognition():
    def __init__(self):
        immagine = ""
    
    def riconoscimento_testo_da_frame(self,frame):
        # Converti il frame in formato RGB (richiesto da pytesseract)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Converti il frame RGB in un oggetto immagine di PIL
        immagine = Image.fromarray(frame_rgb)
        
        # Riconosci il testo nell'immagine utilizzando Tesseract OCR
        testo_riconosciuto = pytesseract.image_to_string(immagine)

        return testo_riconosciuto