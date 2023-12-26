import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db
from triangolazione_Distanza import calculate_distance

#inizializzo il collegamento con firebase
cred = credentials.Certificate("./seebeyond-8bdb7-firebase-adminsdk-jl1pf-61313a1aab.json")
firebase_admin.initialize_app(cred, {'databaseURL':'https://seebeyond-8bdb7-default-rtdb.europe-west1.firebasedatabase.app/'})

num_classes = 3 #numero di classi + 1 per il background
# Crea un'istanza del modello Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)

# Carica il checkpoint
checkpoint = torch.load('./model/SeeBeyond_70P.pth', map_location=torch.device('cpu'))

# Carica solo i pesi del modello
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

# Definisci le classi utilizzate dal modello
CLASSES = ['background', 'Gatto', 'Auto']

# Funzione per eseguire l'inferenza sul frame della webcam
def inferenza_webcam(frame, D, old_label):
    # Converte il frame da BGR a RGB (OpenCV utilizza BGR di default)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Trasforma l'immagine per adattarla al formato richiesto dal modello
    img = F.to_tensor(img)
    img = img.unsqueeze(0)  # Aggiunge una dimensione per il batch

    # Esegui l'inferenza
    with torch.no_grad():
        prediction = model(img)

    # Estrai le predizioni rilevanti
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    print(boxes)
    # Visualizza i risultati sul frame
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            if old_label != label:
                db.reference('Oggetti Rilevati/'+str(label)).set({'name': CLASSES[label], 'read': False, 'score': score*100, 'distance': D})
                old_label = label
            box = [int(coord) for coord in box]
            frame = cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            label_with_score = f"{CLASSES[label]}: {score*100:.2f}%"  # Aggiungi il punteggio di confidenza alla label
            frame = cv2.putText(frame, label_with_score, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            print(label_with_score)

    return frame, old_label

impostazioni_ref = db.reference('Impostazioni')

object_ref = db.reference('Oggetti Rilevati')

cap = calculate_distance()

old_label = 0

while True:
    X,Y,D,frame1,frame2 = cap.triangolazione()
    print(D)

    #calculate_distance()

    status = impostazioni_ref.child('Status').get()
    if status == 'ON' and D != 0:
        # Esegui l'inferenza
        frame1, old_label = inferenza_webcam(frame1, D, old_label)
    
    # Visualizza il frame risultante
    cv2.imshow("Left Camera 1", frame1)
    cv2.imshow("Right Camera 2", frame2)

    # Interrompi l'esecuzione se viene premuto 'q', se non messo non visualizza nulla
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia la risorsa della webcam e chiudi tutte le finestre
#cv2.destroyAllWindows()

