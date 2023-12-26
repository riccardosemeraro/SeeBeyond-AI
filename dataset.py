from pycocotools.coco import COCO
import requests
import os
import json
from tqdm import tqdm
import shutil

#questo script scarica le immagini della categoria indicata dal database COCO e ne scarica anche le annotazioni (box e label)

annotations_dir = './dataset_Automatic/annotations'
images_dir = './dataset_Automatic/images'

# Inizializza l'API COCO
coco = COCO(os.path.join(annotations_dir, 'instances_train2017.json'))

# Seleziona le categorie di interesse
categories = coco.getCatIds(catNms=['cat', 'car'])

# Scarica le immagini e le relative annotazioni
for category in categories:
    category_name = coco.loadCats(category)[0]["name"] #estrae nomi categorie
    
    #crea cartelle per le categorie
    category_folder_img = os.path.join(images_dir, category_name)
    shutil.rmtree(category_folder_img, ignore_errors=True) #rimuove cartelle se già esistenti
    os.makedirs(category_folder_img, exist_ok=True)

    #crea cartelle per le annotazioni
    category_folder_ann = os.path.join(annotations_dir, category_name)
    shutil.rmtree(category_folder_ann, ignore_errors=True) #rimuove cartelle se già esistenti
    os.makedirs(category_folder_ann, exist_ok=True)

    imgIds = coco.getImgIds(catIds=[category])[:1]  # Numero di immagini per categoria totali

    for imgId in tqdm(imgIds, desc=f'Downloading {coco.loadCats(category)[0]["name"]} images'):
        img_info = coco.loadImgs(imgId)[0]

        # Scarica l'immagine
        img_url = img_info['coco_url']
        img_filename = f"{category_name}_{imgId}.jpg"
        img_path = os.path.join(images_dir+"/"+category_name, img_filename)
        img_data = requests.get(img_url).content
        with open(img_path, 'wb') as img_file:
            img_file.write(img_data)

        # Filtra le annotazioni solo per le categorie di interesse
        annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=categories)
        anns = coco.loadAnns(annIds)
        filtered_anns = [ann for ann in anns if ann['category_id'] in categories]

        # Salva solo le annotazioni filtrate
        ann_path = os.path.join(annotations_dir+"/"+category_name, f'annotations_{img_info["id"]}.json')
        with open(ann_path, 'w') as ann_file:
            json.dump(filtered_anns, ann_file)

print("Download completato!")
