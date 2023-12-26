import torch
import os
import json
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import os
from PIL import Image, ImageDraw
from torchvision.transforms import ToPILImage
import shutil #per rimuovere directory


class CustomCocoDataset(Dataset):
    def __init__(self, root, classes, image_size=None, transform=None, target_transform=None):
        self.root = root
        self.classes = classes
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.data = self.load_data()

    def load_data(self):
        data = []
        i = 0

        for class_name in self.classes:
            i += 1 #contatore per il numero di classi, assegna il target
            class_path = os.path.join(self.root, 'annotations', class_name)
            for ann_filename in os.listdir(class_path):
                if ann_filename.endswith(".json"):
                    json_path = os.path.join(class_path, ann_filename)
                    image_filename = ann_filename.replace('.json', '.jpg').replace('annotations', class_name)
                    with open(json_path, "r") as json_file:
                        annotations = json.load(json_file)
                        for annotation in annotations:
                            '''
                            # Convert the bounding box coordinates from COCO format to Fast R-CNN format
                            x_min, y_min, width, height = annotation['bbox']
                            x_max = x_min + width
                            y_max = y_min + height
                            boxes = torch.tensor([x_min, y_min, x_max, y_max])
                            '''
                            '''
                            #estre le coordinate del bounding box
                            x_min, y_min, width, height = annotation['bbox']

                            # carica immagine per ottenere dimensioni originali
                            original_width, original_height = Image.open(os.path.join(self.root, 'images', class_name, image_filename)).size #prende le dimensioni originali dell'immagine

                            # Calcola il rapporto tra le dimensioni originali e quelle ridimensionate
                            width_ratio = self.image_size / original_width
                            height_ratio = self.image_size / original_height

                            # Ridimensiona le coordinate del bounding box
                            x_min_resized = x_min * width_ratio
                            y_min_resized = y_min * height_ratio
                            width_resized = width * width_ratio
                            height_resized = height * height_ratio
                            
                            # Calcola le coordinate x_max e y_max necessarie a Faster R-CNN
                            x_max_resized = x_min_resized + width_resized
                            y_max_resized = y_min_resized + height_resized

                            # Converte le coordinate ridimensionate in tensori
                            #boxes = torch.tensor([x_min_resized, y_min_resized, x_max_resized, y_max_resized])

                            # Assegna il target (indice di classe)
                            #labels = torch.tensor(i)
                            '''
                            boxes = []
                            labels = []
                            for annotation in annotations:
                                #estre le coordinate del bounding box
                                x_min, y_min, width, height = annotation['bbox']

                                # carica immagine per ottenere dimensioni originali
                                original_width, original_height = Image.open(os.path.join(self.root, 'images', class_name, image_filename)).size #prende le dimensioni originali dell'immagine

                                # Calcola il rapporto tra le dimensioni originali e quelle ridimensionate
                                width_ratio = self.image_size / original_width
                                height_ratio = self.image_size / original_height

                                # Ridimensiona le coordinate del bounding box
                                x_min_resized = x_min * width_ratio
                                y_min_resized = y_min * height_ratio
                                width_resized = width * width_ratio
                                height_resized = height * height_ratio
                                
                                # Calcola le coordinate x_max e y_max necessarie a Faster R-CNN
                                x_max_resized = x_min_resized + width_resized
                                y_max_resized = y_min_resized + height_resized
                                
                                boxes.append(torch.tensor([x_min_resized, y_min_resized, x_max_resized, y_max_resized]))
                                labels.append(torch.tensor(i))

                            data.append({
                                "image_path": os.path.join(self.root, 'images', class_name, image_filename),
                                "annotations": {"boxes": boxes, "labels": labels}
                            })
                            '''
                            data.append({
                                "image_path": os.path.join(self.root, 'images', class_name, image_filename),
                                "annotations": {"boxes": boxes.tolist(), "labels": labels.tolist()}
                            })'''

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        image_path = item["image_path"]
        annotations = item["annotations"]

        # Carica l'immagine
        img = Image.open(image_path).convert("RGB")

        # Applica le trasformazioni
        if self.transform is not None:
            img = self.transform(img)

        # Converti le liste in tensori quando le recuperi
        #boxes = torch.tensor(annotations["boxes"], dtype=torch.float32)
        #labels = torch.tensor(annotations["labels"], dtype=torch.int64)

        # Converti le liste in tensori quando le recuperi
        boxes = [box for box in annotations["boxes"]]
        boxes = torch.stack([torch.tensor(box, dtype=torch.float32) for box in boxes])
        labels = torch.tensor(annotations["labels"], dtype=torch.int64)

        # Restituisci l'immagine e le annotazioni come tensori
        return img, {"boxes": boxes, "labels": labels}
    
    
    def save_dataset_with_boxes_labels(self, dataset, output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)
        i=0

        for image, annotations in dataset:
            i += 1
            # Converti il tensore in un'immagine PIL
            to_pil = ToPILImage()
            img = to_pil(image)

            # Disegna i boxes e i labels sull'immagine
            draw = ImageDraw.Draw(img)
            x_min, y_min, x_max, y_max = annotations["boxes"]
            draw.rectangle([x_min, y_min, x_max, y_max], outline="green")
            draw.text((x_min, y_min), str(annotations["labels"]), fill="green")

            # Salva l'immagine con i boxes e i labels
            img.save(os.path.join(output_dir, f"{annotations['labels']}_{i}.png"))

        print("Dataset salvato correttamente nella cartella", output_dir)
