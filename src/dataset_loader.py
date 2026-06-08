
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import config

class BookOCRDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Kupimo sve slike iz zadatog foldera
        self.image_names = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        # Mapiranje: karakter -> broj
        self.char_to_num = {char: idx for idx, char in enumerate(config.CHARACTERS)}
        
    def __len__(self):
        return len(self.image_names)
        
    def __getitem__(self, idx):
        # 1. Učitavanje i obrada slike
        img_name = self.image_names[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        image = Image.open(img_path).convert('L') # Grayscale (crno-belo)
        image = image.resize((config.IMG_WIDTH, config.IMG_HEIGHT))
        
        image = np.array(image, dtype=np.float32) / 255.0 # Normalizacija na [0, 1]
        image = np.expand_dims(image, axis=0) # Dodajemo dimenziju kanala (1, 64, 256)
        image = torch.tensor(image, dtype=torch.float32)
        
        # 2. Učitavanje i obrada tekstualnog labela
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(self.data_dir, txt_name)
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            
        # Pretvaramo slova u brojeve prema rečniku
        label = [self.char_to_num[char] for char in text if char in self.char_to_num]
        label = torch.tensor(label, dtype=torch.long)
        
        label_length = torch.tensor(len(label), dtype=torch.long)
        
        return image, label, label_length

# TEST CODE: Da proverimo da li sve radi
if __name__ == "__main__":
    import sys
    # Provera da li kod vidi dataset folder iz root-a projekta
    train_path = os.path.join(os.getcwd(), 'dataset/train')
    print(f"Pokušavam da učitam slike iz: {train_path}")
    
    try:
        dataset = BookOCRDataset(train_path)
        print(f"Uspešno učitan dataset! Broj slika u train folderu: {len(dataset)}")
        if len(dataset) > 0:
            img, label, length = dataset[0]
            print(f"Dimenzije slike: {img.shape} (Očekivano: torch.Size([1, 64, 256]))")
            print(f"Numerički label prve knjige: {label}")
            print(f"Dužina teksta: {length.item()} karaktera.")
    except Exception as e:
        print("Došlo je do greške prilikom testiranja dataloadera:", e)