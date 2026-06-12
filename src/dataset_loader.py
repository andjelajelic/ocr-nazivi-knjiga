# src/dataset_loader.py
import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import config

class BookOCRDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        sve_slike = []
        ekstenzije = ["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]
        
        for ext in ekstenzije:
            sve_slike.extend(glob.glob(os.path.join(data_dir, ext)))
            
        self.image_paths = sorted(sve_slike)
        self.char_to_idx = config.CHAR_TO_IDX
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        
        # 1. Učitavanje i obrada slike
        image = Image.open(img_path).convert('L') # Grayscale
        image = image.resize((config.IMG_WIDTH, config.IMG_HEIGHT)) # 256x64
        
        # Pretvaranje u tenzor preko numpy-ja (izbegavamo stare warning-e)
        img_np = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(img_np).unsqueeze(0) # Dimenzija: [1, 64, 256]
        
        # 2. Učitavanje i obrada tekstualnog labela
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            text = ""
            
        # Pretvaramo slova u brojeve prateći velika i mala slova
        label_indices = []
        for char in text:
            if char in self.char_to_idx:
                label_indices.append(self.char_to_idx[char])
            else:
                if ' ' in self.char_to_idx:
                    label_indices.append(self.char_to_idx[' '])
                else:
                    label_indices.append(0)
                
        label_tensor = torch.tensor(label_indices, dtype=torch.long)
        label_length = torch.tensor([len(label_indices)], dtype=torch.long)
        
        return image_tensor, label_tensor, label_length

if __name__ == "__main__":
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