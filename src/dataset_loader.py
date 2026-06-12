# src/dataset_loader.py
import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import config


class BookOCRDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.data_dir = data_dir
        self.augment = augment

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
        image = Image.open(img_path).convert('L')  # Grayscale
        image = image.resize((config.IMG_WIDTH, config.IMG_HEIGHT), Image.LANCZOS)

        img_np = np.array(image, dtype=np.float32) / 255.0

        # Normalizacija (mean i std za grayscale tekst slike)
        mean = 0.5
        std = 0.5
        img_np = (img_np - mean) / std

        image_tensor = torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W]

        # 2. Učitavanje labela
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            text = ""

        # Pretvaramo karaktere u indekse
        label_indices = []
        for char in text:
            if char in self.char_to_idx:
                label_indices.append(self.char_to_idx[char])
            else:
                # Nepoznati karakter -> razmak
                label_indices.append(self.char_to_idx.get(' ', 1))

        label_tensor = torch.tensor(label_indices, dtype=torch.long)
        label_length = torch.tensor([len(label_indices)], dtype=torch.long)

        return image_tensor, label_tensor, label_length


def collate_fn(batch):
    """
    Custom collate funkcija za DataLoader.
    Potrebna jer labele imaju različite dužine.
    """
    images, labels, label_lengths = zip(*batch)

    images = torch.stack(images, 0)  # [B, 1, H, W]

    # Spajamo sve labele u jedan 1D tensor (zahtev CTC lossa)
    all_labels = torch.cat(labels, 0)
    label_lengths = torch.cat(label_lengths, 0)

    return images, all_labels, label_lengths


if __name__ == "__main__":
    train_path = os.path.join(os.getcwd(), 'dataset/train')
    print(f"Pokušavam da učitam slike iz: {train_path}")

    try:
        dataset = BookOCRDataset(train_path)
        print(f"Uspešno učitan dataset! Broj slika: {len(dataset)}")
        if len(dataset) > 0:
            img, label, length = dataset[0]
            print(f"Dimenzije slike: {img.shape}")
            print(f"Min/Max vrednosti: {img.min():.2f} / {img.max():.2f}")
            print(f"Label: {label}")
            print(f"Dužina: {length.item()} karaktera")
    except Exception as e:
        print("Greška:", e)
        raise
