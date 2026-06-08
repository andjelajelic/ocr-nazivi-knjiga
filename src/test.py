# src/test.py
import os
import torch
from torch.utils.data import DataLoader
import config
from dataset_loader import BookOCRDataset
from model import CRNN
from train import decode_predictions

def main():
    # 1. Putanja do test foldera
    test_dir = os.path.join(os.getcwd(), 'dataset/test')
    test_dataset = BookOCRDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 2. Učitavanje modela i težina
    model = CRNN()
    model_path = os.path.join(os.getcwd(), 'ocr_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Greška: Ne postoji sačuvan model na putanji {model_path}! Prvo pokrenite train.py.")
        return
        
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Uspešno učitan sačuvani OCR model! Pokrećem testiranje na test skupu...\n")
    
    # 3. Evaluacija kroz test slike
    brojac = 0
    with torch.no_grad():
        for images, labels, _ in test_loader:
            if brojac >= 5: # Ispisujemo samo prvih 5 slika da ne zatrpavamo terminal
                break
                
            outputs = model(images)
            predikcija = decode_predictions(outputs)
            
            # Vraćamo prave brojeve natrag u tekst
            pravi_tekst = "".join([config.CHARACTERS[idx] for idx in labels[0].tolist()])
            
            print(f"Knjiga {brojac + 1}:")
            print(f"  STVARNO:   '{pravi_tekst}'")
            print(f"  PREDIKCIJA: '{predikcija}'")
            print("-" * 30)
            brojac += 1

if __name__ == "__main__":
    main()