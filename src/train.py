# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import config
from dataset_loader import BookOCRDataset
from model import CRNN

def decode_predictions(output, characters=config.CHARACTERS):
    """Pretvara izlaz iz mreže (verovatnoće) u čitljiv tekst (Greedy Decode)."""
    prob_indices = torch.argmax(output, dim=2)
    if prob_indices.dim() > 1:
        prob_indices = prob_indices.squeeze(1)
        
    prob_indices = prob_indices.tolist()
    decoded_text = ""
    prev_char = None
    for idx in prob_indices:
        char = characters[idx]
        if char != '-' and char != prev_char:
            decoded_text += char
        prev_char = char
    return decoded_text.strip()

def main():
    train_dir = os.path.join(os.getcwd(), 'dataset/train')
    
    # Ako nemate 'val' folder, koristimo 'train' da bismo pratili kako model uči
    val_dir = os.path.join(os.getcwd(), 'dataset/val')
    if not os.path.exists(val_dir):
        val_dir = train_dir

    train_dataset = BookOCRDataset(train_dir)
    val_dataset = BookOCRDataset(val_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    model = CRNN()
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    
    print("\n" + "="*50)
    print("DEMONSTRACIJA: Ponašanje modela PRE početka treninga")
    print("="*50)
    
    model.eval()
    with torch.no_grad():
        if len(train_dataset) > 0:
            probna_slika, pravi_label_kod, _ = train_dataset[0]
            probna_slika = probna_slika.unsqueeze(0)
            izlaz_pre_treninga = model(probna_slika)
            predikcija_pre = decode_predictions(izlaz_pre_treninga)
            pravi_tekst = "".join([config.CHARACTERS[idx] for idx in pravi_label_kod.tolist()])
            print(f"STVARNI TEKST: '{pravi_tekst}'")
            print(f"PREDIKCIJA:    '{predikcija_pre}' (Očekivano je da bude prazno ili glupost)")
        else:
            print("Dataset je prazan! Proveri putanju dataset/train.")
            return
    print("="*50 + "\n")
    
    BROJ_EPOHA = 150
    print(f"Pokrećem trening na {BROJ_EPOHA} epoha...")
    
    for epoch in range(1, BROJ_EPOHA + 1):
        model.train()
        ukupni_gubitak = 0
        
        for images, labels, label_lengths in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            input_lengths = torch.tensor([outputs.size(0)], dtype=torch.long)
            loss = criterion(outputs, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            ukupni_gubitak += loss.item()
            
        prosecni_loss = ukupni_gubitak / len(train_loader)
        
        # --- ŽIVA VALIDACIJA I ISPIS NAPRETKA ---
        if epoch == 1 or epoch % 10 == 0 or epoch == BROJ_EPOHA:
            print(f"\n[Epoha {epoch}/{BROJ_EPOHA}] -> Prosečan Loss (Greška): {prosecni_loss:.4f}")
            
            # Pokretanje brze provere na jednom primeru iz val skupa bez računanja gradijenata
            model.eval()
            with torch.no_grad():
                # Uzimamo prvu sliku iz validacije za test
                v_img, v_label, _ = val_dataset[0]
                v_img = v_img.unsqueeze(0) # Dodajemo batch dimenziju
                
                v_output = model(v_img)
                ziva_predikcija = decode_predictions(v_output)
                stvarni_tekst_val = "".join([config.CHARACTERS[idx] for idx in v_label.tolist()])
                
                print(f"  [ŽIVI MONITOR] -> Slika 1:")
                print(f"    STVARNO:    '{stvarni_tekst_val}'")
                print(f"    MREŽA KAŽE: '{ziva_predikcija}'")
                print("-" * 30)
            
    model_save_path = os.path.join(os.getcwd(), 'ocr_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"\nTrening završen! Model sačuvan na: {model_save_path}")

if __name__ == "__main__":
    main()