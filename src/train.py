# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import config
from dataset_loader import BookOCRDataset
from model import CRNN

# Funkcija koja pretvara brojeve (indekse) nazad u tekst koji ljudi razumeju
def decode_predictions(output, characters=config.CHARACTERS):
    # output dimenzije: [Sequence_Length, Batch_Size, Num_Classes]
    # Uzimamo najverovatniji karakter za svaki od 64 slajsa
    prob_indices = torch.argmax(output, dim=2).squeeze(1).tolist()
    
    decoded_text = ""
    prev_char = None
    
    for idx in prob_indices:
        char = characters[idx]
        # CTC Loss pravilo: preskačemo '-' (blank) i duplirane karaktere u nizu
        if char != '-' and char != prev_char:
            decoded_text += char
        prev_char = char
        
    return decoded_text.strip()

def main():
    # 1. Putanje do foldera
    train_dir = os.path.join(os.getcwd(), 'dataset/train')
    val_dir = os.path.join(os.getcwd(), 'dataset/val')
    
    # 2. Učitavanje podataka
    train_dataset = BookOCRDataset(train_dir)
    val_dataset = BookOCRDataset(val_dir)
    
    # DataLoaderi nam služe da automatski pakuju podatke (kod nas je batch_size=1 jer su tekstovi različitih dužina)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # 3. Inicijalizacija modela, funkcije gubitka (Loss) i optimizatora
    model = CRNN()
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # --- ZAHTEV ASISTENTA: Predikcija PRE treninga ---
    print("\n" + "="*50)
    print("DEMONSTRACIJA: Ponašanje modela PRE početka treninga")
    print("="*50)
    
    model.eval() # Prebacujemo model u mod za evaluaciju
    with torch.no_grad():
        # Uzimamo prvu sliku iz trening skupa za probu
        probna_slika, pravi_label_kod, _ = train_dataset[0]
        probna_slika = probna_slika.unsqueeze(0) # Dodajemo batch dimenziju
        
        # Propuštamo kroz netreniranu mrežu
        izlaz_pre_treninga = model(probna_slika)
        predikcija_pre = decode_predictions(izlaz_pre_treninga)
        
        # Vraćamo prave brojeve nazad u tekst radi ispisa
        pravi_tekst = "".join([config.CHARACTERS[idx] for idx in pravi_label_kod.tolist()])
        
        print(f"STVARNI TEKST NA KORICI:  '{pravi_tekst}'")
        print(f"PREDIKCIJA NETRENIRANE MREŽE: '{predikcija_pre}'")
    print("="*50 + "\n")
    
    # 4. Pokretanje PRAVOG treninga na 50 epoha
    BROJ_EPOHA = 50
    print(f"Pokrećem pravi trening na {BROJ_EPOHA} epoha... Ovo može potrajati par minuta.")
    
    for epoch in range(1, BROJ_EPOHA + 1):
        model.train() # Vraćamo model u mod za treniranje
        ukupni_gubitak = 0
        
        for images, labels, label_lengths in train_loader:
            optimizer.zero_grad()
            
            # Prolaz kroz model
            outputs = model(images) # [64, 1, 71]
            
            # CTC Loss očekuje dužinu izlazne sekvence (kod nas je uvek 64)
            input_lengths = torch.tensor([outputs.size(0)], dtype=torch.long)
            
            # Računanje greške
            loss = criterion(outputs, labels, input_lengths, label_lengths)
            
            # Backpropagation (učenje)
            loss.backward()
            optimizer.step()
            
            ukupni_gubitak += loss.item()
            
        prosecni_loss = ukupni_gubitak / len(train_loader)
        
        # Ispisujemo napredak na svakih 5 epoha (ili na prvoj i poslednjoj) da ne zatrpavamo terminal
        if epoch == 1 or epoch % 5 == 0 or epoch == BROJ_EPOHA:
            print(f"Epoha [{epoch}/{BROJ_EPOHA}] -> Prosečan Loss (Greška): {prosecni_loss:.4f}")
            
    # 5. Čuvanje naučenih težina modela
    model_save_path = os.path.join(os.getcwd(), 'ocr_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"\nTrening završen! Model uspešno sačuvan na putanji: {model_save_path}")

if __name__ == "__main__":
    main()