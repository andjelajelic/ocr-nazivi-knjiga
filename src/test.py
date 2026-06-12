# src/test.py
import os
import torch
from torch.utils.data import DataLoader
import config
from dataset_loader import BookOCRDataset
from model import CRNN

def decode_ctc_prediction(output, characters=config.CHARACTERS):
    """Pravilno dekodiranje CTC izlaza za jednu sliku/isečak."""
    prob_indices = torch.argmax(output, dim=2)[:, 0]
    indices_list = prob_indices.tolist()
    
    decoded_text = ""
    prev_idx = None
    for idx in indices_list:
        if idx == prev_idx or idx >= len(characters):
            prev_idx = idx
            continue
        char = characters[idx]
        if char != '-':
            decoded_text += char
        prev_idx = idx
    return decoded_text.strip()

def get_book_id(filename):
    """Izvlači ID knjige iz naziva fajla (npr. '19_L02.jpg' -> '19')."""
    basename = os.path.basename(filename)
    return basename.split('_')[0]

def main():
    test_dir = os.path.join(os.getcwd(), 'dataset/test')
    if not os.path.exists(test_dir):
        test_dir = os.path.join(os.getcwd(), 'dataset/train')
        print("⚠️  Upozorenje: 'dataset/test' ne postoji. Testiramo na TRAIN skupu sa grupisanjem.\n")
        
    test_dataset = BookOCRDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = CRNN()
    model_path = os.path.join(os.getcwd(), 'ocr_model.pth')
    
    if not os.path.exists(model_path):
        print(f"❌ Greška: Ne postoji sačuvan model na putanji {model_path}!")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("🚀 Model uspešno učitan! Grupišem isečke po ID-u knjige...\n")
    
    # Rečnici u koje ćemo pakovati podatke za svaku knjigu posebno
    knjige_stvarno = {}      # { '1': "Stvarni Naslov Iz Delova" }
    knjige_predikcija = {}   # { '1': "Predvidjeni Naslov Iz Delova" }
    
    brojac = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch[0]
            labels = batch[1]
            
            # 1. Saznajemo kojoj knjizi pripada ovaj isečak
            putanja_slike = test_dataset.image_paths[brojac]
            knjiga_id = get_book_id(putanja_slike)
            
            # 2. Pustimo isečak kroz model i dekodiramo tekst
            outputs = model(images)
            predikcija_isecka = decode_ctc_prediction(outputs)
            stvarni_tekst_isecka = "".join([config.CHARACTERS[idx] for idx in labels[0].tolist() if idx < len(config.CHARACTERS)])
            
            # 3. Ako prvi put vidimo ovu knjigu, napravi prazan string, inače dodaj razmak i nadoveži
            if knjiga_id not in knjige_stvarno:
                knjige_stvarno[knjiga_id] = stvarni_tekst_isecka
                knjige_predikcija[knjiga_id] = predikcija_isecka
            else:
                knjige_stvarno[knjiga_id] += " " + stvarni_tekst_isecka
                knjige_predikcija[knjiga_id] += " " + predikcija_isecka
                
            brojac += 1

    # 4. Lep ispis grupisanih rezultata (prikazaćemo prvih 5 kompletnih knjiga)
    prikazano_knjiga = 0
    for kid in sorted(knjige_stvarno.keys(), key=lambda x: int(x) if x.isdigit() else x):
        if prikazano_knjiga >= 5:
            break
            
        print(f"📚 KNJIGA sa ID: {kid}")
        print(f"   CELI STVARNI NASLOV:   '{knjige_stvarno[kid]}'")
        print(f"   CELA OCR PREDIKCIJA:   '{knjige_predikcija[kid]}'")
        print("=" * 60)
        prikazano_knjiga += 1

if __name__ == "__main__":
    main()