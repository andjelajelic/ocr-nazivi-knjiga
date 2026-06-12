# src/test.py
import os
import torch
from torch.utils.data import DataLoader
import config
from dataset_loader import BookOCRDataset, collate_fn
from model import CRNN
from train import decode_predictions, compute_cer


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Koristim uređaj: {device}")

    test_dir = os.path.join(os.getcwd(), 'dataset/test')
    if not os.path.exists(test_dir):
        test_dir = os.path.join(os.getcwd(), 'dataset/train')
        print("Nema 'test' foldera, koristim 'train' za evaluaciju.\n")

    test_dataset = BookOCRDataset(test_dir, augment=False)
    print(f"Broj slika za testiranje: {len(test_dataset)}\n")

    model = CRNN().to(device)
    model_path = os.path.join(os.getcwd(), 'ocr_model.pth')

    if not os.path.exists(model_path):
        print(f"GREŠKA: Model nije pronađen na: {model_path}")
        print("Pokreni train.py prvo!")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model učitan. Pokrećem evaluaciju...\n")
    print("=" * 55)

    image_paths = test_dataset.image_paths
    ukupni_cer = 0.0
    tacnih = 0
    total = min(len(test_dataset), 20)  # Prikazujemo max 20

    with torch.no_grad():
        for i in range(total):
            img, label, _ = test_dataset[i]
            img = img.unsqueeze(0).to(device)

            output = model(img)
            predikcija = decode_predictions(output)
            stvarno = "".join([config.CHARACTERS[idx] for idx in label.tolist()])

            cer = compute_cer(predikcija, stvarno)
            ukupni_cer += cer

            if predikcija == stvarno:
                tacnih += 1
                status = "✓"
            else:
                status = "✗"

            ime_fajla = os.path.basename(image_paths[i])
            print(f"{status} [{ime_fajla}]")
            print(f"  STVARNO:    '{stvarno}'")
            print(f"  PREDIKCIJA: '{predikcija}'")
            print(f"  CER: {cer*100:.1f}%")
            print("-" * 40)

    print(f"\n{'='*55}")
    print(f"REZULTATI NA {total} SLIKA:")
    print(f"  Tačnost (exact match): {tacnih}/{total} = {tacnih/total*100:.1f}%")
    print(f"  Prosečan CER: {ukupni_cer/total*100:.1f}%")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
