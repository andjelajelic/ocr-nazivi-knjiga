# src/test.py

import os
from collections import defaultdict

import torch

import config
from dataset_loader import BookOCRDataset
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

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.eval()

    print("Model učitan. Pokrećem evaluaciju...\n")
    print("=" * 55)

    image_paths = test_dataset.image_paths

    ukupni_cer = 0.0
    tacnih = 0

    total = min(len(test_dataset), 20)

    # =====================================================
    # EVALUACIJA PO LINIJAMA
    # =====================================================

    with torch.no_grad():
        for i in range(total):

            img, label, _ = test_dataset[i]

            img = img.unsqueeze(0).to(device)

            output = model(img)

            predikcija = decode_predictions(output)

            stvarno = "".join(
                [config.CHARACTERS[idx] for idx in label.tolist()]
            )

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
            print(f"  CER: {cer * 100:.1f}%")
            print("-" * 40)

    print(f"\n{'=' * 55}")
    print(f"REZULTATI NA {total} SLIKA:")
    print(
        f"  Tačnost (exact match): "
        f"{tacnih}/{total} = {tacnih / total * 100:.1f}%"
    )
    print(
        f"  Prosečan CER: "
        f"{ukupni_cer / total * 100:.1f}%"
    )
    print(f"{'=' * 55}")

    # =====================================================
    # REKONSTRUKCIJA KNJIGA
    # =====================================================

    print("\n")
    print("=" * 55)
    print("REKONSTRUKCIJA KNJIGA")
    print("=" * 55)

    knjige_pred = defaultdict(list)
    knjige_true = defaultdict(list)

    with torch.no_grad():

        for i in range(len(test_dataset)):

            img, label, _ = test_dataset[i]

            img = img.unsqueeze(0).to(device)

            output = model(img)

            predikcija = decode_predictions(output)

            stvarno = "".join(
                [config.CHARACTERS[idx] for idx in label.tolist()]
            )

            ime_fajla = os.path.basename(image_paths[i])

            try:
                # 299_L01.jpg -> 299
                knjiga_id = ime_fajla.split("_")[0]

                # L01.jpg -> 01
                red_deo = ime_fajla.split("_")[1]

                redni_broj = int(
                    ''.join(filter(str.isdigit, red_deo))
                )

            except Exception:
                print(
                    f"Ne mogu da parsiram ime fajla: {ime_fajla}"
                )
                continue

            knjige_pred[knjiga_id].append(
                (redni_broj, predikcija)
            )

            knjige_true[knjiga_id].append(
                (redni_broj, stvarno)
            )

    broj_tacnih_knjiga = 0
    ukupno_knjiga = len(knjige_pred)

    for knjiga_id in sorted(knjige_pred.keys()):

        pred_linije = sorted(
            knjige_pred[knjiga_id],
            key=lambda x: x[0]
        )

        true_linije = sorted(
            knjige_true[knjiga_id],
            key=lambda x: x[0]
        )

        pred_tekst = " ".join(
            [tekst for _, tekst in pred_linije]
        )

        true_tekst = " ".join(
            [tekst for _, tekst in true_linije]
        )

        knjiga_cer = compute_cer(
            pred_tekst,
            true_tekst
        )

        if pred_tekst == true_tekst:
            broj_tacnih_knjiga += 1
            status = "✓"
        else:
            status = "✗"

        print(f"\n{status} KNJIGA {knjiga_id}")

        print("\nSTVARNO:")
        print(true_tekst)

        print("\nPREDIKCIJA:")
        print(pred_tekst)

        print(f"\nCER: {knjiga_cer * 100:.1f}%")

        print("-" * 55)

    print("\n")
    print("=" * 55)
    print("REZULTATI PO KNJIGAMA")
    print("=" * 55)

    print(
        f"Tačno rekonstruisanih knjiga: "
        f"{broj_tacnih_knjiga}/{ukupno_knjiga}"
    )

    if ukupno_knjiga > 0:
        print(
            f"Accuracy: "
            f"{100 * broj_tacnih_knjiga / ukupno_knjiga:.1f}%"
        )

    print("=" * 55)


if __name__ == "__main__":
    main()