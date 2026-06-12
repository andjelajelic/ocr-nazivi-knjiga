# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import config
from dataset_loader import BookOCRDataset, collate_fn
from model import CRNN


def decode_predictions(output, characters=config.CHARACTERS):
    """
    Greedy CTC decode.
    output: [T, B, C] tensor sa log-verovatnoćama
    """
    # Uzimamo argmax po dimenziji klasa
    # output: [T, B, C] -> argmax -> [T, B]
    prob_indices = output.argmax(dim=2)  # [T, B]

    results = []
    batch_size = prob_indices.shape[1]

    for b in range(batch_size):
        indices = prob_indices[:, b].tolist()  # [T]
        decoded = []
        prev_idx = None

        for idx in indices:
            # CTC pravilo: preskoči blank (0) i uzastopne iste karaktere
            if idx != 0 and idx != prev_idx:
                if idx < len(characters):
                    decoded.append(characters[idx])
            prev_idx = idx

        results.append("".join(decoded).strip())

    return results[0] if batch_size == 1 else results


def compute_cer(predicted, actual):
    """Character Error Rate - meri koliko je blizu predikcija stvarnom tekstu."""
    if len(actual) == 0:
        return 0.0 if len(predicted) == 0 else 1.0

    # Levenshtein distanca
    m, n = len(actual), len(predicted)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        new_dp = [i] + [0] * n
        for j in range(1, n + 1):
            if actual[i-1] == predicted[j-1]:
                new_dp[j] = dp[j-1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j-1], dp[j-1])
        dp = new_dp

    return dp[n] / len(actual)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Koristim uređaj: {device}")

    train_dir = os.path.join(os.getcwd(), 'dataset/train')
    val_dir = os.path.join(os.getcwd(), 'dataset/val')
    if not os.path.exists(val_dir):
        val_dir = train_dir
        print("NAPOMENA: Nema 'val' foldera, koristim 'train' za validaciju.")

    train_dataset = BookOCRDataset(train_dir, augment=True)
    val_dataset = BookOCRDataset(val_dir, augment=False)

    print(f"Train skup: {len(train_dataset)} slika")
    print(f"Val skup:   {len(val_dataset)} slika")

    if len(train_dataset) == 0:
        print("GREŠKA: Dataset je prazan! Proveri putanju dataset/train.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,          # batch > 1 stabilizuje trening
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    model = CRNN().to(device)

    # Koristimo CTCLoss sa reduction='mean'
    # VAŽNO: expects log_softmax ulaz (što model sada vraća)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # LR scheduler: smanjuje lr kada loss stagnira
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5
    )

    # --- PRE-TRENING DEMO ---
    print("\n" + "="*55)
    print("PONAŠANJE MODELA PRE TRENINGA")
    print("="*55)
    model.eval()
    with torch.no_grad():
        v_img, v_label, _ = val_dataset[0]
        v_img = v_img.unsqueeze(0).to(device)
        izlaz_pre = model(v_img)
        pred_pre = decode_predictions(izlaz_pre)
        pravi = "".join([config.CHARACTERS[i] for i in v_label.tolist()])
        print(f"STVARNO:    '{pravi}'")
        print(f"PREDIKCIJA: '{pred_pre}' (očekivano: besmislica)")
    print("="*55 + "\n")

    BROJ_EPOHA = 150
    best_val_loss = float('inf')
    model_save_path = os.path.join(os.getcwd(), 'ocr_model.pth')

    print(f"Pokrećem trening na {BROJ_EPOHA} epoha...\n")

    for epoch in range(1, BROJ_EPOHA + 1):
        # --- TRENING FAZA ---
        model.train()
        ukupni_gubitak = 0.0
        broj_batch = 0

        for images, labels, label_lengths in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # [T, B, C]

            T = outputs.size(0)
            B = outputs.size(1)
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

            loss = criterion(outputs, labels, input_lengths, label_lengths)

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                # Gradient clipping - sprečava eksplodiranje gradijenata
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                ukupni_gubitak += loss.item()
                broj_batch += 1

        prosecni_loss = ukupni_gubitak / max(broj_batch, 1)
        scheduler.step(prosecni_loss)

        # --- VALIDACIJA I ISPIS ---
        if epoch == 1 or epoch % 10 == 0 or epoch == BROJ_EPOHA:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n[Epoha {epoch:3d}/{BROJ_EPOHA}] Loss: {prosecni_loss:.4f} | LR: {current_lr:.6f}")

            model.eval()
            val_preds = []
            val_actuals = []

            with torch.no_grad():
                # Prikazujemo prve 3 primera iz val skupa
                num_prikaz = min(3, len(val_dataset))
                for i in range(num_prikaz):
                    v_img, v_label, _ = val_dataset[i]
                    v_img = v_img.unsqueeze(0).to(device)

                    v_output = model(v_img)
                    predikcija = decode_predictions(v_output)
                    stvarno = "".join([config.CHARACTERS[idx] for idx in v_label.tolist()])

                    cer = compute_cer(predikcija, stvarno)
                    val_preds.append(predikcija)
                    val_actuals.append(stvarno)

                    if i == 0:  # Detaljan prikaz samo za prvi primer
                        print(f"  Primer {i+1}:")
                        print(f"    STVARNO:    '{stvarno}'")
                        print(f"    PREDIKCIJA: '{predikcija}'")
                        print(f"    CER: {cer*100:.1f}%")

                avg_cer = sum(compute_cer(p, a) for p, a in zip(val_preds, val_actuals)) / len(val_preds)
                print(f"  Prosečan CER na val skupu: {avg_cer*100:.1f}%")

            # Čuvamo najbolji model
            if prosecni_loss < best_val_loss:
                best_val_loss = prosecni_loss
                torch.save(model.state_dict(), model_save_path)
                print(f"  ✓ Novi najbolji model sačuvan (loss: {best_val_loss:.4f})")

            print("-" * 45)
            model.train()

    print(f"\nTrening završen! Najbolji model sačuvan na: {model_save_path}")
    print(f"Najmanji loss postignut: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
