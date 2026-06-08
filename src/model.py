# src/model.py
import torch
import torch.nn as nn
import config

class CRNN(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(CRNN, self).__init__()
        
        # 1. CNN - Izvlačenje vizuelnih karakteristika iz slike
        self.cnn = nn.Sequential(
            # Prvi blok: ulaz je 1 kanal (crno-belo), izlaz 64 kanala
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Slika postaje 32x128
            
            # Drugi blok
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Slika postaje 16x64
            
            # Treći blok
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Koristimo nesimetričan pooling da očuvamo horizontalnu rezoluciju (širinu reči)
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), # Slika postaje 8x64
            
            # Četvrti blok
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), # Slika postaje 4x64
            
            # Peti blok (Završna konvolucija bez paddinga po visini)
            nn.Conv2d(512, 512, kernel_size=(4, 1), stride=1), # Slika postaje 1x64
            nn.ReLU(inplace=True)
        )
        
        # 2. RNN - Dva sloja Bidirectional LSTM-a za prepoznavanje sekvence slova
        # Ulazna veličina je 512 (broj kanala iz CNN-a), hidden size je 256
        self.rnn = nn.Sequential(
            nn.LSTM(input_size=512, hidden_size=256, num_layers=2, bidirectional=True, batch_first=False)
        )
        
        # 3. Linearni izlazni sloj - mapira izlaz iz LSTM-a u naše klase (slova iz rečnika)
        # Pošto je LSTM bidirectionalan, njegova izlazna veličina je hidden_size * 2 = 512
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # x ima dimenzije: [Batch_Size, 1, 64, 256]
        
        # Prolazak kroz CNN
        features = self.cnn(x) # Izlaz: [Batch_Size, 512, 1, 64]
        
        # Uklanjamo dimenziju visine koja je sada 1
        features = features.squeeze(2) # Izlaz: [Batch_Size, 512, 64]
        
        # Menjamo raspored dimenzija za LSTM koji očekuje: [Sequence_Length, Batch_Size, Features]
        # Kod nas je Sequence_Length zapravo širina slike kroz koju klizimo (64 slajsa)
        features = features.permute(2, 0, 1) # Izlaz: [64, Batch_Size, 512]
        
        # Prolazak kroz LSTM
        lstm_out, _ = self.rnn(features) # Izlaz: [64, Batch_Size, 512]
        
        # Prolazak kroz finalni linearni sloj da dobijemo verovatnoće za svako slovo
        out = self.fc(lstm_out) # Izlaz: [64, Batch_Size, NUM_CLASSES]
        
        return out

# Kratak test arhitekture
if __name__ == "__main__":
    model = CRNN()
    # Pravimo lažnu sliku (random brojevi) dimenzija kao naša knjiga da vidimo da li prolazi kroz mrežu
    probna_slika = torch.randn(1, 1, 64, 256)
    izlaz = model(probna_slika)
    print("Mreža uspešno kreirana!")
    print(f"Dimenzije izlaza iz mreže: {izlaz.shape} (Očekivano: [64, 1, Broj_Klasa])")