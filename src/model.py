# src/model.py
import torch
import torch.nn as nn
import config

class CRNN(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(CRNN, self).__init__()
        
        # 1. CNN - Izvlačenje vizuelnih karakteristika iz slike
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), 
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), 
            
            nn.Conv2d(512, 512, kernel_size=(4, 1), stride=1), 
            nn.ReLU(inplace=True)
        )
        
        # 2. RNN - Dva sloja Bidirectional LSTM-a
        self.rnn = nn.Sequential(
            nn.LSTM(input_size=512, hidden_size=256, num_layers=2, bidirectional=True, batch_first=False)
        )
        
        # 3. Linearni izlazni sloj
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        features = self.cnn(x) 
        features = features.squeeze(2) 
        features = features.permute(2, 0, 1) 
        lstm_out, _ = self.rnn(features) 
        out = self.fc(lstm_out) 
        return out

if __name__ == "__main__":
    model = CRNN()
    probna_slika = torch.randn(1, 1, 64, 256)
    izlaz = model(probna_slika)
    print("Mreža uspešno kreirana!")
    print(f"Dimenzije izlaza iz mreže: {izlaz.shape} (Očekivano: [64, 1, Broj_Klasa])")