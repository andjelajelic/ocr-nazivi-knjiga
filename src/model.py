# src/model.py
import torch
import torch.nn as nn
import config

class BidirectionalLSTM(nn.Module):
    """Wrapper oko LSTM-a koji ispravno raspakuje tuple (output, hidden)."""
    def __init__(self, input_size, hidden_size, num_layers):
        super(BidirectionalLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=False,
            dropout=0.3 if num_layers > 1 else 0.0
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class CRNN(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(CRNN, self).__init__()

        # 1. CNN - Izvlačenje vizuelnih karakteristika
        self.cnn = nn.Sequential(
            # Blok 1: [1, 64, 256] -> [64, 32, 128]
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Blok 2: [64, 32, 128] -> [128, 16, 64]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Blok 3: [128, 16, 64] -> [256, 8, 64]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Blok 4: [256, 8, 64] -> [512, 4, 64]
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Blok 5: [512, 4, 64] -> [512, 1, 64]
            nn.Conv2d(512, 512, kernel_size=(4, 1), stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # 2. RNN - Bidirectional LSTM sa ispravnim wrapperom
        self.rnn = BidirectionalLSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2
        )

        # 3. Izlazni sloj: 256*2 (bidirectional) -> num_classes
        self.fc = nn.Linear(512, num_classes)

        # Inicijalizacija težina
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # CNN: [B, 1, 64, 256] -> [B, 512, 1, W]
        features = self.cnn(x)

        # Uklanjamo visinu (koja je sada 1): [B, 512, 1, W] -> [B, 512, W]
        features = features.squeeze(2)

        # Permutujemo za RNN: [B, 512, W] -> [W, B, 512]  (T, B, C format)
        features = features.permute(2, 0, 1)

        # RNN: [W, B, 512] -> [W, B, 512]
        rnn_out = self.rnn(features)

        # Linearni sloj: [W, B, 512] -> [W, B, num_classes]
        out = self.fc(rnn_out)

        # Log-softmax za CTC loss
        out = out.log_softmax(2)

        return out


if __name__ == "__main__":
    model = CRNN()
    probna_slika = torch.randn(1, 1, 64, 256)
    izlaz = model(probna_slika)
    print("Mreža uspešno kreirana!")
    print(f"Dimenzije izlaza: {izlaz.shape}")
    print(f"Očekivano: [64, 1, {config.NUM_CLASSES}]")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Ukupan broj parametara: {total_params:,}")
