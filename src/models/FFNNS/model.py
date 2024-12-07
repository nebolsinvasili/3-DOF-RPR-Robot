import torch.nn as nn


class FFNNS_model(nn.Module):
    def __init__(self):
        super(FFNNS_model, self).__init__()
        self.fc1 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(3, 1)
        self.relu = nn.ReLU()  # Активационная функция ReLU

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Входной слой + ReLU
        x = self.fc2(x)  # Скрытый слой + ReLU
        return x
