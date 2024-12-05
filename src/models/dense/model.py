import torch.nn as nn


class LegModel(nn.Module):
    def __init__(self):
        super(LegModel, self).__init__()
        self.fc1 = nn.Linear(3, 1024 * 2)  # Входной слой: 3 -> 128
        self.fc2 = nn.Linear(1024 * 2, 1024 * 3)  # Скрытый слой: 128 -> 256
        self.fcc = nn.Linear(1024 * 3, 1024 * 4)  # Скрытый слой: 128 -> 256
        self.fc3 = nn.Linear(1024 * 4, 1024 * 5)  # Скрытый слой: 256 -> 512
        self.fc4 = nn.Linear(1024 * 5, 1)  # Выходной слой: 512 -> 1
        self.relu = nn.ReLU()  # Активационная функция ReLU

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Входной слой + ReLU
        x = self.relu(self.fc2(x))  # Скрытый слой + ReLU
        x = self.relu(self.fcc(x))
        x = self.relu(self.fc3(x))  # Скрытый слой + ReLU
        x = self.fc4(x)  # Выходной слой (линейная активация)
        return x
