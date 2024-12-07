from models.metrics import accuracy
import torch
from torch import nn, optim
from torchsummary import summary

from .trainer import ModelTrainer
from .data import (
    X_test,
    X_train,
    X_val,
    y_test,
    y_train,
    y_val,
)
from .FFNNS.model import FFNNS_model

print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


RANDOM_SEED = 42
EPOCHS = 20
BATCH_SIZE = 32

torch.manual_seed(RANDOM_SEED)

models = {"FFNNS": FFNNS_model().to(device)}
for name, model in models.items():
    summary(model, X_train.size())

    MSE = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    trainer = ModelTrainer(model=model, loss_fn=MSE, optimizer=optimizer, metrics={'accuracy': accuracy}, device=device)
    trainer.fit(
        train_data=(X_train, y_train),
        valid_data=(X_val, y_val),
        test_data=(X_test, y_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        save_path=rf"src\models\{name}",
        model_name=name,
    )
