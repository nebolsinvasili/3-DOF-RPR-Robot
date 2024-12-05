import torch
from torch import nn, optim

from ..trainer import ModelTrainer
from .data import (
    X_train,
    X_val,
    y_train,
    y_val,
)
from .model import LegModel

print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


RANDOM_SEED = 42
EPOCHS = 20
BATCH_SIZE = 8

torch.manual_seed(RANDOM_SEED)


model = LegModel().to(device)

MSE = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

trainer = ModelTrainer(model=model, loss_fn=MSE, optimizer=optimizer, device=device)
trainer.fit(
    train_data=(X_train, y_train),
    valid_data=(X_val, y_val),
    epochs=EPOCHS,
    save_path=r"C:\Users\nebolsinvasili\Documents\projects\python\rpr\src\models\dense\model",
    model_name="Dense_1",
)
