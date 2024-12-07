import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data_path = (
    r"C:\Users\nebolsinvasili\Documents\projects\python\rpr\src\models\data\test.csv"
)

df = pd.read_csv(data_path, sep=",", parse_dates=True)

# Выделяем входные и выходные данные
X = df[["x", "y", "fi"]].values  # Входные признаки
y = df["Ld_1"].values  # Целевые значения

# Нормализация данных
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_X.fit_transform(X)  # Нормализуем входные данные
y = y.reshape(-1, 1)  # Нормализуем целевые данные

# Разделение на тренировочные, проверочные и тестовые наборы
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.8, shuffle=True, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.1, shuffle=True, random_state=42
)

# Преобразуем данные в тензоры
X_train, y_train = (
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)
X_val, y_val = (
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32),
)
X_test, y_test = (
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32),
)
