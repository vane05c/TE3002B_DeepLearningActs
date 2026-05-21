import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# =========================================================
# DEVICE CONFIGURATION
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing device: {DEVICE}")


# =========================================================
# HYPERPARAMETERS
# =========================================================

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# =========================================================
# FILE PATHS
# =========================================================

TWO_CLASS_PATH = r"C:\Users\anarg\Downloads\two-class.txt"

FOUR_CLASS_PATH = r"C:\Users\anarg\Downloads\dataset-with-four-classes.txt"

PARKINSON_PATH = r"C:\Users\anarg\Downloads\parkinsons_updrs.data"


# =========================================================
# MLP CLASSIFIER
# =========================================================

class MLPClassifier(nn.Module):

    def __init__(self, input_size, num_classes):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes)

        )

    def forward(self, x):

        return self.network(x)


# =========================================================
# MLP REGRESSOR
# =========================================================

class MLPRegressor(nn.Module):

    def __init__(self, input_size, output_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_size)

        )

    def forward(self, x):

        return self.network(x)


# =========================================================
# TRAINING FUNCTION
# =========================================================

def train_model(model, train_loader, criterion, optimizer):

    model.train()

    loss_history = []

    for epoch in range(EPOCHS):

        total_loss = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:

            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

    return loss_history


# =========================================================
# BINARY CLASSIFICATION
# =========================================================

def binary_classification():

    print("\n====================================")
    print("1. BINARY CLASSIFICATION")
    print("====================================")

    data = np.loadtxt(TWO_CLASS_PATH)

    y = data[:, 0]
    X = data[:, 1:]

    y = y.astype(int)

    y = y - np.min(y)

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    accuracies = []

    fold = 1

    last_loss_history = None

    for train_idx, test_idx in skf.split(X, y):

        print(f"\nFold {fold}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)

        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(X_train, y_train)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        model = MLPClassifier(
            input_size=X.shape[1],
            num_classes=2
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )

        loss_history = train_model(
            model,
            train_loader,
            criterion,
            optimizer
        )

        last_loss_history = loss_history

        model.eval()

        with torch.no_grad():

            outputs = model(X_test.to(DEVICE))

            predictions = torch.argmax(outputs, dim=1)

            accuracy = accuracy_score(
                y_test.cpu(),
                predictions.cpu()
            )

            accuracies.append(accuracy)

            print(f"Accuracy: {accuracy:.4f}")

        fold += 1

    print("\nFinal Results")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f}")

    # =====================================================
    # ACCURACY GRAPH
    # =====================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, 6),
        accuracies,
        marker='o'
    )

    plt.title("Binary Classification Accuracy per Fold")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")

    plt.grid(True)

    plt.savefig("binary_accuracy.png")

    plt.show()

    # =====================================================
    # LOSS GRAPH
    # =====================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, EPOCHS + 1),
        last_loss_history
    )

    plt.title("Binary Classification Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.grid(True)

    plt.savefig("binary_loss.png")

    plt.show()


# =========================================================
# FOUR-CLASS CLASSIFICATION
# =========================================================

def four_class_classification():

    print("\n====================================")
    print("2. FOUR-CLASS CLASSIFICATION")
    print("====================================")

    data = np.loadtxt(FOUR_CLASS_PATH)

    y = data[:, 0]
    X = data[:, 1:]

    y = y.astype(int)

    y = y - np.min(y)

    num_classes = len(np.unique(y))

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    accuracies = []

    fold = 1

    last_loss_history = None

    for train_idx, test_idx in skf.split(X, y):

        print(f"\nFold {fold}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)

        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(X_train, y_train)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        model = MLPClassifier(
            input_size=X.shape[1],
            num_classes=num_classes
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )

        loss_history = train_model(
            model,
            train_loader,
            criterion,
            optimizer
        )

        last_loss_history = loss_history

        model.eval()

        with torch.no_grad():

            outputs = model(X_test.to(DEVICE))

            predictions = torch.argmax(outputs, dim=1)

            accuracy = accuracy_score(
                y_test.cpu(),
                predictions.cpu()
            )

            accuracies.append(accuracy)

            print(f"Accuracy: {accuracy:.4f}")

        fold += 1

    print("\nFinal Results")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f}")

    # =====================================================
    # ACCURACY GRAPH
    # =====================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, 6),
        accuracies,
        marker='o'
    )

    plt.title("Four-Class Accuracy per Fold")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")

    plt.grid(True)

    plt.savefig("fourclass_accuracy.png")

    plt.show()

    # =====================================================
    # LOSS GRAPH
    # =====================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, EPOCHS + 1),
        last_loss_history
    )

    plt.title("Four-Class Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.grid(True)

    plt.savefig("fourclass_loss.png")

    plt.show()


# =========================================================
# PARKINSON REGRESSION
# =========================================================

def parkinson_regression():

    print("\n====================================")
    print("3. PARKINSON REGRESSION")
    print("====================================")

    df = pd.read_csv(PARKINSON_PATH)

    X = df.drop(
        columns=[
            "subject#",
            "motor_UPDRS",
            "total_UPDRS"
        ]
    )

    y = df[[
        "motor_UPDRS",
        "total_UPDRS"
    ]]

    X = X.values
    y = y.values

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    mse_scores = []

    fold = 1

    last_loss_history = None

    for train_idx, test_idx in kf.split(X):

        print(f"\nFold {fold}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler_X = StandardScaler()

        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)

        scaler_y = StandardScaler()

        y_train = scaler_y.fit_transform(y_train)
        y_test = scaler_y.transform(y_test)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)

        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TensorDataset(X_train, y_train)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        model = MLPRegressor(
            input_size=X.shape[1],
            output_size=2
        ).to(DEVICE)

        criterion = nn.MSELoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )

        loss_history = train_model(
            model,
            train_loader,
            criterion,
            optimizer
        )

        last_loss_history = loss_history

        model.eval()

        with torch.no_grad():

            predictions = model(X_test.to(DEVICE))

            mse = mean_squared_error(
                y_test.cpu(),
                predictions.cpu()
            )

            mse_scores.append(mse)

            print(f"MSE: {mse:.4f}")

        fold += 1

    print("\nFinal Results")
    print(f"Mean MSE: {np.mean(mse_scores):.4f}")

    # =====================================================
    # MSE GRAPH
    # =====================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, 6),
        mse_scores,
        marker='o'
    )

    plt.title("Parkinson Regression MSE per Fold")
    plt.xlabel("Fold")
    plt.ylabel("MSE")

    plt.grid(True)

    plt.savefig("parkinson_mse.png")

    plt.show()

    # =====================================================
    # LOSS GRAPH
    # =====================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, EPOCHS + 1),
        last_loss_history
    )

    plt.title("Parkinson Regression Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.grid(True)

    plt.savefig("parkinson_loss.png")

    plt.show()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    binary_classification()

    four_class_classification()

    parkinson_regression()