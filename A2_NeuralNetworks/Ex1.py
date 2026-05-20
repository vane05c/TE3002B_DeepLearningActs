import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD

# =========================================================
# LOAD DATASET
# =========================================================
data = pd.read_csv(
    "A2_NeuralNetworks/misterious_data_1.txt",
    header=None,
    sep=r"\s+"
)

# First column = labels (1/2 → remap to 0/1)
y = data.iloc[:, 0].values
y = np.where(y == 1, 0, 1)  # 1→0, 2→1

# Remaining columns = features (raw, unscaled — scaling is done per fold)
X = data.iloc[:, 1:].values

# =========================================================
# PARAMETERS
# =========================================================
EPOCHS       = 100
K_FOLDS      = 5
LEARNING_RATE = 0.01
BATCH_METHODS = {
    "SGD":       1,
    "MiniBatch": 16,
    "Batch":     len(X)
}

# =========================================================
# FUNCTION: CROSS-VALIDATED TRAINING
# =========================================================
def evaluate_model(model_type, batch_size):
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_errors = []

    for train_idx, test_idx in kf.split(X):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train,     y_test     = y[train_idx], y[test_idx]

        # -----------------------------------------------------
        # Scale INSIDE the fold to avoid data leakage
        # -----------------------------------------------------
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test  = scaler.transform(X_test_raw)      # fit already done on train

        # -----------------------------------------------------
        # Label encoding per model type
        # -----------------------------------------------------
        if model_type == "svm":
            # Hinge loss expects -1 / +1
            y_train_model = np.where(y_train == 0, -1, 1).astype(np.float32)
            y_test_model  = np.where(y_test  == 0, -1, 1).astype(np.float32)
        else:
            y_train_model = y_train.astype(np.float32)
            y_test_model  = y_test.astype(np.float32)

        # -----------------------------------------------------
        # Build model — use Input() layer to suppress warning
        # -----------------------------------------------------
        model = Sequential()
        model.add(Input(shape=(X.shape[1],)))

        if model_type == "perceptron":
            model.add(Dense(1, activation='sigmoid'))
            model.compile(
                optimizer=SGD(learning_rate=LEARNING_RATE),
                loss='binary_crossentropy'
            )
        elif model_type == "svm":
            model.add(Dense(1, activation='linear'))
            model.compile(
                optimizer=SGD(learning_rate=LEARNING_RATE),
                loss='hinge'
            )

        # -----------------------------------------------------
        # Epoch loop
        # -----------------------------------------------------
        epoch_errors = []
        for epoch in range(EPOCHS):
            model.fit(
                X_train, y_train_model,
                epochs=1,
                batch_size=batch_size,
                verbose=0
            )

            predictions = model.predict(X_test, verbose=0).flatten()

            if model_type == "perceptron":
                pred_labels = (predictions > 0.5).astype(int)
                true_labels = y_test_model.astype(int)
            elif model_type == "svm":
                pred_labels = np.where(predictions >= 0, 1, -1).astype(int)
                true_labels = y_test_model.astype(int)

            error = 1 - accuracy_score(true_labels, pred_labels)
            epoch_errors.append(error)

        fold_errors.append(epoch_errors)

    return np.mean(fold_errors, axis=0)

# =========================================================
# RUN ALL EXPERIMENTS
# =========================================================
results = {}

for method, batch_size in BATCH_METHODS.items():
    print(f"Running Perceptron - {method}...")
    results[f"Perceptron_{method}"] = evaluate_model("perceptron", batch_size)

for method, batch_size in BATCH_METHODS.items():
    print(f"Running SVM - {method}...")
    results[f"SVM_{method}"] = evaluate_model("svm", batch_size)

# =========================================================
# PLOT
# =========================================================
plt.figure(figsize=(12, 8))
for label, errors in results.items():
    plt.plot(range(1, EPOCHS + 1), errors, label=label)

plt.xlabel("Epoch")
plt.ylabel("Average Classification Error")
plt.title("Average Classification Error vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()