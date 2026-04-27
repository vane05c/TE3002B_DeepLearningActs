import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample

# --- PASO 1: Carga de datos ---
try:
    data = pd.read_csv('A1_ClassicalML/misterious_data_3.txt', sep=r'\s+', header=None, engine='python')
    X = data.iloc[:, 1:].values  # Características (todas menos la primera columna)
    y = data.iloc[:, 0].values   # Etiquetas (columna 0)
    
    # Aseguramos que las etiquetas sean enteros para bincount
    y = y.astype(int)
    print("--- Carga Exitosa ---")
    print(f"Distribución original de clases: {np.bincount(y)[1:]} (Clase 1 y Clase 2)")
except Exception as e:
    print(f"Error al cargar el archivo: {e}")

# --- PASO 2: División de datos (Evitar Data Leakage) ---
# Stratify asegura que la proporción de clases se mantenga en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- PASO 3: Escalado de características ---
# El escalador solo aprende de X_train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate_svm(X_tr, y_tr, X_te, y_te, title, balanced=None):
    """
    Entrena un SVM Lineal y reporta métricas detalladas.
    Retorna el modelo entrenado.
    """
    model = SVC(kernel='linear', class_weight=balanced, random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    
    print(f"\n--- {title} ---")
    acc = accuracy_score(y_te, preds)
    print(f"Accuracy General: {acc:.4f}")
    # classification_report nos da Precision y Recall por cada clase (1 y 2)
    print(classification_report(y_te, preds))
    return acc

# 1. Evaluación Base
evaluate_svm(X_train_scaled, y_train, X_test_scaled, y_test, "1. SVM Base (Sin balancear)")

# --- PASO 4: Balanceo (Aplicado SOLO al set de entrenamiento) ---
train_df = pd.DataFrame(X_train_scaled)
train_df['target'] = y_train

c1 = train_df[train_df['target'] == 1]
c2 = train_df[train_df['target'] == 2]

# Identificar minoritaria y mayoritaria dinámicamente
minority = c1 if len(c1) < len(c2) else c2
majority = c2 if len(c1) < len(c2) else c1

# 2. Undersampling
majority_down = resample(majority, replace=False, n_samples=len(minority), random_state=42)
down_train = pd.concat([majority_down, minority])
evaluate_svm(down_train.drop('target', axis=1), down_train['target'], 
             X_test_scaled, y_test, "2. Undersampling")

# 3. Oversampling
minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
up_train = pd.concat([majority, minority_up])
evaluate_svm(up_train.drop('target', axis=1), up_train['target'], 
             X_test_scaled, y_test, "3. Oversampling")

# 4. Class Weight Balanced
evaluate_svm(X_train_scaled, y_train, X_test_scaled, y_test, 
             "4. SVM con class_weight='balanced'", balanced='balanced')