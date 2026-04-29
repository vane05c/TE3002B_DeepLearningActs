import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# --- PASO 1: Carga de datos ---
# Cargamos el dataset Wine como un objeto de sklearn
wine_data = load_wine()
X, y = wine_data.data, wine_data.target

# Mostramos variables predictoras y dimensiones
print("--- Wine Dataset Information ---")
print(f"Prerdictor variables: {wine_data.feature_names}")
print(f"Total number of variables: {len(wine_data.feature_names)}")
print(f"Number of observations per class: {pd.Series(y).value_counts().sort_index().to_dict()}")
# Nota: Las clases 0, 1 y 2 representan diferentes tipos de vinos italianos.
# Los predictores representan análisis químicos como alcohol, ácido málico, etc.

# --- PASO 2: Definición de modelos y evaluación ---
# Creamos un diccionario para almacenar los modelos a evaluar
models = {
    "Linear SVM": SVC(kernel='linear'),
    "RBF SVM": SVC(kernel='rbf'),
    "k-NN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42) # Clasificador adicional
}

results = {}

print("\n--- Accuracy Results (5-fold CV) ---")
for name, model in models.items():
    # El escalador solo 've' los datos de entrenamiento de cada fold
    pipeline = Pipeline([
        ('scaler', StandardScaler()), # Normalización: media 0 y varianza 1
        ('clf', model)                 # Clasificador
    ])
    
    # Ejecutamos Validación Cruzada
    # Parámetros:
    # - pipeline: El flujo de procesamiento completo
    # - X, y: Datos y etiquetas
    # - cv=5: Número de pliegues (folds)
    # - scoring='accuracy': Métrica de evaluación
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    
    # Almacenamos el promedio de los 5 pliegues
    results[name] = cv_scores.mean()
    print(f"{name}: {results[name]:.4f}")

# --- PASO 3: Identificar el ganador ---
best_model = max(results, key=results.get)
print(f"\nThe best classifier is {best_model}, with an accuracy of {results[best_model]:.4f}")