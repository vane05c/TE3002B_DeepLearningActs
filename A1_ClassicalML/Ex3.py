import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import Counter

data = np.loadtxt(r"C:\Users\anarg\OneDrive\Documentos\Tec\Implementation of intelligent robotics\misterious_data_2.txt")

y = data[:, 0]
X = data[:, 1:]

print("Shape:", X.shape)
print("Classes:", Counter(y))

min_class = min(Counter(y).values())
cv_value = min(5, min_class)

print("Using cv =", cv_value)

# 1. k-NN (k = 1 to 40)
k_values = range(1, 41)
knn_scores = []

for k in k_values:
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k))
    ])
    
    scores = cross_val_score(model, X, y, cv=cv_value)
    knn_scores.append(scores.mean())

best_k = k_values[np.argmax(knn_scores)]
best_knn_acc = max(knn_scores)

print("Best k:", best_k)
print("Best accuracy:", best_knn_acc)

plt.figure()
plt.plot(k_values, knn_scores, marker='o', color='hotpink')
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("k-NN Accuracy vs k")
plt.grid()
plt.show()

# 2. Linear SVM
C_values = np.arange(0.000001, 0.000101, 0.000002)
svm_linear_scores = []

for C in C_values:
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear', C=C))
    ])
    
    scores = cross_val_score(model, X, y, cv=cv_value)
    svm_linear_scores.append(scores.mean())

best_C = C_values[np.argmax(svm_linear_scores)]
best_svm_linear_acc = max(svm_linear_scores)

print("Best C:", best_C)
print("Best accuracy:", best_svm_linear_acc)

plt.figure()
plt.plot(C_values, svm_linear_scores, color='hotpink')
plt.xscale('log')  # 🔥 hace que se vea mucho mejor
plt.xlabel("C (log scale)")
plt.ylabel("Accuracy")
plt.title("Linear SVM - C vs Accuracy")
plt.grid()
plt.show()

# 3. RBF SVM
gamma_values = np.arange(0.000001, 0.000201, 0.000002)
svm_rbf_scores = []

for gamma in gamma_values:
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', gamma=gamma))
    ])
    
    scores = cross_val_score(model, X, y, cv=cv_value)
    svm_rbf_scores.append(scores.mean())

best_gamma = gamma_values[np.argmax(svm_rbf_scores)]
best_svm_rbf_acc = max(svm_rbf_scores)

print("Best gamma:", best_gamma)
print("Best accuracy:", best_svm_rbf_acc)

plt.figure()
plt.plot(gamma_values, svm_rbf_scores, color='hotpink')
plt.xlabel("gamma")
plt.ylabel("Accuracy")
plt.title("RBF SVM - Gamma vs Accuracy")
plt.grid()
plt.show()