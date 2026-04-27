import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE

data = np.loadtxt("misterious_data_4.txt")

y = data[:, 0]
X = data[:, 1:]

#Part 1
model = SVC(kernel='linear')

k_values = list(range(10, 101, 10))
filter_scores = []

print("Filter Method")
print("Features\tAccuracy")

for k in k_values:
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    
    scores = cross_val_score(model, X_new, y, cv=5)
    mean_acc = scores.mean()
    
    filter_scores.append(mean_acc)
    print(f"{k}\t\t{mean_acc:.4f}")

#Part 2
sfs_scores = []
k_values_sfs = list(range(1, 11))

print("\nSequential Feature Selection")
print("Features\tAccuracy")

for k in k_values_sfs:
    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=k,
        direction='forward',
        cv=5,
        n_jobs=-1
    )
    
    X_new = sfs.fit_transform(X, y)
    
    scores = cross_val_score(model, X_new, y, cv=5)
    mean_acc = scores.mean()
    
    sfs_scores.append(mean_acc)
    print(f"{k}\t\t{mean_acc:.4f}")

#Part 3    
rfe_scores = []
k_values_rfe = list(range(1, 11))

print("\nRFE Method")
print("Features\tAccuracy")

for k in k_values_rfe:
    rfe = RFE(model, n_features_to_select=k)
    X_new = rfe.fit_transform(X, y)
    
    scores = cross_val_score(model, X_new, y, cv=5)
    mean_acc = scores.mean()
    
    rfe_scores.append(mean_acc)
    print(f"{k}\t\t{mean_acc:.4f}")

#Plotting
plt.figure()
plt.plot(k_values, filter_scores, marker='o', color='hotpink')
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.title("Filter Method (SelectKBest)")
plt.grid()
plt.show()

plt.figure()
plt.plot(k_values_sfs, sfs_scores, marker='s', color='hotpink')
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.title("Sequential Feature Selection")
plt.grid()
plt.show()

plt.figure()
plt.plot(k_values_rfe, rfe_scores, marker='^', color='hotpink')
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.title("Recursive Feature Elimination (RFE)")
plt.grid()
plt.show()