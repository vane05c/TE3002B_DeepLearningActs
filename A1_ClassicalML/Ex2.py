import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split

# Load data
data = np.loadtxt("misterious_data_1.txt")

y = data[:, 0]
X = data[:, 1:]

model = SVC(kernel='linear')

# ======================
# Part 1 (random subsets)
percentages = np.arange(0.10, 1.00, 0.05)
accuracies = []

print("Part 1: Random Sampling")
print("Percentage\tAccuracy")

for p in percentages:
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=p, stratify=y
    )
    
    scores = cross_val_score(model, X_sub, y_sub, cv=5)
    mean_acc = scores.mean()
    
    accuracies.append(mean_acc)
    
    print(f"{int(p*100)}%\t\t{mean_acc:.4f}")

# Plot Part 1
plt.figure()
plt.plot(percentages * 100, accuracies, marker='o', label="Random subsets", color='hotpink')
plt.xlabel("Percentage of Data Used (%)")
plt.ylabel("Accuracy")
plt.title("Learning Curve (Linear SVM)")
plt.grid()


# ======================
# Part 2 (fixed 10%)
X_sub, _, y_sub, _ = train_test_split(
    X, y, train_size=0.10, stratify=y, random_state=42
)

scores = cross_val_score(model, X_sub, y_sub, cv=5)

print("\nPart 2: Fixed 10%")
print("Fold accuracies:", scores)
print("Mean accuracy:", scores.mean())
print("Std deviation:", scores.std())


# ======================
# Part 3 (incremental data)

# Shuffle ONCE
rng = np.random.RandomState(42)
indices = rng.permutation(len(X))

X_shuffled = X[indices]
y_shuffled = y[indices]

incremental_accuracies = []

print("\nPart 3: Incremental Sampling")
print("Percentage\tAccuracy")

for p in percentages:
    n_samples = int(p * len(X_shuffled))
    
    X_sub = X_shuffled[:n_samples]
    y_sub = y_shuffled[:n_samples]
    
    scores = cross_val_score(model, X_sub, y_sub, cv=5)
    mean_acc = scores.mean()
    
    incremental_accuracies.append(mean_acc)
    
    print(f"{int(p*100)}%\t\t{mean_acc:.4f}")

# Plot Part 3
plt.figure()
plt.plot(percentages * 100, incremental_accuracies, marker='s', color='hotpink')
plt.xlabel("Percentage of Data Used (%)")
plt.ylabel("Accuracy")
plt.title("Incremental Learning Curve")
plt.grid()
plt.show()