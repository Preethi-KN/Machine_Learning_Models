# 🤖 K-Nearest Neighbors (KNN) Algorithm

## 📖 Table of Contents

- [Overview](#overview)
- [What is K in KNN?](#what-is-k-in-knn)
- [How to Choose the Value of K](#how-to-choose-the-value-of-k)
- [Distance Metrics](#distance-metrics)
- [How KNN Works (Step-by-Step)](#how-knn-works-step-by-step)
- [Python Implementation from Scratch](#python-implementation-from-scratch)
- [Using Scikit-Learn](#using-scikit-learn)
- [Applications](#applications)
- [Advantages](#advantages)
- [Disadvantages](#disadvantages)
- [Summary Cheat Sheet](#summary-cheat-sheet)
- [References](#references)

---

## Overview

**K-Nearest Neighbor (KNN)** is one of the simplest and most widely used supervised machine learning algorithms. It can be applied to both **classification** and **regression** tasks by identifying the `K` closest data points to a given input and basing the prediction on those neighbors.

### Key Characteristics

| Property | Description |
|---|---|
| **Type** | Supervised Learning |
| **Tasks** | Classification & Regression |
| **Learning Style** | Lazy Learner (Instance-Based) |
| **Parametric?** | No (Non-parametric) |
| **Training Phase** | None — stores the entire dataset |

### Why "Lazy Learner"?

KNN is called a **lazy learner** because it does **not** learn an explicit model during training. Instead, it memorizes the entire training dataset and defers all computation to the prediction (query) phase — when a new data point needs to be classified or predicted.

### Intuition

Imagine you're trying to classify a new data point (e.g., a fruit) among categories:

- KNN looks at the **K closest** known data points (neighbors).
- It counts which **category appears most** among those neighbors.
- The new point is assigned that **majority category**.

```
New Point ──► Find K Nearest Neighbors ──► Majority Vote ──► Predicted Class
```

---

## What is K in KNN?

`K` is a positive integer that tells the algorithm **how many neighboring data points** to consider when making a prediction.

### Effect of K Value

```
Small K  →  Low bias, High variance  →  Risk of Overfitting
Large K  →  High bias, Low variance  →  Risk of Underfitting
```

### Example

Suppose `K = 3` and the 3 nearest neighbors of a test point are:

```
Neighbor 1: Class A  ✅
Neighbor 2: Class A  ✅
Neighbor 3: Class B  ❌
```

**Prediction → Class A** (majority vote: 2 vs 1)

---

## How to Choose the Value of K

Choosing the right `K` is critical for model performance. The following methods help identify the optimal value:

### 1. Cross-Validation (Recommended)

Split your dataset into `k` folds. Train on some folds and test on the remaining ones. Repeat for each fold, then average the results. Choose the `K` that gives the best average accuracy.

```python
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5)
    print(f"K={k}, Accuracy={scores.mean():.4f}")
```

### 2. Elbow Method

Plot the error rate vs. different values of `K`. The "elbow point" — where the error rate stops decreasing sharply — is the optimal `K`.

```
Error Rate
   |
   |  \
   |   \
   |    \___________
   |
   +-------------------> K value
        ^ Elbow Point
```

### 3. Use Odd Values of K

For binary classification, always prefer **odd values of K** to avoid ties in the majority vote.

---

## Distance Metrics

KNN relies on a **distance metric** to measure how close two data points are. The most commonly used metrics are:

### 1. Euclidean Distance

The straight-line (shortest) distance between two points. Most commonly used.

```
d(x, y) = √[ Σ (xᵢ - yᵢ)² ]
```

**Best for:** Continuous, numeric data where straight-line distance is meaningful.

### 2. Manhattan Distance

The sum of absolute differences along each dimension — like navigating a city grid.

```
d(x, y) = Σ |xᵢ - yᵢ|
```

**Best for:** High-dimensional data or when features have different scales.

### 3. Minkowski Distance

A generalized formula that unifies both Euclidean and Manhattan distances via parameter `p`.

```
d(x, y) = ( Σ |xᵢ - yᵢ|^p )^(1/p)
```

| p Value | Equivalent Distance |
|---|---|
| `p = 1` | Manhattan Distance |
| `p = 2` | Euclidean Distance |
| `p > 2` | Other Minkowski variants |

---

## How KNN Works (Step-by-Step)

```
┌─────────────────────────────────────────────────────────────┐
│                    KNN Algorithm Flow                       │
└─────────────────────────────────────────────────────────────┘

Step 1: Choose K
   └─► Decide how many neighbors to consider

Step 2: Calculate Distance
   └─► Compute distance from the test point to ALL training points
       using Euclidean, Manhattan, or Minkowski distance

Step 3: Find Nearest Neighbors
   └─► Sort all distances
   └─► Select the K points with the smallest distances

Step 4: Make a Prediction
   ├─► Classification: Majority vote among K neighbors
   └─► Regression:     Average of K neighbors' values

Step 5: Output Result
   └─► Return the predicted class label or value
```

### Classification vs Regression

| Task | Prediction Method | Example |
|---|---|---|
| **Classification** | Majority voting among K neighbors | Spam / Not Spam |
| **Regression** | Mean (average) of K neighbors' values | House price prediction |

---

## Python Implementation from Scratch

### Step 1: Import Libraries

```python
import numpy as np
from collections import Counter
```

### Step 2: Define Euclidean Distance

```python
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
```

### Step 3: KNN Prediction Function

```python
def knn_predict(training_data, training_labels, test_point, k):
    distances = []

    # Calculate distance from test point to each training point
    for i in range(len(training_data)):
        dist = euclidean_distance(test_point, training_data[i])
        distances.append((dist, training_labels[i]))

    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[0])

    # Get labels of K nearest neighbors
    k_nearest_labels = [label for _, label in distances[:k]]

    # Return the most common label (majority vote)
    return Counter(k_nearest_labels).most_common(1)[0][0]
```

### Step 4: Run the Prediction

```python
# Training data
training_data   = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
training_labels = ['A',    'A',    'A',    'B',    'B'   ]

# Test point
test_point = [4, 5]
k = 3

# Predict
prediction = knn_predict(training_data, training_labels, test_point, k)
print(f"Predicted Class: {prediction}")
# Output: Predicted Class: A
```

**How it works:** The test point `[4, 5]` is closest to `[3,4]`, `[2,3]`, and `[6,7]`. Two of the three are labeled `'A'`, so the prediction is `'A'`.

---

## Using Scikit-Learn

For production use, the `scikit-learn` library provides a highly optimized KNN implementation.

### Classification

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load dataset
X, y = load_iris(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (important for KNN!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Build and train model
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

# Evaluate
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

### Regression

```python
from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)
predictions = knn_reg.predict(X_test)
```

> ⚠️ **Important:** Always scale your features before applying KNN (use `StandardScaler` or `MinMaxScaler`), as KNN is highly sensitive to the magnitude of features.

---

## Applications

KNN finds practical use in a wide range of real-world domains:

| Domain | Use Case |
|---|---|
| 🎬 **Recommendation Systems** | Suggests movies, products, or music by finding users with similar preferences |
| 📧 **Spam Detection** | Classifies emails by comparing them to known spam/non-spam examples |
| 🛒 **Customer Segmentation** | Groups customers by comparing their shopping behavior to others |
| 🎙️ **Speech Recognition** | Matches spoken words to known audio patterns |
| 🏥 **Medical Diagnosis** | Classifies patient symptoms against known disease patterns |
| 💳 **Fraud Detection** | Flags transactions that are unusually different from a user's normal behavior |

---

## Advantages

- ✅ **Simple and intuitive** — Easy to understand and implement with minimal configuration.
- ✅ **No training step** — Just stores the data; prediction happens on the fly.
- ✅ **Few hyperparameters** — Only requires `K` and a distance metric.
- ✅ **Naturally handles multi-class problems** — No special adjustments needed.
- ✅ **Versatile** — Works for both classification and regression tasks.
- ✅ **Non-parametric** — Makes no assumptions about the underlying data distribution.

---

## Disadvantages

- ❌ **Slow at prediction time** — Must compute distance to every training point for each query; O(n·d) per prediction.
- ❌ **High memory usage** — Stores the entire training dataset in memory.
- ❌ **Sensitive to feature scale** — Features with larger ranges can dominate distance calculations (always normalize!).
- ❌ **Curse of Dimensionality** — In high-dimensional spaces, distance metrics lose meaning, reducing effectiveness.
- ❌ **Sensitive to irrelevant features** — Noisy or redundant features can degrade performance.
- ❌ **Overfitting risk** — Very small K values (e.g., K=1) can overfit noisy data.

---

## Summary Cheat Sheet

```
┌──────────────────────────────────────────────────────────────────┐
│                      KNN Quick Reference                         │
├──────────────────┬───────────────────────────────────────────────┤
│ Algorithm Type   │ Supervised, Non-parametric, Lazy Learner       │
│ Tasks            │ Classification & Regression                    │
│ Key Parameter    │ K (number of neighbors)                        │
│ Distance Metrics │ Euclidean, Manhattan, Minkowski                │
│ Prediction       │ Majority Vote (classification) /               │
│                  │ Mean Value (regression)                        │
│ Best K Selection │ Cross-Validation or Elbow Method               │
│ Preprocessing    │ Feature Scaling is REQUIRED                    │
│ Complexity       │ O(n·d) per prediction (n=samples, d=features) │
│ Small K Effect   │ Overfitting (high variance)                    │
│ Large K Effect   │ Underfitting (high bias)                       │
└──────────────────┴───────────────────────────────────────────────┘
```

---

## References

- [GeeksforGeeks — K-Nearest Neighbor (KNN) Algorithm](https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/)
- [Scikit-Learn KNN Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
- [Implementation of KNN Classifier using Sklearn](https://www.geeksforgeeks.org/machine-learning/ml-implementation-of-knn-classifier-using-sklearn/)
- [Mathematical Explanation of KNN](https://www.geeksforgeeks.org/machine-learning/mathematical-explanation-of-k-nearest-neighbour/)
- [Weighted KNN](https://www.geeksforgeeks.org/machine-learning/weighted-k-nn/)
