# 🧠 Support Vector Machine (SVM) Algorithm


## 📖 Table of Contents

- [Overview](#overview)
- [Key Concepts & Terminology](#key-concepts--terminology)
- [How SVM Works](#how-svm-works)
- [Handling Non-Linearly Separable Data](#handling-non-linearly-separable-data)
- [Mathematical Foundation](#mathematical-foundation)
  - [Hyperplane Equation](#hyperplane-equation)
  - [Distance from a Point to Hyperplane](#distance-from-a-point-to-hyperplane)
  - [Hard Margin Optimization](#hard-margin-optimization)
  - [Soft Margin Optimization](#soft-margin-optimization)
  - [Dual Problem & Kernel Trick](#dual-problem--kernel-trick)
  - [Decision Boundary](#decision-boundary)
- [Kernel Functions](#kernel-functions)
- [Types of SVM](#types-of-svm)
- [Python Implementation from Scratch](#python-implementation-from-scratch)
- [Implementation with Scikit-Learn](#implementation-with-scikit-learn)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [SVM for Regression (SVR)](#svm-for-regression-svr)
- [Applications](#applications)
- [Advantages](#advantages)
- [Disadvantages](#disadvantages)
- [SVM vs KNN — Quick Comparison](#svm-vs-knn--quick-comparison)
- [Summary Cheat Sheet](#summary-cheat-sheet)
- [References](#references)

---

## Overview

**Support Vector Machine (SVM)** is a powerful supervised machine learning algorithm used primarily for **classification** tasks, though it also supports **regression** (known as SVR — Support Vector Regression).

The core idea behind SVM is to find the **optimal hyperplane** that best separates data points belonging to different classes. The "best" hyperplane is the one that **maximizes the margin** — the gap between the two classes. A larger margin means better generalization to unseen data.

### Key Characteristics

| Property | Description |
|---|---|
| **Type** | Supervised Learning |
| **Tasks** | Classification & Regression |
| **Learning Style** | Eager Learner (builds a model during training) |
| **Parametric?** | No (Non-parametric) |
| **Decision Boundary** | Hyperplane (linear or non-linear via kernels) |
| **Works Best On** | High-dimensional, small-to-medium datasets |

### Core Intuition

```
Given labelled training data, SVM finds the hyperplane that:

  1. Correctly separates classes        → Accuracy
  2. Maximizes the margin between them  → Generalization

The data points closest to the hyperplane are called SUPPORT VECTORS.
They alone define the position and orientation of the hyperplane.
```

---

## Key Concepts & Terminology

### 📐 Hyperplane

A **decision boundary** that separates data points into different classes. In an n-dimensional feature space, the hyperplane is an (n–1)-dimensional subspace.

```
  2D  → the hyperplane is a LINE
  3D  → the hyperplane is a PLANE
  nD  → the hyperplane is an (n-1)-dimensional surface
```

Mathematically represented as:
```
w·x + b = 0
```
Where `w` is the normal vector (perpendicular to the hyperplane) and `b` is the bias/offset.

---

### 📌 Support Vectors

The **closest data points** to the hyperplane from each class. These are the most critical points in the dataset — if they were removed, the hyperplane would shift.

```
      Class A   ║   Class B
        o        ║        x
      o  [o] ← ─╫─ → [x]  x
        o        ║        x
                 ║
           Hyperplane
     [o] and [x] are the Support Vectors
```

---

### ↔️ Margin

The **perpendicular distance** between the hyperplane and the nearest support vectors on each side. SVM maximizes this margin.

```
   |← margin →|← margin →|
   |           |           |
   o   o    [o]|[x]    x   x
               |
          Hyperplane (w·x + b = 0)
```

---

### ⚖️ Hard Margin vs Soft Margin

| Type | Description | When to Use |
|---|---|---|
| **Hard Margin** | Perfectly separates all data; allows zero misclassification | Only when data is perfectly linearly separable |
| **Soft Margin** | Allows some violations/misclassifications via slack variables | Real-world noisy or overlapping data |

---

### 🎛️ Regularization Parameter C

`C` controls the trade-off between **margin size** and **classification accuracy** on training data.

```
High C  → Narrow margin, fewer training errors  → Risk of Overfitting
Low C   → Wide margin, more training errors     → Risk of Underfitting
```

---

### 📉 Hinge Loss

The loss function used by SVM, which penalizes misclassifications and margin violations:

```
Loss = max(0, 1 − yᵢ(w·xᵢ + b))
```

| Condition | Loss |
|---|---|
| Point correctly classified & outside margin | 0 (no penalty) |
| Point correctly classified but inside margin | > 0 (small penalty) |
| Point misclassified | > 0, grows proportionally to distance of violation |

---

## How SVM Works

SVM follows a structured process to find the optimal separating hyperplane:

```
┌────────────────────────────────────────────────────────────────┐
│                    SVM Algorithm Flow                          │
└────────────────────────────────────────────────────────────────┘

Step 1: Input Training Data
   └─► Labelled dataset with features X and class labels y ∈ {+1, -1}

Step 2: Choose Kernel Function
   └─► Linear, Polynomial, RBF, or Sigmoid

Step 3: Map Data (if non-linear)
   └─► Transform data into higher-dimensional feature space via kernel

Step 4: Find Optimal Hyperplane
   └─► Solve the optimization problem to maximize margin
   └─► Subject to: all points correctly classified (hard) or
       allowing slack variables (soft margin)

Step 5: Identify Support Vectors
   └─► Data points closest to the hyperplane (αᵢ > 0)

Step 6: Construct Decision Boundary
   └─► Use support vectors and bias term b to define hyperplane

Step 7: Predict New Points
   └─► sign(w·x + b)
       +1 → Class A
       -1 → Class B
```

### Visual Representation

```
         Class A (−1)         Class B (+1)
                  Margin
     o              |    x
       o    [o] ←───+───→ [x]    x
     o              |    x
                    |
            Hyperplane: w·x + b = 0

  Positive side: w·x + b ≥ +1  →  Class B (+1)
  Negative side: w·x + b ≤ −1  →  Class A (−1)
  On boundary:   w·x + b  = 0
```

---

## Handling Non-Linearly Separable Data

In real-world scenarios, data is rarely linearly separable. SVM handles this elegantly using the **kernel trick**.

### The Kernel Trick

Instead of explicitly transforming data into a higher-dimensional space (which is computationally expensive), the kernel trick computes the **dot product** in the higher-dimensional space directly through a kernel function — without ever computing the transformation.

```
Original Space (1D)           Higher-Dimensional Space (2D)
        ──────────────         ─────────────────────────────
  o  o  o   x  x  x    →             o  o  o
  Not separable by line              ─────────────  ← Hyperplane
                                        x  x  x
                                     Linearly separable!
```

---

## Mathematical Foundation

### Hyperplane Equation

```
w^T · x + b = 0
```

| Symbol | Meaning |
|---|---|
| `w` | Normal vector (perpendicular to the hyperplane) |
| `x` | Input feature vector |
| `b` | Bias/offset term (distance from origin) |

---

### Distance from a Point to Hyperplane

```
        w^T · xᵢ + b
dᵢ  =  ─────────────
             ‖w‖
```

Where `‖w‖` is the Euclidean norm of the weight vector.

---

### Hard Margin Optimization

For perfectly linearly separable data, SVM solves:

**Objective (minimize):**
```
    1
  ─────  ‖w‖²
    2
```

**Subject to constraint:**
```
yᵢ (w^T · xᵢ + b) ≥ 1    for all i = 1, 2, ..., m
```

This ensures every data point is **correctly classified** and **lies outside** the margin.

---

### Soft Margin Optimization

For noisy or non-separable data, slack variables `ζᵢ ≥ 0` are introduced:

**Objective (minimize):**
```
    1              m
  ─────  ‖w‖²  + C Σ ζᵢ
    2             i=1
```

**Subject to constraints:**
```
yᵢ (w^T · xᵢ + b) ≥ 1 − ζᵢ    and    ζᵢ ≥ 0    for all i
```

| Symbol | Meaning |
|---|---|
| `C` | Regularization parameter (penalty for misclassification) |
| `ζᵢ` | Slack variable (degree of margin violation for point i) |

---

### Dual Problem & Kernel Trick

The dual formulation maximizes Lagrange multipliers `αᵢ`, enabling kernel functions:

**Dual Objective (maximize):**
```
    m   m
 1  Σ   Σ  αᵢ αⱼ tᵢ tⱼ K(xᵢ, xⱼ)  −  Σ αᵢ
 2  i=1 j=1                              i=1
```

| Symbol | Meaning |
|---|---|
| `αᵢ` | Lagrange multiplier for i-th training sample |
| `tᵢ` | Class label (+1 or −1) for i-th sample |
| `K(xᵢ, xⱼ)` | Kernel function (similarity between two points) |

**Support Vectors** are the training points where `αᵢ > 0`.

---

### Decision Boundary

After solving the dual problem, predictions are made using:

```
ŷ = sign( Σ αᵢ tᵢ K(xᵢ, x) + b )
```

**Prediction Rule:**
```
w^T · x + b ≥ 0   →  Predict Class +1
w^T · x + b < 0   →  Predict Class −1
```

---

## Kernel Functions

Kernels transform data into higher-dimensional spaces to make non-linear data linearly separable. Choosing the right kernel is crucial for performance.

### 1. 📏 Linear Kernel

```
K(xᵢ, xⱼ) = xᵢ^T · xⱼ
```

**Use when:** Data is already linearly separable. Fastest and simplest option.

**Best for:** Text classification, document categorization.

---

### 2. 🔢 Polynomial Kernel

```
K(xᵢ, xⱼ) = (γ · xᵢ^T · xⱼ + r)^d
```

| Parameter | Meaning |
|---|---|
| `d` | Degree of the polynomial |
| `γ` | Scale parameter |
| `r` | Coefficient (bias) |

**Use when:** Features have polynomial relationships.

**Best for:** Image processing, NLP.

---

### 3. 🔵 Radial Basis Function (RBF) / Gaussian Kernel

```
K(xᵢ, xⱼ) = exp(−γ ‖xᵢ − xⱼ‖²)
```

| Parameter | Meaning |
|---|---|
| `γ` | Controls the "spread" of the kernel |

**Small γ** → Wide influence radius → Smoother boundary → Underfitting risk  
**Large γ** → Narrow influence radius → Tighter boundary → Overfitting risk

**Use when:** No prior knowledge of data distribution. Most popular default choice.

**Best for:** General-purpose non-linear classification.

---

### 4. 〰️ Sigmoid Kernel

```
K(xᵢ, xⱼ) = tanh(γ · xᵢ^T · xⱼ + r)
```

**Use when:** Mimicking neural network-like behavior.

**Best for:** Binary classification tasks similar to logistic regression.

---

### Kernel Comparison Summary

| Kernel | Formula | Complexity | Best Use Case |
|---|---|---|---|
| Linear | `xᵢ·xⱼ` | Low | Linearly separable data |
| Polynomial | `(γxᵢ·xⱼ + r)^d` | Medium | Image processing |
| RBF/Gaussian | `exp(−γ‖xᵢ−xⱼ‖²)` | High | General non-linear tasks |
| Sigmoid | `tanh(γxᵢ·xⱼ + r)` | Medium | Neural network-like tasks |

---

## Types of SVM

### 🔵 Linear SVM

Separates data using a **straight-line hyperplane** (in 2D) or a flat hyperplane (in higher dimensions). Works only when data is linearly separable.

```
   o  o  o  |  x  x  x
             |
        Linear Boundary
```

**Pros:** Fast, interpretable, works great on high-dimensional sparse data.  
**Cons:** Fails when data has non-linear relationships.

---

### 🔴 Non-Linear SVM

Uses **kernel functions** to implicitly map data into higher dimensions where a linear separator can be found.

```
Original Space (not separable)     Feature Space (separable)
    o  x  o  x  o                      o  o  o
                         →              ────────
                                        x  x  x
```

**Pros:** Handles complex, real-world patterns.  
**Cons:** Slower to train; harder to tune and interpret.

---

## Python Implementation from Scratch

```python
import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, C=1.0, n_epochs=1000):
        self.lr = learning_rate
        self.C = C
        self.n_epochs = n_epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Convert labels to {-1, +1}
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_epochs):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # Point is correctly classified outside margin
                    self.w -= self.lr * (2 * self.w)
                else:
                    # Margin violation or misclassification
                    self.w -= self.lr * (2 * self.w - self.C * y_[idx] * x_i)
                    self.b -= self.lr * (self.C * y_[idx])

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
```

### Usage

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate sample data
X, y = make_classification(n_samples=200, n_features=2, random_state=42)
y = np.where(y == 0, -1, 1)   # Convert to {-1, +1}

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Train custom SVM
svm = SVM(learning_rate=0.001, C=1.0, n_epochs=1000)
svm.fit(X_train, y_train)

# Evaluate
predictions = svm.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.4f}")
```

---

## Implementation with Scikit-Learn

### Binary Classification — Breast Cancer Dataset

```python
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load data
X, y = load_breast_cancer(return_X_y=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (CRITICAL for SVM!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Build SVM classifier
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train, y_train)

# Evaluate
y_pred = svm.predict(X_test)
print(f"Accuracy: {svm.score(X_test, y_test):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Malignant', 'Benign']))
```

### Multiclass Classification

SVM natively supports binary classification but handles multiclass through:

```python
# One-vs-One (OvO) — default in SVC
svm_ovo = SVC(kernel='rbf', decision_function_shape='ovo')

# One-vs-Rest (OvR)
svm_ovr = SVC(kernel='rbf', decision_function_shape='ovr')
```

### Visualizing the Decision Boundary

```python
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

# Train on 2 features for visualization
X_2d = X[:, :2]
svm_2d = SVC(kernel='linear', C=1)
svm_2d.fit(X_2d, y)

DecisionBoundaryDisplay.from_estimator(
    svm_2d,
    X_2d,
    response_method="predict",
    alpha=0.8,
    cmap="Pastel1"
)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, s=20, edgecolors='k')
plt.title("SVM Decision Boundary")
plt.show()
```

---

## Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Description | Typical Range |
|---|---|---|
| `C` | Regularization strength (penalty for errors) | `0.01` to `1000` |
| `kernel` | Type of kernel function | `linear`, `rbf`, `poly`, `sigmoid` |
| `gamma` | Kernel coefficient (RBF/poly/sigmoid) | `scale`, `auto`, `0.001`–`10` |
| `degree` | Degree for polynomial kernel | `2`, `3`, `4` |

### Grid Search with Cross-Validation

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    'C':      [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma':  ['scale', 'auto', 0.001, 0.01, 0.1]
}

grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best Parameters : {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
```

### Effect of C and Gamma (RBF Kernel)

```
                C →  Small          Large
         ┌────────────────────────────────────┐
  γ Small │ Wide margin    │ Narrower margin  │
         │ Underfitting   │ Good generalize  │
         ├────────────────┼──────────────────┤
  γ Large │ Complex, tight │ Very tight fit  │
         │ Overfitting    │ Severe overfit  │
         └────────────────────────────────────┘
```

---

## SVM for Regression (SVR)

SVM can also solve **regression** problems using **Support Vector Regression (SVR)**. Instead of maximizing margin, it fits data within an **epsilon (ε) tube** and minimizes violations outside it.

```python
from sklearn.svm import SVR
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X, y)
predictions = svr.predict(X)
```

### SVR Key Parameters

| Parameter | Description |
|---|---|
| `epsilon (ε)` | Width of the "tube" — points inside have no penalty |
| `C` | Penalty for points outside the tube |
| `kernel` | Same kernels as SVM classification |

---

## Applications

SVM is widely used across many real-world domains:

| Domain | Application |
|---|---|
| 📧 **Email Filtering** | Classifying emails as spam or not spam |
| 🖼️ **Image Classification** | Recognizing faces, handwritten digits (MNIST), objects |
| 🧬 **Bioinformatics** | Cancer classification from gene expression data |
| 💰 **Finance** | Credit scoring, fraud detection, stock movement prediction |
| 🏥 **Medical Diagnosis** | Diagnosing diseases from patient data (e.g., breast cancer) |
| 📝 **Text Classification** | Sentiment analysis, document categorization |
| 🎙️ **Speech Recognition** | Speaker identification from audio features |
| 🌍 **Remote Sensing** | Land cover classification from satellite images |

---

## Advantages

- ✅ **Effective in high-dimensional spaces** — performs well even when the number of features exceeds the number of samples.
- ✅ **Memory efficient** — uses only the support vectors (a subset of training data) to define the decision boundary.
- ✅ **Versatile** — different kernel functions allow it to adapt to linear and non-linear data.
- ✅ **Robust to overfitting** — especially in high-dimensional feature spaces when `C` is tuned properly.
- ✅ **Handles outliers** — the soft-margin formulation allows some misclassification, reducing the impact of noisy data.
- ✅ **Strong theoretical foundation** — rooted in convex optimization and statistical learning theory.

---

## Disadvantages

- ❌ **Slow on large datasets** — training time scales quadratically to cubically with sample size; impractical for millions of data points.
- ❌ **Difficult hyperparameter tuning** — choosing the right kernel, C, and gamma requires experimentation.
- ❌ **Sensitive to feature scaling** — features with large ranges dominate the distance; always normalize input data.
- ❌ **Limited interpretability** — the hyperplane in high-dimensional space is hard to visualize or explain.
- ❌ **Noise sensitivity** — overlapping classes or noisy labels can significantly degrade performance.
- ❌ **No probability estimates by default** — `probability=True` in sklearn uses Platt scaling, which adds overhead.

---

## SVM vs KNN — Quick Comparison

| Feature | SVM | KNN |
|---|---|---|
| **Type** | Eager Learner (trains a model) | Lazy Learner (no training phase) |
| **Decision Boundary** | Hyperplane (global, parametric) | Local, instance-based |
| **Training Speed** | Moderate to slow | Instant (no training) |
| **Prediction Speed** | Fast | Slow (computes all distances) |
| **Memory Usage** | Low (only support vectors) | High (stores entire dataset) |
| **Works Best On** | High-dimensional data, clear margins | Small, low-dimensional datasets |
| **Kernel Trick** | Yes (non-linear via kernels) | No (distance-based only) |
| **Interpretability** | Low | High (easy to visualize) |
| **Noisy Data** | Handles with soft-margin C | Very sensitive to noise |

---

## Summary Cheat Sheet

```
┌────────────────────────────────────────────────────────────────────┐
│                     SVM Quick Reference                            │
├──────────────────────┬─────────────────────────────────────────────┤
│ Algorithm Type       │ Supervised, Non-parametric, Eager Learner   │
│ Tasks                │ Classification & Regression (SVR)           │
│ Core Goal            │ Maximize margin between classes             │
│ Key Concepts         │ Hyperplane, Support Vectors, Margin,        │
│                      │ Kernel, Hard/Soft Margin, Hinge Loss        │
│ Kernels              │ Linear, Polynomial, RBF, Sigmoid            │
│ Key Parameters       │ C (regularization), γ (gamma), degree       │
│ High C Effect        │ Narrow margin → Overfitting risk            │
│ Low C Effect         │ Wide margin  → Underfitting risk            │
│ Feature Scaling      │ REQUIRED (use StandardScaler)              │
│ Handles Non-Linear   │ Yes — via kernel trick                      │
│ Support Vectors      │ Points where αᵢ > 0                        │
│ Loss Function        │ Hinge Loss                                  │
│ Training Complexity  │ O(n²) to O(n³) — slow on large datasets    │
│ Prediction Speed     │ Fast (linear in support vectors)           │
│ Best For             │ High-dim data, clear margins, small-medium  │
│                      │ datasets, image & text classification       │
└──────────────────────┴─────────────────────────────────────────────┘
```

---

## References

- [GeeksforGeeks — Support Vector Machine (SVM) Algorithm](https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm/)
- [GeeksforGeeks — Non-Linear SVM](https://www.geeksforgeeks.org/machine-learning/ml-non-linear-svm/)
- [GeeksforGeeks — Kernel Functions in SVM](https://www.geeksforgeeks.org/machine-learning/major-kernel-functions-in-support-vector-machine-svm/)
- [GeeksforGeeks — Hinge Loss & SVM](https://www.geeksforgeeks.org/machine-learning/hinge-loss-relationship-with-support-vector-machines/)
- [Scikit-Learn SVM Documentation](https://scikit-learn.org/stable/modules/svm.html)
- [Cortes, C. & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273–297.](https://link.springer.com/article/10.1007/BF00994018)

---