# 🌳 Decision Tree (DT) Algorithm
---

## 📖 Table of Contents

- [Overview](#overview)
- [Key Concepts & Terminology](#key-concepts--terminology)
- [Tree Structure Anatomy](#tree-structure-anatomy)
- [How a Decision Tree Works](#how-a-decision-tree-works)
- [Real-Life Example — Weather Activity Predictor](#real-life-example--weather-activity-predictor)
- [Attribute Selection Measures](#attribute-selection-measures)
  - [Entropy](#1-entropy)
  - [Information Gain](#2-information-gain)
  - [Gini Index](#3-gini-index)
  - [Information Gain vs Gini Index](#information-gain-vs-gini-index)
- [Building a Decision Tree — Worked Example](#building-a-decision-tree--worked-example)
- [Decision Tree Algorithms](#decision-tree-algorithms)
- [Overfitting & Tree Pruning](#overfitting--tree-pruning)
- [Implementation with Scikit-Learn](#implementation-with-scikit-learn)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Decision Tree for Regression](#decision-tree-for-regression)
- [Ensemble Methods Built on Decision Trees](#ensemble-methods-built-on-decision-trees)
- [Applications](#applications)
- [Advantages](#advantages)
- [Disadvantages](#disadvantages)
- [DT vs SVM vs KNN — Quick Comparison](#dt-vs-svm-vs-knn--quick-comparison)
- [Summary Cheat Sheet](#summary-cheat-sheet)
- [References](#references)

---

## Overview

A **Decision Tree** is a supervised machine learning algorithm that models decisions and their possible consequences as a **tree-like flowchart structure**. It is used for both **classification** and **regression** tasks and is renowned for being highly **interpretable** — every decision path can be read like a set of plain English rules.

### Key Characteristics

| Property | Description |
|---|---|
| **Type** | Supervised Learning |
| **Tasks** | Classification & Regression |
| **Learning Style** | Eager Learner (builds a model during training) |
| **Parametric?** | No (Non-parametric) |
| **Interpretability** | Very High — fully explainable rule-based model |
| **Preprocessing Needed** | Minimal — no feature scaling required |

### Core Intuition

```
  A Decision Tree answers a sequence of questions about features,
  narrowing down to a final prediction — just like a game of 20 Questions.

  Is Income > $50K?
      │
      ├── YES → Is Age > 30?
      │             │
      │             ├── YES → Has Previous Purchases?
      │             │             ├── YES → ✅ Purchase
      │             │             └── NO  → ❌ No Purchase
      │             └── NO  → ❌ No Purchase
      └── NO  → ❌ No Purchase
```

---

## Key Concepts & Terminology

### 🌱 Root Node
The **topmost node** of the tree representing the entire dataset. The first (best) splitting feature is placed here. It has no parent node.

### 🔀 Internal Node (Decision Node)
A node that **tests a condition** on a feature and branches based on the result. Each internal node has one or more child nodes.

### 🍃 Leaf Node (Terminal Node)
A **final output node** that holds the class label (classification) or a predicted value (regression). No further splitting occurs here.

### ➡️ Branch (Edge)
A **connection between nodes** representing a specific outcome of a test condition (e.g., Yes / No, Low / Medium / High).

### 📏 Depth
The **number of levels** in the tree from root to the deepest leaf. A deeper tree captures more complexity but risks overfitting.

### 🧹 Pruning
The process of **removing branches** that have little predictive power, reducing tree complexity and preventing overfitting.

### 🎯 Purity
A node is **pure** if all samples in it belong to a single class. The goal of splitting is to increase node purity.

### 🔁 Recursive Partitioning
The strategy of **repeatedly splitting** subsets of data based on the best feature at each step, until stopping criteria are met.

---

## Tree Structure Anatomy

```
                    ┌─────────────────────┐
                    │     ROOT NODE        │  ← Best splitting feature
                    │   "Outlook?"         │
                    └──────────┬──────────┘
               ┌───────────────┼───────────────┐
            Sunny           Cloudy           Rainy
               │               │               │
    ┌──────────┴──────┐  ┌─────┴──────┐  ┌────┴────────────┐
    │ INTERNAL NODE   │  │  LEAF NODE │  │  INTERNAL NODE  │
    │  "Humidity?"    │  │  "Hiking"  │  │  "Wind?"        │
    └────┬────────────┘  └────────────┘  └─────┬───────────┘
      High │  Normal                       Strong │  Weak
           │    │                                │    │
    ┌──────┴┐ ┌─┴──────┐               ┌────────┴┐ ┌─┴──────┐
    │ LEAF  │ │ LEAF   │               │  LEAF   │ │ LEAF   │
    │"Swim" │ │"Hike"  │               │"Inside" │ │"Hike"  │
    └───────┘ └────────┘               └─────────┘ └────────┘
```

---

## How a Decision Tree Works

Decision Trees use **recursive binary splitting** (or multi-way splitting) to partition the dataset into progressively purer subsets.

```
┌────────────────────────────────────────────────────────────────────┐
│                  Decision Tree Training Flow                       │
└────────────────────────────────────────────────────────────────────┘

Step 1: Start with the full dataset at the Root Node

Step 2: Select the Best Feature to Split On
   └─► Evaluate every feature using an Attribute Selection Measure:
       - Information Gain (ID3, C4.5)
       - Gini Index       (CART)

Step 3: Split the Dataset
   └─► Divide data into subsets based on the chosen feature's values

Step 4: Recursive Splitting
   └─► Repeat Steps 2–3 on each subset until a stopping criterion is met:
       • All samples in a node belong to one class (pure leaf)
       • No features remaining to split on
       • Tree has reached maximum depth
       • Node has fewer than min_samples_split samples

Step 5: Assign Leaf Labels
   ├─► Classification: Majority class label of the samples in that node
   └─► Regression:     Mean value of the samples in that node

Step 6: Predict New Data Points
   └─► Traverse the tree from root to leaf following feature conditions
       → Return the leaf node's label/value
```

---

## Real-Life Example — Weather Activity Predictor

This example predicts the best activity (Hiking, Swimming, Stay Inside) based on weather conditions.

### Dataset

| Outlook | Humidity | Wind   | Activity    |
|---------|----------|--------|-------------|
| Sunny   | High     | Weak   | Swimming    |
| Sunny   | Normal   | Weak   | Hiking      |
| Sunny   | Normal   | Strong | Hiking      |
| Cloudy  | High     | Weak   | Hiking      |
| Cloudy  | Normal   | Strong | Hiking      |
| Rainy   | Normal   | Weak   | Hiking      |
| Rainy   | High     | Strong | Stay Inside |
| Rainy   | Normal   | Strong | Stay Inside |

### Decision Tree Built from This Data

```
                      [Outlook?]
                     /    |    \
                  /       |      \
              Sunny    Cloudy    Rainy
                /         |         \
        [Humidity?]   "Hiking" ✅   [Wind?]
          /     \                   /     \
        High   Normal           Strong    Weak
          |       |               |         |
      "Swim" ✅ "Hike" ✅    "Inside" ✅  "Hike" ✅
```

### Prediction Example

**Query:** Outlook = Sunny, Humidity = High  
**Path:** Root → Sunny → High Humidity → **"Swimming"** ✅

---

## Attribute Selection Measures

The critical question at each node is: **"Which feature should we split on?"**

The answer is determined by an **Attribute Selection Measure (ASM)** that quantifies the quality of each possible split.

---

### 1. Entropy

**Entropy** measures the **impurity** or uncertainty in a set of examples. The higher the entropy, the more mixed (impure) the set.

**Formula:**
```
H(S) = − Σ  pᵢ · log₂(pᵢ)
```

| Symbol | Meaning |
|---|---|
| `S` | The current dataset/subset |
| `pᵢ` | Proportion of class i in the dataset |

**Entropy Range:**
```
H = 0    →  Pure node (all one class)     — Best
H = 1    →  Maximum impurity (50/50 mix)  — Worst (binary case)
```

**Concrete Example:**

```
Dataset X = {a, a, a, b, b, b, b, b}
  Total = 8 | Class a = 3 | Class b = 5

H(X) = −[ (3/8) · log₂(3/8) + (5/8) · log₂(5/8) ]
     = −[ 0.375 × (−1.415) + 0.625 × (−0.678) ]
     = −[ −0.530 − 0.424 ]
     = 0.954   ← High entropy, mixed set
```

**Intuition Table:**

| Scenario | Entropy |
|---|---|
| All samples are Class A | 0.0 (perfectly pure) |
| 75% Class A, 25% Class B | 0.811 |
| 50% Class A, 50% Class B | 1.0 (maximum impurity) |

---

### 2. Information Gain

**Information Gain (IG)** measures the **reduction in entropy** achieved by splitting on a feature. The feature with the **highest IG** is chosen as the splitting criterion.

**Formula:**
```
Gain(S, A) = Entropy(S) − Σ  (|Sᵥ| / |S|) · Entropy(Sᵥ)
                          v ∈ Values(A)
```

| Symbol | Meaning |
|---|---|
| `S` | Current dataset |
| `A` | Candidate feature to split on |
| `Sᵥ` | Subset of S where feature A = value v |
| `\|Sᵥ\| / \|S\|` | Weight (fraction of samples in subset) |

**Interpretation:**

```
High Information Gain  →  Feature creates purer subsets  →  Choose this split!
Low  Information Gain  →  Feature barely reduces entropy  →  Avoid this split
```

**Building Rules (ID3 Algorithm):**

```
1. Start with ALL training instances at the root node
2. Compute Information Gain for each feature
3. Choose the feature with the HIGHEST Information Gain to split on
4. Recursively build subtrees on each subset
5. STOP when:
   • All instances in a node share the same class label → assign that label
   • No features remain                               → majority vote
   • No instances remain                              → parent's majority vote
```

---

### 3. Gini Index

**Gini Index** measures the **probability of misclassification** if a sample is randomly classified using the distribution of class labels in the node.

**Formula:**
```
Gini(S) = 1 − Σ  pᵢ²
               i=1..n
```

**Gini Range:**
```
Gini = 0.0   →  Perfectly pure (all one class)    — Best
Gini = 0.5   →  Maximum impurity (binary case)    — Worst
```

**Example Calculation:**
```
Node with 4 samples: 3 "Yes", 1 "No"

p(Yes) = 3/4 = 0.75
p(No)  = 1/4 = 0.25

Gini = 1 − (0.75² + 0.25²)
     = 1 − (0.5625 + 0.0625)
     = 1 − 0.625
     = 0.375
```

**Additional Properties of Gini Index:**

- Calculated by summing squared probabilities of each class and subtracting from 1.
- Lower Gini Index → more homogeneous (purer) node.
- Higher Gini Index → more heterogeneous (impure) node.
- Faster to compute than entropy (no logarithm calculation).
- More sensitive to changes in class probabilities.
- Used by the **CART** algorithm (scikit-learn default).
- Tends to favour balanced splits (equal-sized child nodes).

---

### Information Gain vs Gini Index

| Feature | Information Gain (Entropy) | Gini Index |
|---|---|---|
| **Formula** | `Entropy(S) − Σ weighted Entropy(Sᵥ)` | `1 − Σ pᵢ²` |
| **Computation** | Slower (uses logarithm) | Faster (no logarithm) |
| **Used By** | ID3, C4.5 algorithms | CART algorithm (sklearn) |
| **Bias** | Favours features with many values | Favours binary splits |
| **Sensitivity** | Less sensitive to probability shifts | More sensitive to shifts |
| **Default in sklearn** | No | Yes (`criterion='gini'`) |
| **Split Type** | Multi-way | Binary |

---

## Building a Decision Tree — Worked Example

### Dataset (3 features, 2 classes)

| X | Y | Z | Class |
|---|---|---|-------|
| 1 | 1 | 1 | I     |
| 1 | 1 | 0 | I     |
| 0 | 0 | 1 | II    |
| 1 | 0 | 0 | II    |

**Initial Entropy of S:**
```
Class I  = 2 samples
Class II = 2 samples

H(S) = −[(2/4)·log₂(2/4) + (2/4)·log₂(2/4)]
     = −[0.5·(−1) + 0.5·(−1)]
     = 1.0
```

**Information Gain for Feature Y:**
```
Y = 1 → {I, I}       → H = 0.0 (pure!)
Y = 0 → {II, II}     → H = 0.0 (pure!)

Gain(S, Y) = 1.0 − [(2/4)·0.0 + (2/4)·0.0] = 1.0  ← MAXIMUM!
```

**Result:** Feature Y gives perfect Information Gain = 1.0 → chosen as root.

**Final Tree:**
```
            [Y = ?]
           /        \
         Y=1         Y=0
          |            |
       Class I ✅   Class II ✅
```

---

## Decision Tree Algorithms

Several algorithms implement the decision tree concept, each with different splitting strategies:

| Algorithm | Split Criterion | Tree Type | Pruning | Key Feature |
|---|---|---|---|---|
| **ID3** | Information Gain (Entropy) | Multi-way | No | Earliest algorithm; handles categorical only |
| **C4.5** | Gain Ratio | Multi-way | Yes (post-prune) | Extension of ID3; handles continuous features |
| **C5.0** | Gain Ratio (improved) | Multi-way | Yes | Faster, less memory; commercial version of C4.5 |
| **CART** | Gini Index / MSE | Binary | Yes | Produces binary trees; used in sklearn |

### Algorithm Comparison at a Glance

```
ID3  →  Entropy + IG  →  Multi-way splits  →  Categorical only  →  No pruning
C4.5 →  Gain Ratio    →  Multi-way splits  →  Continuous + cat  →  Post-pruning
CART →  Gini / MSE    →  Binary splits     →  Both types        →  Cost-complexity pruning
```

---

## Overfitting & Tree Pruning

A fully grown decision tree **memorizes** the training data — capturing every noise pattern — leading to **overfitting** and poor generalization.

### Signs of Overfitting

```
High training accuracy + Low test accuracy = Overfitting
```

### Pruning Strategies

#### 1. Pre-Pruning (Early Stopping)
Stop tree growth before it becomes too complex by enforcing constraints **during** training:

| Parameter | Description |
|---|---|
| `max_depth` | Limit the maximum depth of the tree |
| `min_samples_split` | Minimum samples required to split an internal node |
| `min_samples_leaf` | Minimum samples required to be at a leaf node |
| `max_features` | Limit number of features considered for splitting |
| `min_impurity_decrease` | Only split if impurity decreases by this threshold |

#### 2. Post-Pruning (Cost-Complexity Pruning)
Grow the full tree, then **remove branches** that don't improve generalization:

```python
from sklearn.tree import DecisionTreeClassifier

# Cost-complexity pruning
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas        # Effective alphas
impurities  = path.impurities

# Train trees at each alpha and pick the best via cross-validation
clfs = [DecisionTreeClassifier(ccp_alpha=alpha).fit(X_train, y_train)
        for alpha in ccp_alphas]
```

#### Visual Effect of Pruning

```
Unpruned Tree (overfit):        Pruned Tree (generalised):
        [A]                              [A]
       /   \                            /   \
     [B]   [C]           →           [B]   Leaf
    /   \   |                       /   \
  [D]  [E] [F]                   Leaf  Leaf
  / \   |
 L  L   L
```
---

## Implementation with Scikit-Learn

### Classification — Iris Dataset

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load dataset
X, y = load_iris(return_X_y=True)
target_names = ['Setosa', 'Versicolor', 'Virginica']
feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build Decision Tree
clf = DecisionTreeClassifier(
    criterion='gini',       # 'gini' or 'entropy'
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Visualize the tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=feature_names, class_names=target_names,
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree — Iris Dataset")
plt.tight_layout()
plt.savefig("decision_tree_iris.png", dpi=150)
plt.show()

# Print text rules
print("\nDecision Rules:")
print(export_text(clf, feature_names=feature_names))
```

### Feature Importance

```python
import pandas as pd

importances = pd.Series(clf.feature_importances_, index=feature_names)
importances.sort_values(ascending=True).plot(kind='barh', title='Feature Importances')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()
```

### Multiclass Classification with Custom Dataset

```python
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=500, n_features=10,
                            n_classes=3, n_informative=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = DecisionTreeClassifier(criterion='entropy', max_depth=6)
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
```

---

## Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `criterion` | string | `'gini'` | Split quality measure: `'gini'` or `'entropy'` |
| `max_depth` | int / None | `None` | Maximum depth of tree; `None` = fully grown |
| `min_samples_split` | int / float | `2` | Minimum samples to split an internal node |
| `min_samples_leaf` | int / float | `1` | Minimum samples required at a leaf node |
| `max_features` | int / float / str | `None` | Features to consider for best split |
| `max_leaf_nodes` | int / None | `None` | Limit total number of leaf nodes |
| `min_impurity_decrease` | float | `0.0` | Minimum impurity decrease to trigger a split |
| `ccp_alpha` | float | `0.0` | Complexity parameter for cost-complexity pruning |

### Grid Search with Cross-Validation

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion':        ['gini', 'entropy'],
    'max_depth':        [3, 5, 7, 10, None],
    'min_samples_split':[2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features':     ['sqrt', 'log2', None]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
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

### Effect of max_depth on Performance

```
max_depth = 1  → Underfitting   (too simple; high bias)
max_depth = 5  → Good balance   (generalizes well)
max_depth = 20 → Overfitting    (memorizes training data)

Accuracy
   |
   |          ____________________
   |         /                    \
   |        /                      \  (test starts declining)
   |_______/                        \________________
   |
   +--------------------------------------------> max_depth
        ^optimal
```

---

## Decision Tree for Regression

For **regression tasks**, the CART algorithm uses **Mean Squared Error (MSE)** or **Mean Absolute Error (MAE)** as the split criterion instead of Gini or entropy.

**Split Criterion for Regression:**
```
MSE(S) = (1/|S|) · Σ (yᵢ − ȳ)²
```

Leaf node prediction = **mean** of target values in that node.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=300, n_features=5, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg = DecisionTreeRegressor(
    criterion='squared_error',   # MSE
    max_depth=5,
    min_samples_leaf=5
)
reg.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score
y_pred = reg.predict(X_test)
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²   : {r2_score(y_test, y_pred):.4f}")
```

---

## Ensemble Methods Built on Decision Trees

Decision Trees are the foundation for some of the most powerful ML algorithms:

| Method | Description | Key Advantage |
|---|---|---|
| 🌲 **Random Forest** | Trains many trees on random subsets of features and data (bagging); majority vote | Reduces overfitting; high accuracy |
| ⚡ **Gradient Boosting** | Trains trees sequentially; each corrects errors of the previous (boosting) | Handles complex patterns |
| 🚀 **XGBoost** | Optimized gradient boosting with regularization | Speed + accuracy on tabular data |
| 💡 **LightGBM** | Leaf-wise gradient boosting with histogram binning | Very fast on large datasets |
| 🎯 **CatBoost** | Gradient boosting optimized for categorical features | Minimal preprocessing |
| 📦 **AdaBoost** | Boosting with weighted sample resampling | Reduces bias, handles weak learners |

---

## Applications

Decision Trees are applied across many real-world domains:

| Domain | Application |
|---|---|
| 🏥 **Healthcare** | Disease diagnosis based on patient symptoms and lab results |
| 💳 **Banking & Finance** | Credit risk scoring, loan approval, fraud detection |
| 🛒 **E-Commerce** | Customer segmentation and purchase prediction |
| 🎯 **Marketing** | Churn prediction; targeting the right customers |
| 🌿 **Agriculture** | Predicting crop yield based on soil and weather data |
| ⚙️ **Manufacturing** | Fault detection and predictive maintenance |
| 📧 **Email** | Spam classification based on email content features |
| 🎓 **Education** | Predicting student performance and dropout risk |

---

## Advantages

- ✅ **Highly Interpretable** — decisions can be read and explained as plain if-then rules.
- ✅ **No Feature Scaling Needed** — unaffected by feature magnitude; no normalization or standardization required.
- ✅ **Handles Both Types** — works with numerical and categorical features natively.
- ✅ **Non-Parametric** — makes no assumptions about the data distribution.
- ✅ **Fast Predictions** — prediction is just traversing a tree path: O(log n).
- ✅ **Handles Missing Values** — certain implementations handle missing data gracefully.
- ✅ **Feature Importance** — automatically ranks features by their contribution to splits.
- ✅ **Minimal Data Preprocessing** — no need for dummy encoding or scaling.

---

## Disadvantages

- ❌ **Prone to Overfitting** — a fully grown tree memorizes training data; pruning or ensembles are needed.
- ❌ **Unstable** — small changes in data can produce very different trees (high variance).
- ❌ **Biased toward High-Cardinality Features** — features with many unique values may be favoured unfairly.
- ❌ **Poor Extrapolation** — cannot predict values outside the range of training targets (regression).
- ❌ **Greedy Splitting** — each split is locally optimal but may not produce a globally optimal tree.
- ❌ **Imbalanced Data** — without rebalancing, biased toward majority class.
- ❌ **Not Ideal for XOR-like Patterns** — some relationships require complex trees and many splits.

---

## DT vs SVM vs KNN — Quick Comparison

| Feature | Decision Tree | SVM | KNN |
|---|---|---|---|
| **Interpretability** | ⭐⭐⭐ Very High | ⭐ Low | ⭐⭐ Medium |
| **Training Speed** | Fast | Moderate–Slow | Instant |
| **Prediction Speed** | Very Fast O(depth) | Fast | Slow O(n·d) |
| **Feature Scaling** | Not Required | Required | Required |
| **Overfitting Risk** | High (without pruning) | Low (with soft margin) | High (small K) |
| **Non-Linear Data** | Yes (deep trees) | Yes (kernels) | Yes (local) |
| **Large Datasets** | Good | Poor (quadratic) | Poor (memory) |
| **High-Dim Data** | Moderate | Excellent | Poor |
| **Missing Values** | Handles natively | Cannot handle | Cannot handle |
| **Categorical Data** | Handles natively | Requires encoding | Requires encoding |

---

## Summary Cheat Sheet

```
┌────────────────────────────────────────────────────────────────────┐
│                  Decision Tree Quick Reference                     │
├────────────────────────┬───────────────────────────────────────────┤
│ Algorithm Type         │ Supervised, Non-parametric, Eager Learner │
│ Tasks                  │ Classification & Regression               │
│ Tree Components        │ Root, Internal Nodes, Branches, Leaves    │
│ Split Criteria         │ Information Gain (Entropy), Gini Index    │
│ Entropy Range          │ 0 (pure) to 1 (max impurity)             │
│ Gini Range             │ 0 (pure) to 0.5 (max impurity, binary)   │
│ Algorithms             │ ID3, C4.5, C5.0, CART                    │
│ ID3 / C4.5 Uses        │ Information Gain / Gain Ratio            │
│ CART Uses              │ Gini Index (Classification), MSE (Reg)   │
│ sklearn Default        │ CART with Gini Index                     │
│ Overfitting Fix        │ Pre-Pruning (max_depth) / Post-Pruning   │
│ Feature Scaling        │ NOT Required                              │
│ Prediction Complexity  │ O(depth) — very fast                     │
│ Key Strength           │ Interpretability — readable rule sets     │
│ Key Weakness           │ Overfitting on noisy data                │
│ Ensemble Extensions    │ Random Forest, XGBoost, LightGBM         │
└────────────────────────┴───────────────────────────────────────────┘
```

---

## References

- [GeeksforGeeks — Decision Tree Introduction with Example](https://www.geeksforgeeks.org/machine-learning/decision-tree-introduction-example/)
- [GeeksforGeeks — Decision Tree Algorithm](https://www.geeksforgeeks.org/machine-learning/decision-tree/)
- [GeeksforGeeks — ID3 Algorithm](https://www.geeksforgeeks.org/machine-learning/id3-algorithm/)
- [GeeksforGeeks — Gini Impurity and Information Gain](https://www.geeksforgeeks.org/machine-learning/gini-impurity-and-information-gain/)
- [Scikit-Learn Decision Tree Documentation](https://scikit-learn.org/stable/modules/tree.html)
- [Breiman, L. et al. (1984). Classification and Regression Trees (CART). Wadsworth.](https://www.taylorfrancis.com/books/mono/10.1201/9781315139470/classification-regression-trees-leo-breiman)
- [Quinlan, J.R. (1986). Induction of Decision Trees. Machine Learning, 1(1), 81–106.](https://link.springer.com/article/10.1007/BF00116251)

---