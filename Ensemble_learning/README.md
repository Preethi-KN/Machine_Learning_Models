# 🌲 Ensemble Learning 

## 📑 Table of Contents

- [What is Ensemble Learning?](#-what-is-ensemble-learning)
- [Why Decision Trees as Base Learners?](#-why-decision-trees-as-base-learners)
- [Core Idea — The Wisdom of the Crowd](#-core-idea--the-wisdom-of-the-crowd)
- [Types of Ensemble Learning](#-types-of-ensemble-learning)
  - [1. Bagging (Bootstrap Aggregating)](#1-bagging-bootstrap-aggregating)
  - [2. Boosting](#2-boosting)
  - [3. Stacking (Stacked Generalization)](#3-stacking-stacked-generalization)
- [Ensemble Learning Techniques Overview](#-ensemble-learning-techniques-overview)
- [🌳 Random Forest — Full Deep Dive](#-random-forest--full-deep-dive)
  - [What is Random Forest?](#what-is-random-forest)
  - [How Random Forest Works — Step by Step](#how-random-forest-works--step-by-step)
  - [Key Features](#key-features)
  - [Assumptions](#assumptions)
  - [Random Forest for Classification](#random-forest-for-classification-titanic-survival)
  - [Random Forest for Regression](#random-forest-for-regression-house-price-prediction)
  - [Hyperparameters](#important-hyperparameters)
  - [Advantages](#advantages-of-random-forest)
  - [Limitations](#limitations-of-random-forest)
- [Other Ensemble Techniques](#-other-ensemble-techniques)
  - [Random Subspace Method](#1-random-subspace-method-bagging)
  - [Gradient Boosting (GBM)](#2-gradient-boosting-machines-gbm)
  - [XGBoost — Full Deep Dive](#3-extreme-gradient-boosting-xgboost--full-deep-dive)
    - [How XGBoost Works](#how-xgboost-works--step-by-step)
    - [Mathematics Behind XGBoost](#mathematics-behind-xgboost)
    - [How XGBoost Improves Traditional GBM](#how-xgboost-improves-traditional-gradient-boosting)
    - [XGBoost Implementation](#xgboost-implementation--wholesale-customers)
    - [XGBoost Hyperparameters](#xgboost-hyperparameters)
    - [Advantages & Disadvantages](#xgboost-advantages--disadvantages)
  - [AdaBoost](#4-adaboost-adaptive-boosting)
  - [CatBoost](#5-catboost)
  - [Voting Classifier](#6-voting-classifier)
    - [Hard Voting](#hard-voting)
    - [Soft Voting](#soft-voting)
    - [Voting Implementation](#voting-classifier--python-implementation)
    - [When to Use Voting](#when-to-use-voting)
- [Benefits of Ensemble Learning](#-benefits-of-ensemble-learning)
- [Bagging vs Boosting vs Stacking vs Voting — Quick Comparison](#-bagging-vs-boosting-vs-stacking-vs-voting--quick-comparison)
- [When to Use Which Technique?](#-when-to-use-which-technique)
- [Summary](#-summary)

---

## 🤔 What is Ensemble Learning?

**Ensemble Learning** is a machine learning strategy where **multiple models are combined** instead of relying on a single model. Even if the individual models (called *weak learners*) are not very accurate on their own, combining their predictions produces a result that is **more accurate, stable, and reliable** than any single model could achieve.

Think of it like this: instead of trusting one expert's opinion, you poll a group of experts and go with the majority decision. The group collectively makes fewer mistakes.

```
Single Model:          [ Model A ] ──→ Prediction
                              ↑ prone to errors, bias, overfitting

Ensemble:   [ Model A ]
            [ Model B ]  ──→ Combine ──→ Better Prediction ✅
            [ Model C ]
```

### Key Properties

| Property | Description |
|---|---|
| **Multiple Models** | Uses many small/weak models together |
| **Better Accuracy** | Combined results outperform individual models |
| **Reduced Errors** | Mistakes of one model are cancelled out by others |
| **Versatility** | Works with Decision Trees, Neural Networks, SVMs, etc. |

---

## 🌿 Why Decision Trees as Base Learners?

**Decision Trees (DT)** are the most common base learner in ensemble methods because:

- They are **fast to train** and easy to implement
- They are **highly flexible** — can model complex non-linear patterns
- They are **prone to overfitting** when used alone (high variance) — which makes them perfect candidates for ensemble correction
- They are **interpretable** at an individual level
- They handle both **classification and regression** tasks

> A single Decision Tree is like a person who memorises training data — great on training set, poor on new data. Ensembles fix this by averaging out the "memorisation" across many trees.

---

## 💡 Core Idea — The Wisdom of the Crowd

```
Training Data ──→ [Subset 1] → Tree 1 → Prediction 1
               ──→ [Subset 2] → Tree 2 → Prediction 2   →  COMBINE  → Final Prediction
               ──→ [Subset 3] → Tree 3 → Prediction 3
                         ...
```

- **Classification:** Final prediction = **majority vote** across all trees
- **Regression:** Final prediction = **average** of all tree outputs

---

## 🗂️ Types of Ensemble Learning

There are **three main types** of ensemble methods:

---

### 1. Bagging (Bootstrap Aggregating)

**Bagging** trains multiple models **independently and in parallel** on different random subsets of the training data. The subsets are created by **sampling with replacement** (Bootstrap Sampling).

```
Original Dataset
       │
  ┌────┴────┐
  │Bootstrap│ (sampling with replacement)
  └────┬────┘
       │
  ─────┼─────────────────
  │         │         │
Subset1   Subset2   Subset3
  │         │         │
Model1    Model2    Model3
  │         │         │
  └────┬────┘─────────┘
       │
   Aggregate
  (vote/avg)
       │
  Final Output
```

#### How Bagging Works

1. **Bootstrap Sampling:** The dataset is randomly divided into multiple subsets by sampling *with replacement* — so some samples may appear more than once, and others may be absent entirely
2. **Base Model Training:** A separate Decision Tree is trained on each subset, independently and often in parallel for speed
3. **Prediction Aggregation:** Predictions from all models are combined using:
   - **Majority voting** for classification
   - **Averaging** for regression
4. **OOB (Out-of-Bag) Evaluation:** Samples that were *not* included in a training subset act as a natural validation set — no need for separate cross-validation
5. **Final Prediction:** The combined output gives a more reliable, lower-variance result

#### Bagging — Python Implementation

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base classifier
base_classifier = DecisionTreeClassifier()

# Bagging with 10 decision trees
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)
bagging_classifier.fit(X_train, y_train)

# Evaluate
y_pred = bagging_classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
# Output: Accuracy: 1.0
```

**Why it works:** Because each tree sees a slightly different dataset, individual overfitting tendencies cancel out when predictions are averaged.

---

### 2. Boosting

**Boosting** trains models **sequentially** — each new model focuses on the **errors made by the previous models**. Misclassified samples get higher weights so future learners pay more attention to them.

```
Data → Weak Learner 1 → Errors?
                           ↓ (upweight hard samples)
       Weak Learner 2 → Errors?
                           ↓
       Weak Learner 3 → ...
                           ↓
   Weighted Combination of All → Strong Final Model
```

#### How Boosting Works

1. **Initialize Weights:** All training samples start with equal weights
2. **Train Weak Learner:** Train a simple model (often a Decision Tree with depth 1, called a *decision stump*)
3. **Sequential Learning:** Each new model is trained on the *same data* but with updated weights that emphasise the previously misclassified examples
4. **Weight Adjustment:** Misclassified samples receive higher weights; correctly classified samples receive lower weights
5. **Final Prediction:** All weak learners are combined with **weighted voting** — better-performing models get higher votes

#### Key Difference from Bagging

| | Bagging | Boosting |
|---|---|---|
| Training order | Parallel (independent) | Sequential (dependent) |
| Focus | Reduce variance | Reduce bias |
| Sample weighting | Equal | Adaptive (errors get higher weight) |
| Risk | Overfitting less likely | Can overfit noisy data |

#### Boosting — Python Implementation (AdaBoost)

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision stump as weak learner
base_classifier = DecisionTreeClassifier(max_depth=1)

# AdaBoost with 50 weak learners
adaboost_classifier = AdaBoostClassifier(
    base_classifier, n_estimators=50, learning_rate=1.0, random_state=42
)
adaboost_classifier.fit(X_train, y_train)

y_pred = adaboost_classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
# Output: Accuracy: 1.0
```

---

### 3. Stacking (Stacked Generalization)

**Stacking** is a more advanced ensemble technique where **multiple different model types** (not just Decision Trees) are trained, and their predictions are used as **inputs to a final meta-model** that learns the best way to combine them.

```
Training Data
     │
  ┌──┴──────────────────────────┐
  │                             │
Model A (Random Forest)   Model B (SVM)   Model C (KNN)
     │                        │                │
  Pred_A                   Pred_B           Pred_C
     └──────────┬───────────────┘
                │
         Meta-Model (e.g. Logistic Regression)
                │
         Final Prediction
```

- **Level 0 (Base models):** Diverse algorithms trained on the training data
- **Level 1 (Meta-model):** A model trained on the *predictions* of the Level 0 models
- The meta-model learns which base model to trust more for which type of input

> Stacking is less commonly used in practice than Bagging and Boosting due to its complexity, but can achieve state-of-the-art results.

---

## 📊 Ensemble Learning Techniques Overview

| Technique | Category | Base Learner | Key Idea |
|---|---|---|---|
| **Random Forest** | Bagging | Decision Trees | Multiple trees on bootstrap subsets + random feature selection |
| **Random Subspace Method** | Bagging | Any | Random subsets of *features* (not samples) |
| **Gradient Boosting (GBM)** | Boosting | Decision Trees | Sequential trees correcting residual errors |
| **XGBoost** | Boosting | Decision Trees | Optimised GBM with regularisation + parallelism |
| **AdaBoost** | Boosting | Decision Stumps | Adaptive weights on misclassified samples |
| **CatBoost** | Boosting | Decision Trees | Native handling of categorical features |
| **Stacking** | Stacking | Any (mixed) | Meta-model learns from base model predictions |

---

## 🌳 Random Forest — Full Deep Dive

### What is Random Forest?

**Random Forest** is the most popular and widely-used ensemble algorithm. It is a **Bagging** technique that builds many Decision Trees, each trained on a **random bootstrap sample** of the data and using a **random subset of features** at each split. The final prediction is made by aggregating all individual tree predictions.

```
                     Training Dataset
                           │
          ┌────────────────┼────────────────┐
          │                │                │
    Bootstrap 1      Bootstrap 2      Bootstrap 3   ...  Bootstrap N
    (rand sample)    (rand sample)    (rand sample)
          │                │                │
    Tree 1           Tree 2           Tree 3       ...  Tree N
    (rand features)  (rand features)  (rand features)
          │                │                │
     Pred_1           Pred_2           Pred_3
          │                │                │
          └────────────────┼────────────────┘
                           │
              Classification: Majority Vote
              Regression:     Average
                           │
                    Final Prediction ✅
```

---

### How Random Forest Works — Step by Step

#### Step 1: Bootstrap Sampling
From the original dataset of N samples, create T bootstrap samples (random sampling **with replacement**). Each bootstrap sample has the same size N but contains approximately **63.2%** unique samples (the rest are duplicates).

```
Original:  [A, B, C, D, E, F]
Bootstrap: [A, A, C, D, F, F]   ← some repeated, some missing
```

Samples that are *not* selected (~36.8%) become the **Out-of-Bag (OOB)** set — a free internal validation set.

#### Step 2: Feature Randomness at Each Split
When building each tree, at every node split, Random Forest does **not** evaluate all features. It selects a **random subset of m features** out of the total M features:

```
Classification:  m = √M    (square root of total features)
Regression:      m = M/3   (one-third of total features)
```

This creates **decorrelated trees** — trees that are different from each other. If one feature is very dominant, standard bagging would pick it for most trees, reducing diversity. Random feature selection prevents this.

#### Step 3: Grow Full Decision Trees
Each tree is typically grown to its **maximum depth** (no pruning) on its bootstrap sample using only the random feature subset at each split. Individual trees have **high variance but low bias** — ideal for bagging.

#### Step 4: Aggregate Predictions

- **Classification:**
  ```
  Tree 1: Class A
  Tree 2: Class B       → Majority Vote → Final: Class A ✅
  Tree 3: Class A
  ```

- **Regression:**
  ```
  Tree 1: 250,000
  Tree 2: 270,000       → Average → Final: 258,333 ✅
  Tree 3: 255,000
  ```

#### Step 5: OOB Error Estimation
For each sample, collect predictions only from trees that did **not** use that sample in training (OOB trees). Average those predictions to get the OOB error — a reliable estimate of generalisation error without needing a separate validation set.

---

### Key Features

| Feature | Details |
|---|---|
| **Handles Missing Data** | Requires preprocessing (imputation); Scikit-learn does not support raw missing values |
| **Feature Importance** | Measures which features contribute most to predictions using mean decrease in impurity (Gini) or mean decrease in accuracy |
| **Scales with Big Data** | Handles thousands of features and millions of samples efficiently |
| **Dual Task Support** | Works for both classification (`RandomForestClassifier`) and regression (`RandomForestRegressor`) |
| **No Normalisation Needed** | Decision Trees are scale-invariant — no need to standardise features |
| **Robust to Outliers** | Majority voting / averaging naturally reduces outlier influence |

---

### Assumptions

1. **Independence of trees:** Each tree makes its own predictions without communicating with other trees
2. **Randomness:** Each tree is built using random samples and random feature subsets to reduce correlated errors
3. **Sufficient data:** Adequate data ensures different trees learn different patterns, improving diversity and accuracy
4. **Diversity improves accuracy:** Combining different trees that disagree reduces error better than trees that all agree

---

### Random Forest for Classification (Titanic Survival)

**Goal:** Predict whether a Titanic passenger survived based on features like class, age, sex, and fare.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load dataset
titanic_data = pd.read_csv('titanic.csv')
titanic_data = titanic_data.dropna(subset=['Survived'])

# Feature selection and preprocessing
X = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = titanic_data['Survived']

X['Sex'] = X['Sex'].map({'female': 0, 'male': 1})   # Encode categorical
X['Age'] = X['Age'].fillna(X['Age'].median())         # Fill missing values

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Random Forest with 100 trees
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluate
y_pred = rf_classifier.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Single prediction
sample = X_test.iloc[0:1]
prediction = rf_classifier.predict(sample)
print(f"\nSample Passenger: {sample.iloc[0].to_dict()}")
print(f"Predicted: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")
```

**Expected Output:**
```
Accuracy: 0.82

Classification Report:
              precision    recall  f1-score   support
           0       0.84      0.88      0.86       105
           1       0.80      0.74      0.77        74
    accuracy                           0.82       179
```

**Feature Importance Insight:**
After training, you can inspect which features matter most:
```python
import matplotlib.pyplot as plt
import numpy as np

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]

plt.bar(range(len(features)), importances[indices])
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
plt.title("Feature Importances — Random Forest (Titanic)")
plt.show()
```

Typically: `Sex > Fare > Age > Pclass > SibSp > Parch`

---

### Random Forest for Regression (House Price Prediction)

**Goal:** Predict median house values using the California Housing dataset.

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load California Housing data
california_housing = fetch_california_housing()
california_data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
california_data['MEDV'] = california_housing.target

# Split features and target
X = california_data.drop('MEDV', axis=1)
y = california_data['MEDV']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Random Forest Regressor with 100 trees
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Evaluate
y_pred = rf_regressor.predict(X_test)
mse  = mean_squared_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score:    {r2:.2f}")

# Single sample prediction
single_data = X_test.iloc[0].values.reshape(1, -1)
predicted_value = rf_regressor.predict(single_data)
print(f"\nPredicted Value: {predicted_value[0]:.2f}")
print(f"Actual Value:    {y_test.iloc[0]:.2f}")
```

**Expected Output:**
```
Mean Squared Error: 0.26
R-squared Score:    0.80

Predicted Value: 4.15
Actual Value:    4.53
```

**Interpretation:**
- **MSE of 0.26** means the average squared error per prediction is small
- **R² of 0.80** means the model explains 80% of the variance in house prices — a strong result

---

### Important Hyperparameters

| Parameter | Description | Typical Range |
|---|---|---|
| `n_estimators` | Number of trees in the forest | 100–500 |
| `max_depth` | Maximum depth of each tree | `None` (full), or 5–30 |
| `max_features` | Features considered at each split | `"sqrt"` (classification), `"log2"`, or float |
| `min_samples_split` | Minimum samples needed to split a node | 2–10 |
| `min_samples_leaf` | Minimum samples required at a leaf | 1–5 |
| `bootstrap` | Whether to use bootstrap samples | `True` (default) |
| `oob_score` | Whether to use OOB samples for validation | `True` or `False` |
| `n_jobs` | Parallel threads for training | `-1` (use all CPU cores) |
| `random_state` | Seed for reproducibility | Any integer |

**Quick tuning tip:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'max_features': ['sqrt', 'log2']
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best params:", grid_search.best_params_)
```

---

### Advantages of Random Forest

| Advantage | Explanation |
|---|---|
| ✅ **High Accuracy** | Aggregating many trees produces very accurate predictions even on large, complex datasets |
| ✅ **Reduces Overfitting** | Combining many decorrelated trees smooths out individual tree variance |
| ✅ **No Feature Scaling Needed** | Decision Trees are not distance-based — no need to normalise or standardise |
| ✅ **Handles High Dimensionality** | Works well with datasets that have thousands of features |
| ✅ **Built-in Feature Importance** | Automatically ranks features by predictive power |
| ✅ **Works on Missing Data** | After imputation, handles datasets with incomplete values |
| ✅ **OOB Validation** | Free internal cross-validation using out-of-bag samples |
| ✅ **Parallelisable** | Trees are independent — can be trained in parallel with `n_jobs=-1` |

---

### Limitations of Random Forest

| Limitation | Explanation |
|---|---|
| ❌ **Computationally Expensive** | Training 100s of deep trees takes significant memory and CPU time |
| ❌ **Slower Inference** | Predictions require passing data through many trees, slower than a single model |
| ❌ **Hard to Interpret** | Unlike a single Decision Tree, a Random Forest is a "black box" — no simple visual explanation |
| ❌ **Not Ideal for Sparse Data** | Performs less well on text/NLP tasks where linear models or gradient boosting may excel |
| ❌ **Memory Intensive** | Storing hundreds of full Decision Trees requires significant RAM |

---

## 📦 Other Ensemble Techniques

### 1. Random Subspace Method (Bagging)

A variation of Bagging that trains models on **random subsets of features** (columns) rather than random subsets of samples (rows). Increases model diversity and reduces overfitting on high-dimensional data.

```
Feature set: [F1, F2, F3, F4, F5, F6]
Model 1 sees: [F1, F3, F5]
Model 2 sees: [F2, F4, F6]
Model 3 sees: [F1, F2, F6]
→ Combine predictions → Final output
```

---

### 2. Gradient Boosting Machines (GBM)

**GBM** builds trees sequentially, where each tree corrects the **residual errors** (gradient of the loss function) of the combined model so far. Unlike AdaBoost which reweights samples, GBM fits the next tree directly to the *errors* (residuals).

```
Prediction = Tree1 + λ·Tree2 + λ·Tree3 + ... + λ·TreeN
```

Where `λ` is the **learning rate** controlling the contribution of each tree.

```python
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbc.fit(X_train, y_train)
print("GBM Accuracy:", accuracy_score(y_test, gbc.predict(X_test)))
```

---

### 3. Extreme Gradient Boosting (XGBoost) — Full Deep Dive

**XGBoost** (eXtreme Gradient Boosting) is a highly optimised, production-grade implementation of Gradient Boosting developed by Tianqi Chen (2014). It extends traditional Gradient Boosting with regularisation, parallelism, and second-order gradient optimisation, making it one of the most powerful and widely-used algorithms for structured/tabular data.

> XGBoost has won more Kaggle competitions than any other algorithm. It is the go-to for tabular data in both competitions and production systems.

```
Traditional GBM                     XGBoost
─────────────────────────────────────────────────────────
Sequential tree building    ──→     Still sequential, but
No regularisation           ──→     L1 + L2 regularisation
First-order gradients only  ──→     Second-order (gradient + Hessian)
Slow on large datasets      ──→     Parallel + cache-aware processing
Struggles with missing data ──→     Sparsity-aware split finding
Greedy full splits          ──→     Approximate split + pruning (max_depth)
```

---

#### How XGBoost Works — Step by Step

```
Step 1: Start with a base prediction
        ŷᵢ⁽⁰⁾ = 0   (or mean of target for regression)

Step 2: Calculate errors (residuals) between predictions and actual values

Step 3: Train Tree 1 on those residuals
        ŷᵢ⁽¹⁾ = ŷᵢ⁽⁰⁾ + η · f₁(xᵢ)

Step 4: Compute NEW residuals from updated predictions

Step 5: Train Tree 2 on the new residuals
        ŷᵢ⁽²⁾ = ŷᵢ⁽¹⁾ + η · f₂(xᵢ)

Step 6: Repeat until K trees are trained

Final:  ŷᵢ = Σ(k=1 to K)  η · fₖ(xᵢ)
```

Where:
- `η` = **learning rate** (shrinks each tree's contribution)
- `fₖ(xᵢ)` = prediction of the k-th tree for sample i
- `K` = total number of trees

Each new tree is not trained on the raw data, but on the **gradient of the loss function** — the direction of steepest error reduction.

---

#### Mathematics Behind XGBoost

##### 1. Prediction Formula

The final prediction is a sum of all K tree outputs:

```
ŷᵢ = Σ fₖ(xᵢ)   for k = 1 to K
```

##### 2. Objective Function

XGBoost minimises an objective that has two parts — a **loss term** and a **regularisation term**:

```
Obj(θ) = Σᵢ l(yᵢ, ŷᵢ)  +  Σₖ Ω(fₖ)
          └─────────────┘    └────────┘
            Loss Function    Regularisation
         (how wrong we are)  (how complex trees are)
```

- `l(yᵢ, ŷᵢ)` — loss function measuring prediction error
  - For regression: Mean Squared Error (MSE)
  - For classification: Log-loss / Cross-entropy
- `Ω(fₖ)` — regularisation penalising tree complexity

##### 3. Regularisation Term

```
Ω(fₜ) = γT  +  ½λ Σⱼ wⱼ²

Where:
  T  = number of leaves in the tree
  γ  = minimum gain required to make a split (leaf penalty)
  λ  = L2 regularisation on leaf weights (prevents large weights)
  wⱼ = weight/score at leaf j
```

This forces trees to stay **simple** — fewer leaves and smaller scores — which prevents overfitting.

##### 4. Iterative Tree Addition

At each step t, one new tree fₜ is added to correct the current errors:

```
ŷᵢ⁽ᵗ⁾ = ŷᵢ⁽ᵗ⁻¹⁾ + fₜ(xᵢ)
```

XGBoost uses a **second-order Taylor expansion** of the loss function to approximate the objective at step t:

```
Obj(t) ≈ Σᵢ [gᵢ fₜ(xᵢ)  +  ½ hᵢ fₜ(xᵢ)²]  +  Ω(fₜ)

Where:
  gᵢ = ∂l(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾)/∂ŷᵢ⁽ᵗ⁻¹⁾    ← first-order gradient
  hᵢ = ∂²l(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾)/∂(ŷᵢ⁽ᵗ⁻¹⁾)²  ← second-order Hessian
```

Using both **gradient (gᵢ)** and **Hessian (hᵢ)** gives XGBoost a much more accurate picture of the loss landscape compared to GBM which only uses the gradient. This enables **faster convergence** and **better generalisation**.

##### 5. Information Gain for Splitting

When deciding how to split a node, XGBoost computes a **gain score** for every possible split and picks the one with the highest gain:

```
Gain = ½ [ GL²/(HL+λ)  +  GR²/(HR+λ)  -  (GL+GR)²/(HL+HR+λ) ]  -  γ

Where:
  GL, GR = sum of gradients in left and right child nodes
  HL, HR = sum of Hessians in left and right child nodes
  λ      = L2 regularisation parameter
  γ      = minimum gain threshold (split only if Gain > 0)
```

If `Gain ≤ 0`, the split is **pruned** — this is how XGBoost avoids unnecessary splits automatically.

---

#### How XGBoost Improves Traditional Gradient Boosting

##### 1. Preventing Overfitting

XGBoost has multiple layers of overfitting protection:

| Mechanism | How it Works |
|---|---|
| **Learning rate (eta/η)** | Scales each tree's contribution — lower = more conservative, needs more trees |
| **L1 regularisation (alpha)** | Adds penalty proportional to absolute leaf weights (sparse solutions) |
| **L2 regularisation (lambda)** | Adds penalty proportional to squared leaf weights (smooth solutions) |
| **Tree pruning (gamma)** | Removes splits that don't improve objective by at least γ |
| **max_depth** | Limits how deep trees can grow |
| **subsample** | Uses only a fraction of rows for each tree (like bagging) |
| **colsample_bytree** | Uses only a fraction of columns (features) for each tree |

##### 2. Level-Wise (Breadth-First) Tree Growth

```
Depth-First (traditional):            Level-Wise (XGBoost):
─────────────────────────             ──────────────────────
         Root                                  Root
        /    \                                /    \
      N1      N2        →          Level 1: N1    N2   (both evaluated)
      /\                                    /\    /\
    N3  N4                         Level 2: N3 N4 N5 N6 (best splits chosen)
```

By evaluating **all nodes at the same depth simultaneously**, XGBoost can identify the globally best splits at each level rather than committing to a branch early.

##### 3. Handling Missing Data — Sparsity-Aware Split Finding

XGBoost does not require imputation before training:

```
Training: For each split, XGBoost tries routing missing values to BOTH
          left and right child, and picks whichever direction reduces loss more.
          This becomes the "default direction" for that node.

Prediction: Any sample with a missing feature automatically follows
            the learned default direction.
```

This makes XGBoost **natively robust** to missing values — a major advantage over Random Forest (Scikit-learn) which requires preprocessing.

##### 4. Cache-Aware Access

XGBoost organises gradient and Hessian data in memory to match CPU cache lines, reducing expensive main-memory reads and dramatically speeding up computation on large datasets.

##### 5. Approximate Greedy Algorithm

For very large datasets, checking every possible split for every feature at every node is too slow. XGBoost uses **weighted quantile sketches** to pre-bucket feature values into approximate split candidates:

```
Instead of: Check all 1,000,000 unique values of feature F
XGBoost:    Use ~256 candidate split points (quantile buckets) → 99.97% as good, 1000× faster
```

---

#### XGBoost Implementation — Wholesale Customers

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# ── Step 1: Load dataset ──────────────────────────────────────────────────────
# Download: https://media.geeksforgeeks.org/wp-content/uploads/20260314100609094989/Wholesale-customers-data.csv
df = pd.read_csv("Wholesale-customers-data.csv")
print(df.head())

# ── Step 2: Prepare features and target ──────────────────────────────────────
X = df.drop('Channel', axis=1)
y = df['Channel'].map({1: 1, 2: 0})   # Binary: 1=Horeca, 0=Retail

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ── Step 3: Define and train the model ───────────────────────────────────────
params = {
    'objective':      'binary:logistic',   # Binary classification
    'max_depth':       4,                  # Tree depth limit
    'learning_rate':   0.1,                # Shrinkage per tree (eta)
    'n_estimators':    100,                # Number of trees
    'alpha':           10                  # L1 regularisation
}

model = XGBClassifier(**params)
model.fit(X_train, y_train)

# ── Step 4: Evaluate ─────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Step 5: Confusion matrix heatmap ─────────────────────────────────────────
plt.figure(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.show()

# ── Step 6: Feature importance ───────────────────────────────────────────────
plt.figure(figsize=(8, 6))
xgb.plot_importance(model)
plt.title("Feature Importance")
plt.show()

# ── Step 7: Visualise one tree ───────────────────────────────────────────────
plt.figure(figsize=(20, 10))
xgb.plot_tree(model, num_trees=0)
plt.show()
```

**Expected Output:**
```
Accuracy: 0.92

Classification Report:
              precision    recall  f1-score   support
           0       0.94      0.94      0.94        89
           1       0.89      0.89      0.89        43
    accuracy                           0.92       132
```

---

#### XGBoost Hyperparameters

| Parameter | Description | Typical Value |
|---|---|---|
| `n_estimators` | Number of boosting rounds (trees) | 100–1000 |
| `learning_rate` (eta) | Shrinkage per tree; lower = more robust but needs more trees | 0.01–0.3 |
| `max_depth` | Maximum depth of each tree | 3–8 |
| `subsample` | Fraction of training rows used per tree | 0.6–1.0 |
| `colsample_bytree` | Fraction of features used per tree | 0.6–1.0 |
| `colsample_bylevel` | Fraction of features used per tree level | 0.6–1.0 |
| `alpha` | L1 regularisation on leaf weights | 0, 1, 10 |
| `lambda` | L2 regularisation on leaf weights | 1 (default) |
| `gamma` | Minimum loss reduction for a split | 0–5 |
| `min_child_weight` | Minimum sum of Hessians in a leaf | 1–10 |
| `objective` | Loss function | `binary:logistic`, `multi:softmax`, `reg:squarederror` |
| `eval_metric` | Metric for validation | `logloss`, `rmse`, `auc` |
| `n_jobs` | Parallelism | `-1` (all cores) |
| `tree_method` | Algorithm for tree building | `hist` (fast), `gpu_hist` (GPU) |
| `early_stopping_rounds` | Stop if validation metric doesn't improve | 10–50 |

**Tuning template:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators':    [100, 300, 500],
    'max_depth':       [3, 5, 7],
    'learning_rate':   [0.01, 0.05, 0.1],
    'subsample':       [0.8, 1.0],
    'colsample_bytree':[0.8, 1.0]
}

grid = GridSearchCV(
    XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)
```

---

#### XGBoost Advantages & Disadvantages

**Advantages:**

| Advantage | Detail |
|---|---|
| ✅ **High accuracy** | Consistently outperforms most single models on tabular data |
| ✅ **Built-in regularisation** | L1 + L2 + gamma + pruning prevent overfitting at multiple levels |
| ✅ **Handles missing values** | Sparsity-aware split finding — no imputation required |
| ✅ **Feature importance** | Automatic ranking of features by gain, weight, or cover |
| ✅ **Parallel & GPU support** | `tree_method='gpu_hist'` can be 10–50× faster than CPU |
| ✅ **Scalable** | Handles millions of rows and thousands of features efficiently |
| ✅ **Cross-validation built-in** | `xgb.cv()` for hyperparameter tuning without extra code |
| ✅ **Early stopping** | Prevents overfitting by stopping when validation metric plateaus |

**Disadvantages:**

| Disadvantage | Detail |
|---|---|
| ❌ **Computationally intensive** | Slower to train than Random Forest on small datasets |
| ❌ **Sensitive to noisy/outlier data** | Boosting can amplify noise — requires careful preprocessing |
| ❌ **Many hyperparameters** | Requires more tuning effort than Random Forest |
| ❌ **Can overfit small datasets** | With too many trees and no regularisation, memorises training data |
| ❌ **Limited interpretability** | Black-box model — SHAP values needed for explainability |

> **Tip for interpretability:** Use `shap` library with XGBoost:
> ```python
> import shap
> explainer = shap.TreeExplainer(model)
> shap_values = explainer.shap_values(X_test)
> shap.summary_plot(shap_values, X_test)
> ```

---

### 4. AdaBoost (Adaptive Boosting)

**AdaBoost** is the original boosting algorithm. It focuses on **challenging examples** by increasing their sample weights after each round, so future weak learners pay extra attention to the hard cases.

```
Round 1: All samples equal weight
         → Train Stump1 → Misclassifies [Sample 3, 7]
Round 2: Sample 3, 7 get HIGHER weight
         → Train Stump2 → Misclassifies [Sample 1, 5]
Round 3: Sample 1, 5 get HIGHER weight
         → ...
Final:   Weighted sum of all stumps → Strong Classifier
```

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),  # Decision stump
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada.fit(X_train, y_train)
print("AdaBoost Accuracy:", accuracy_score(y_test, ada.predict(X_test)))
```

---

### 5. CatBoost

**CatBoost** is a boosting algorithm developed by Yandex that specialises in **categorical feature handling** without requiring manual encoding like One-Hot Encoding. Key advantages:

- Handles categorical features **natively** using ordered target statistics
- **Automatic overfitting detection** during training with built-in validation
- Very high predictive accuracy out of the box with minimal tuning
- Supports both GPU and CPU training

```python
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0)
cat_model.fit(X_train, y_train, cat_features=['Sex'])  # specify categorical columns
print("CatBoost Accuracy:", accuracy_score(y_test, cat_model.predict(X_test)))
```

---

### 6. Voting Classifier

**Voting** is one of the simplest and most intuitive ensemble techniques. It combines predictions from **multiple different model types** (not just Decision Trees) by letting them **vote** on the final answer. Unlike Bagging which uses the same model type, Voting typically combines diverse classifiers (e.g., Logistic Regression + Decision Tree + SVM) to benefit from their complementary strengths.

```
Input Data
    │
    ├──→ Logistic Regression ──→ Class A
    ├──→ Decision Tree       ──→ Class B     →  Vote  →  Final Prediction
    ├──→ Random Forest       ──→ Class A
    └──→ SVM                 ──→ Class A
                                              Majority: Class A ✅
```

There are two types of Voting: **Hard Voting** and **Soft Voting**.

---

#### Hard Voting

In **Hard Voting**, each classifier casts a **single vote** for its predicted class label. The class that receives the **most votes** (majority) becomes the final prediction.

```
Sample: [5.1, 3.5, 1.4, 0.2]

Model 1 (LogReg):    → Class 0   ← vote
Model 2 (DT):        → Class 1   ← vote
Model 3 (RF):        → Class 0   ← vote
Model 4 (SVM):       → Class 0   ← vote

Tally: Class 0 = 3 votes,  Class 1 = 1 vote
Final Prediction: Class 0  ✅
```

Hard Voting counts raw class predictions and goes with the crowd. Every model has an **equal say** regardless of its confidence or accuracy.

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define individual classifiers
clf1 = LogisticRegression(max_iter=200, random_state=42)
clf2 = DecisionTreeClassifier(random_state=42)
clf3 = SVC(random_state=42)             # Note: SVC doesn't output probabilities by default

# Hard Voting Ensemble
hard_voting = VotingClassifier(
    estimators=[('lr', clf1), ('dt', clf2), ('svc', clf3)],
    voting='hard'
)
hard_voting.fit(X_train, y_train)
y_pred = hard_voting.predict(X_test)

print(f"Hard Voting Accuracy: {accuracy_score(y_test, y_pred):.2f}")
# Output: Hard Voting Accuracy: 1.00
```

---

#### Soft Voting

In **Soft Voting**, each classifier outputs a **probability for each class** instead of a single label. The probabilities are **averaged** across all classifiers and the class with the **highest average probability** wins.

```
Sample: [5.1, 3.5, 1.4, 0.2]

              Class 0   Class 1   Class 2
Model 1 (LR): [0.70,    0.20,     0.10]
Model 2 (DT): [0.90,    0.05,     0.05]
Model 3 (RF): [0.80,    0.15,     0.05]
                ─────────────────────────
Average:       [0.80,    0.13,     0.07]

Final Prediction: Class 0 (highest avg probability = 0.80) ✅
```

Soft Voting is **generally more powerful** than Hard Voting because it uses richer information — confident predictions carry more weight than uncertain ones.

```python
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers that support predict_proba
clf1 = LogisticRegression(max_iter=200, random_state=42)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf3 = GaussianNB()

# Soft Voting Ensemble
soft_voting = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    voting='soft'       # Uses predicted probabilities
)
soft_voting.fit(X_train, y_train)
y_pred = soft_voting.predict(X_test)

print(f"Soft Voting Accuracy: {accuracy_score(y_test, y_pred):.2f}")
# Output: Soft Voting Accuracy: 1.00
```

> **Note:** Soft Voting requires all classifiers to support `predict_proba()`. SVC needs `probability=True` to work in soft voting: `SVC(probability=True)`.

---

#### Weighted Voting

You can assign **different weights** to classifiers based on their individual accuracy, giving stronger models more influence:

```python
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf1 = LogisticRegression(max_iter=200, random_state=42)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf3 = GaussianNB()

# Weighted Soft Voting — RF gets 2× the weight
weighted_voting = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    voting='soft',
    weights=[1, 2, 1]     # RF is trusted twice as much
)
weighted_voting.fit(X_train, y_train)
print(f"Weighted Voting Accuracy: {accuracy_score(y_test, weighted_voting.predict(X_test)):.2f}")
```

---

#### Voting Classifier — Python Implementation

Here is a complete comparison of Hard Voting, Soft Voting, and individual models side by side:

```python
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import numpy as np

data = load_iris()
X, y = data.data, data.target

# Individual classifiers
lr  = LogisticRegression(max_iter=200, random_state=42)
dt  = DecisionTreeClassifier(random_state=42)
rf  = RandomForestClassifier(n_estimators=50, random_state=42)
gnb = GaussianNB()

# Hard Voting
hard_vc = VotingClassifier(
    estimators=[('lr', lr), ('dt', dt), ('rf', rf), ('gnb', gnb)],
    voting='hard'
)

# Soft Voting
soft_vc = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('gnb', gnb)],
    voting='soft'
)

# Compare all using 5-fold cross-validation
models = [('Logistic Regression', lr),
          ('Decision Tree', dt),
          ('Random Forest', rf),
          ('GaussianNB', gnb),
          ('Hard Voting', hard_vc),
          ('Soft Voting', soft_vc)]

for name, model in models:
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{name:25s} → Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Expected Output:**
```
Logistic Regression       → Accuracy: 0.953 ± 0.027
Decision Tree             → Accuracy: 0.947 ± 0.040
Random Forest             → Accuracy: 0.960 ± 0.027
GaussianNB                → Accuracy: 0.953 ± 0.027
Hard Voting               → Accuracy: 0.960 ± 0.027
Soft Voting               → Accuracy: 0.967 ± 0.021  ← best & most stable
```

Soft Voting achieves the **highest accuracy and lowest variance**, confirming that averaging probabilities extracts more signal than majority vote alone.

---

#### Hard Voting vs Soft Voting — Comparison

| Aspect | Hard Voting | Soft Voting |
|---|---|---|
| **Mechanism** | Majority vote on class labels | Average of class probabilities |
| **Information used** | Binary predictions only | Full confidence/probability distribution |
| **Accuracy** | Good | Usually better (uses more info) |
| **Requires `predict_proba`** | No | Yes (all classifiers must support it) |
| **Works with SVC** | Yes | Only with `SVC(probability=True)` |
| **Weighted variant** | Yes (`weights` param) | Yes (`weights` param) |
| **When to use** | When models can't output probabilities | When all models support probabilities |

---

#### When to Use Voting

```
✅ Use Voting when:
   - You have 3–5 diverse, well-tuned individual models
   - Models have complementary strengths (LR good on linear, RF on non-linear)
   - You want a quick accuracy boost without complex meta-learning
   - Interpretability matters more than peak performance

❌ Avoid Voting when:
   - Your models are all the same type (use Bagging instead)
   - Your models have very different accuracy levels (worse models drag down better ones)
   - You need the absolute best performance (Stacking usually beats Voting)
   - You're in a competition (Stacking >> Voting for leaderboard gains)
```

---

## ✅ Benefits of Ensemble Learning

| Benefit | Description |
|---|---|
| **Reduced Overfitting** | Aggregating multiple models reduces individual overfitting tendencies |
| **Improved Generalisation** | Performs better on unseen data by balancing variance and bias |
| **Increased Accuracy** | Combined models achieve higher predictive accuracy than any single model |
| **Robustness to Noise** | Noisy/incorrect data points are averaged out across multiple predictions |
| **Flexibility** | Works with Decision Trees, Neural Networks, SVMs, and more |
| **Bias-Variance Tradeoff** | Bagging reduces variance; Boosting reduces bias |

---

## ⚖️ Bagging vs Boosting vs Stacking vs Voting — Quick Comparison

| Aspect | Bagging | Boosting | Stacking | Voting |
|---|---|---|---|---|
| **Training** | Parallel (independent) | Sequential (dependent) | Two-level (parallel + meta) | Parallel (independent) |
| **Goal** | Reduce variance | Reduce bias | Combine diverse model types | Combine diverse model types |
| **Sample handling** | Bootstrap (with replacement) | Weighted samples | All data for each base model | All data for each model |
| **Primary algorithm** | Random Forest | AdaBoost, XGBoost, GBM | Any mixed models | Any mixed models |
| **Overfitting risk** | Low | Medium (needs tuning) | Low (meta-model generalises) | Low |
| **Speed** | Fast (parallel) | Slower (sequential) | Slowest (multi-level) | Fast (parallel) |
| **Interpretability** | Low | Low | Very Low | Medium |
| **Best for** | High-variance models (DT) | High-bias weak learners | Diverse model ensembles | Quick wins with diverse models |
| **Model types** | Same model type | Same model type (weak) | Mixed model types | Mixed model types |
| **Combination method** | Vote / Average | Weighted sum | Meta-model prediction | Hard vote / Soft probability avg |

---

## 🎯 When to Use Which Technique?

```
Your model is overfitting (high variance)?
    └── Try BAGGING → Random Forest

Your model is underfitting (high bias)?
    └── Try BOOSTING → XGBoost / GBM / AdaBoost

You have lots of categorical features?
    └── Try BOOSTING → CatBoost

You have 3–5 good but diverse models already?
    └── Try VOTING → VotingClassifier (soft voting preferred)

You want maximum accuracy and have diverse models?
    └── Try STACKING (beats Voting in most cases)

You have tabular data and need a fast, reliable baseline?
    └── Try Random Forest first (n_estimators=100)

You're in a Kaggle competition?
    └── Try XGBoost or LightGBM → then ensemble winners with Stacking
```

---

## 📝 Summary

```
ENSEMBLE LEARNING
│
├── BAGGING (Reduce Variance)
│   ├── Bootstrap sampling (with replacement)
│   ├── Independent/parallel training
│   ├── Aggregate by voting or averaging
│   ├── Random Forest ★ (most popular)
│   └── Random Subspace Method
│
├── BOOSTING (Reduce Bias)
│   ├── Sequential training — each tree fixes previous errors
│   ├── Focus on previous errors / residuals
│   ├── Weighted combination of learners
│   ├── AdaBoost (adaptive sample reweighting)
│   ├── GBM (residual fitting with gradient descent)
│   ├── XGBoost (optimised GBM + L1/L2 + 2nd-order gradients) ★
│   └── CatBoost (categorical feature specialist)
│
├── STACKING (Combine Diverse Models)
│   ├── Level 0: Multiple diverse base models (parallel)
│   ├── Level 1: Meta-model learns from base predictions
│   └── Most complex but potentially most powerful
│
└── VOTING (Simple Diversity Combination)
    ├── Hard Voting  — majority class label vote
    ├── Soft Voting  — average class probabilities ★
    └── Weighted     — trust stronger models more
```

### Random Forest — At a Glance

| Property | Value |
|---|---|
| Type | Bagging Ensemble |
| Base Learner | Decision Tree |
| Sampling | Bootstrap (with replacement) + Random feature subsets |
| Classification output | Majority vote |
| Regression output | Average |
| Key hyperparameter | `n_estimators` (number of trees) |
| Feature selection per split | `sqrt(M)` for classification, `M/3` for regression |
| Validation | Out-of-Bag (OOB) score |
| Strengths | Accuracy, robustness, feature importance, no scaling needed |
| Weaknesses | Slow on very large datasets, less interpretable |

---

## 📚 References

- [GeeksforGeeks — A Comprehensive Guide to Ensemble Learning](https://www.geeksforgeeks.org/machine-learning/a-comprehensive-guide-to-ensemble-learning/)
- [GeeksforGeeks — Random Forest Algorithm in Machine Learning](https://www.geeksforgeeks.org/machine-learning/random-forest-algorithm-in-machine-learning/)
- [GeeksforGeeks — XGBoost](https://www.geeksforgeeks.org/machine-learning/xgboost/)
- [Scikit-learn — Ensemble Methods Documentation](https://scikit-learn.org/stable/modules/ensemble.html)
- [Scikit-learn — VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [SHAP — Explainability for XGBoost](https://shap.readthedocs.io/en/latest/)

---