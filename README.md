# 🤖 Machine Learning Models and Their Types

## 📌 Table of Contents

- [Overview](#-overview)
- [Machine Learning Landscape](#-machine-learning-landscape)
- [Part 1 — Supervised Learning](#-part-1--supervised-learning)
  - [What is Supervised Learning?](#-what-is-supervised-learning)
  - [1.1 Regression Models](#11-regression-models)
    - [Linear Regression](#1-linear-regression)
    - [Ridge & Lasso Regression](#2-ridge--lasso-regression)
    - [Polynomial Regression](#3-polynomial-regression)
    - [Support Vector Regression (SVR)](#4-support-vector-regression-svr)
  - [1.2 Classification Models](#12-classification-models)
    - [Logistic Regression](#1-logistic-regression)
    - [K-Nearest Neighbors (KNN)](#2-k-nearest-neighbors-knn)
    - [Naive Bayes](#3-naive-bayes)
    - [Support Vector Machine (SVM)](#4-support-vector-machine-svm)
    - [Decision Tree](#5-decision-tree)
  - [1.3 Ensemble Models](#13-ensemble-models)
    - [Random Forest](#1-random-forest)
    - [Gradient Boosting (GBM)](#2-gradient-boosting-gbm)
    - [XGBoost](#3-xgboost)
    - [LightGBM](#4-lightgbm)
    - [AdaBoost](#5-adaboost)
    - [Stacking & Voting](#6-stacking--voting)
- [Part 2 — Unsupervised Learning](#-part-2--unsupervised-learning)
  - [What is Unsupervised Learning?](#-what-is-unsupervised-learning)
  - [2.1 Clustering Models](#21-clustering-models)
    - [K-Means Clustering](#1-k-means-clustering)
    - [Hierarchical Clustering](#2-hierarchical-clustering)
    - [DBSCAN](#3-dbscan)
    - [Gaussian Mixture Models (GMM)](#4-gaussian-mixture-models-gmm)
  - [2.2 Dimensionality Reduction](#22-dimensionality-reduction)
    - [PCA](#1-pca)
    - [t-SNE](#2-t-sne)
    - [UMAP](#3-umap)
    - [Autoencoders](#4-autoencoders)
  - [2.3 Association Rule Learning](#23-association-rule-learning)
    - [Apriori](#1-apriori)
    - [FP-Growth](#2-fp-growth)
- [Part 3 — Semi-Supervised Learning](#-part-3--semi-supervised-learning)
- [Part 4 — Self-Supervised Learning](#-part-4--self-supervised-learning)
- [Part 5 — Reinforcement Learning](#-part-5--reinforcement-learning)
- [Part 6 — Deep Learning Models](#-part-6--deep-learning-models)
  - [Artificial Neural Network (ANN)](#1-artificial-neural-network-ann)
  - [Convolutional Neural Network (CNN)](#2-convolutional-neural-network-cnn)
  - [Recurrent Neural Network (RNN)](#3-recurrent-neural-network-rnn)
  - [LSTM & GRU](#4-lstm--gru)
  - [Transformer](#5-transformer)
  - [Generative Adversarial Network (GAN)](#6-generative-adversarial-network-gan)
- [Model Selection Guide](#-model-selection-guide)
- [Algorithms by Problem Type](#-algorithms-by-problem-type)
- [Comparison of All Models](#-comparison-of-all-models)
- [Evaluation Metrics Quick Reference](#-evaluation-metrics-quick-reference)
- [Key Takeaways](#-key-takeaways)
- [Prerequisites](#-prerequisites)
- [Further Reading](#-further-reading)

---

## 🧭 Overview

**Machine Learning (ML)** is a subset of Artificial Intelligence where systems learn patterns from data and improve their performance automatically — without being explicitly programmed for every scenario.

```
                         ARTIFICIAL INTELLIGENCE
                                  │
                         MACHINE LEARNING
                    ┌─────────────┼─────────────┐
             Supervised      Unsupervised   Reinforcement
              Learning         Learning       Learning
           ┌────┴─────┐      ┌──┴──────┐
        Regression Classification Clustering Dim. Reduction
                                │
                        DEEP LEARNING
              ┌──────┬──────┬──────┬──────┬──────┐
             ANN    CNN    RNN   LSTM  Transformer GAN
```

---

## 🗺️ Machine Learning Landscape

```
Machine Learning
│
├── SUPERVISED LEARNING          (labelled data — known outputs)
│   ├── Regression               → predict continuous values
│   │   ├── Linear Regression
│   │   ├── Ridge / Lasso
│   │   ├── Polynomial Regression
│   │   └── Support Vector Regression
│   ├── Classification           → predict discrete categories
│   │   ├── Logistic Regression
│   │   ├── K-Nearest Neighbors
│   │   ├── Naive Bayes
│   │   ├── Support Vector Machine
│   │   └── Decision Tree
│   └── Ensemble Methods         → combine multiple models
│       ├── Random Forest
│       ├── Gradient Boosting
│       ├── XGBoost / LightGBM
│       └── AdaBoost / Stacking
│
├── UNSUPERVISED LEARNING        (unlabelled data — discover patterns)
│   ├── Clustering               → group similar data
│   │   ├── K-Means
│   │   ├── Hierarchical
│   │   ├── DBSCAN
│   │   └── GMM
│   ├── Dimensionality Reduction → compress feature space
│   │   ├── PCA
│   │   ├── t-SNE / UMAP
│   │   └── Autoencoders
│   └── Association Rules        → find item co-occurrence patterns
│       ├── Apriori
│       └── FP-Growth
│
├── SEMI-SUPERVISED LEARNING     (small labelled + large unlabelled)
│
├── SELF-SUPERVISED LEARNING     (generate own supervision signals)
│
├── REINFORCEMENT LEARNING       (agent learns via rewards/penalties)
│
└── DEEP LEARNING                (multi-layer neural architectures)
    ├── ANN / MLP
    ├── CNN
    ├── RNN / LSTM / GRU
    ├── Transformer / BERT / GPT
    └── GAN
```

---

## 🟢 Part 1 — Supervised Learning

### 📌 What is Supervised Learning?

Supervised learning trains models on **labelled datasets** where each input has a known corresponding output. The model learns the mapping `f(X) → y` so it can predict outputs for new, unseen inputs.

```
Training:
  Input (X) + Known Label (y)  ──►  Learn mapping f(X) → y

Prediction:
  New Input (X_new)  ──►  f(X_new)  ──►  Predicted Output (ŷ)
```

| Type | Output | Example |
|---|---|---|
| **Regression** | Continuous number | House price, temperature forecast |
| **Classification** | Discrete category | Spam/not spam, disease diagnosis |

---

## 1.1 Regression Models

---

### 1. Linear Regression

**The simplest and most foundational ML algorithm.** Finds the best-fit straight line through data by minimising the sum of squared errors between predictions and actual values.

**Formula:**
```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

Where:
  ŷ  = predicted value
  β₀ = intercept (bias)
  β₁ ... βₙ = coefficients (weights)
  x₁ ... xₙ = input features
```

**How it works:**
```
Actual data:              Best-fit line:
  ●                         /
    ●     ●             ── /── minimise distance between
      ●       ●          /     each point and the line
          ●          ── /      (least squares method)
```

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 11.9, 14.2, 16.0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Intercept    : {model.intercept_:.4f}")
print(f"Coefficient  : {model.coef_[0]:.4f}")
print(f"R² Score     : {r2_score(y_test, y_pred):.4f}")
print(f"MSE          : {mean_squared_error(y_test, y_pred):.4f}")
```

| ✅ Pros | ❌ Cons |
|---|---|
| Simple, fast, interpretable | Assumes linear relationship |
| Works well when relationship is linear | Sensitive to outliers |
| Low computational cost | Cannot capture complex patterns |

**Best for:** House price prediction, sales forecasting, trend analysis

---

### 2. Ridge & Lasso Regression

**Regularised versions of Linear Regression** that add a penalty term to prevent overfitting. Essential when the dataset has many features or multicollinearity.

**Formulas:**
```
Linear Regression : minimise  Σ(yᵢ - ŷᵢ)²

Ridge (L2)        : minimise  Σ(yᵢ - ŷᵢ)² + λ·Σβⱼ²
                              (shrinks all coefficients toward zero)

Lasso (L1)        : minimise  Σ(yᵢ - ŷᵢ)² + λ·Σ|βⱼ|
                              (can shrink coefficients to exactly zero
                               → automatic feature selection!)

ElasticNet        : minimise  Σ(yᵢ - ŷᵢ)² + λ₁·Σ|βⱼ| + λ₂·Σβⱼ²
                              (combines both L1 and L2)
```

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Ridge
ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  Ridge(alpha=1.0))
])
ridge.fit(X_train, y_train)

# Lasso — also performs feature selection
lasso = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  Lasso(alpha=0.1))
])
lasso.fit(X_train, y_train)

# ElasticNet — combines Ridge + Lasso
elastic = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  ElasticNet(alpha=0.1, l1_ratio=0.5))
])
elastic.fit(X_train, y_train)
```

| Model | Penalty | Feature Selection | Best For |
|---|---|---|---|
| **Ridge** | L2 (sum of squares) | No — shrinks all | Multicollinear data |
| **Lasso** | L1 (sum of absolutes) | Yes — zeros out some | High-dimensional, sparse |
| **ElasticNet** | L1 + L2 | Partial | Grouped correlated features |

---

### 3. Polynomial Regression

**Extends Linear Regression** to model non-linear relationships by adding polynomial terms.

```
ŷ = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ
```

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

poly_model = Pipeline([
    ('poly',   PolynomialFeatures(degree=3, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model',  LinearRegression())
])
poly_model.fit(X_train, y_train)
print(f"R² (degree 3): {poly_model.score(X_test, y_test):.4f}")
```

> ⚠️ High degree polynomials easily overfit — always validate with cross-validation.

---

### 4. Support Vector Regression (SVR)

Finds a hyperplane that fits the data within a specified **ε-margin** (epsilon tube). Robust to outliers — only data points outside the margin (support vectors) affect the model.

```python
from sklearn.svm import SVR

svr = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))
])
svr.fit(X_train, y_train)
print(f"SVR R²: {svr.score(X_test, y_test):.4f}")
```

---

## 1.2 Classification Models

---

### 1. Logistic Regression

Despite its name, **Logistic Regression is a classification algorithm**. It estimates the probability of a data point belonging to a class using the **sigmoid function**, outputting values between 0 and 1.

**Formula:**
```
         1
P(y=1) = ─────────────
          1 + e^-(β₀ + β₁x₁ + ... + βₙxₙ)

          Sigmoid function maps any value → (0, 1)
          Threshold: P ≥ 0.5 → Class 1 | P < 0.5 → Class 0
```

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline: scale → classify
model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    LogisticRegression(
                    C=1.0,             # inverse regularisation strength
                    max_iter=1000,
                    solver='lbfgs',
                    multi_class='auto'
               ))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred,
      target_names=data.target_names))
```

| ✅ Pros | ❌ Cons |
|---|---|
| Probabilistic output (0–1) | Assumes linear decision boundary |
| Interpretable coefficients | Poor on complex non-linear data |
| Fast training and prediction | Sensitive to multicollinearity |

**Best for:** Binary/multi-class classification, credit risk, medical diagnosis, spam detection

---

### 2. K-Nearest Neighbors (KNN)

A **non-parametric, instance-based** algorithm. Classifies a new point by finding the k closest training examples and taking a **majority vote** (classification) or **average** (regression).

```
New point ★ → find k=3 nearest neighbours:
  ● ● ○
  ↑ ↑ ↑
  1 2 3   → majority = ● → classify as ●

Distance metric (default): Euclidean
  d = √[(x₁-x₂)² + (y₁-y₂)²]
```

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn',    KNeighborsClassifier())
])

# Find optimal k via grid search
param_grid = {'knn__n_neighbors': range(1, 21, 2)}
grid = GridSearchCV(knn_pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"Best k    : {grid.best_params_['knn__n_neighbors']}")
print(f"Best score: {grid.best_score_*100:.2f}%")
print(f"Test score: {grid.score(X_test, y_test)*100:.2f}%")
```

| ✅ Pros | ❌ Cons |
|---|---|
| No training phase (lazy learner) | Slow prediction on large datasets |
| Naturally handles multi-class | Sensitive to irrelevant features |
| Simple and intuitive | High memory — stores all training data |
| Non-parametric — no distribution assumption | Poor performance in high dimensions |

**Best for:** Recommendation systems, image recognition, anomaly detection

---

### 3. Naive Bayes

A **probabilistic classifier** based on Bayes' Theorem. Assumes all features are conditionally **independent** given the class — the "naive" assumption. Despite this simplification, it works remarkably well for text classification.

**Bayes' Theorem:**
```
         P(X|y) · P(y)
P(y|X) = ──────────────
               P(X)

Where:
  P(y|X)  = posterior probability (class given features)
  P(X|y)  = likelihood (features given class)
  P(y)    = prior probability of class
  P(X)    = evidence (constant for all classes)
```

**Variants:**

| Variant | Distribution Assumed | Best For |
|---|---|---|
| **GaussianNB** | Normal (Gaussian) | Continuous numeric features |
| **MultinomialNB** | Multinomial | Word counts, text classification |
| **BernoulliNB** | Bernoulli | Binary/boolean features |
| **ComplementNB** | Complement of class | Imbalanced text datasets |

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# ── For numeric features ──────────────────────────────────────
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB Accuracy: {gnb.score(X_test, y_test)*100:.2f}%")

# ── For text classification ───────────────────────────────────
texts  = ["I love this product", "Terrible, waste of money",
          "Excellent quality", "Do not buy this"]
labels = [1, 0, 1, 0]   # 1=positive, 0=negative

text_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('clf',        MultinomialNB())
])
text_pipeline.fit(texts, labels)
print(text_pipeline.predict(["Great value for money"]))
```

**Best for:** Spam detection, sentiment analysis, text categorisation, real-time prediction

---

### 4. Support Vector Machine (SVM)

SVMs find the **optimal hyperplane** that maximises the **margin** between classes. Data points closest to the hyperplane are called **support vectors**. Kernel functions extend SVM to non-linear decision boundaries.

```
Linear SVM:
  Class A (●)    Class B (○)
      ●               ○
    ●   ●    ║    ○   ○
      ●      ║  ○      ○
             ║
       ← margin →
      hyperplane   (maximum margin = best generalisation)

Kernel trick: projects data to higher dimensions where it becomes linearly separable
```

**Kernel types:**

| Kernel | Formula | Use Case |
|---|---|---|
| **Linear** | K(x,y) = xᵀy | Linearly separable, high-dimensional (text) |
| **RBF (Gaussian)** | K(x,y) = exp(-γ\|x-y\|²) | Most common; non-linear data |
| **Polynomial** | K(x,y) = (γxᵀy + r)ᵈ | Polynomial decision boundaries |
| **Sigmoid** | K(x,y) = tanh(γxᵀy + r) | Neural-network-like boundaries |

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    SVC(probability=True))
])

# Tune C (regularisation) and kernel
param_grid = {
    'svm__C'     : [0.1, 1, 10, 100],
    'svm__kernel': ['linear', 'rbf', 'poly'],
    'svm__gamma' : ['scale', 'auto']
}
grid_svm = GridSearchCV(svm_pipeline, param_grid, cv=5, n_jobs=-1)
grid_svm.fit(X_train, y_train)

print(f"Best params  : {grid_svm.best_params_}")
print(f"Best CV score: {grid_svm.best_score_*100:.2f}%")
```

| ✅ Pros | ❌ Cons |
|---|---|
| Effective in high dimensions | Slow on large datasets |
| Robust to outliers | Sensitive to feature scaling |
| Works well for text classification | Choice of kernel can be tricky |
| Maximum margin → good generalisation | Not probabilistic (need `probability=True`) |

**Best for:** Image classification, text categorisation, bioinformatics, face detection

---

### 5. Decision Tree

A **tree-structured model** that splits data recursively based on feature values to create a flowchart-like decision path. Highly interpretable — you can visualise the exact rules it uses.

```
                    ┌───────────────────┐
                    │  Age > 30?        │
                    └────────┬──────────┘
                   YES       │        NO
               ┌─────────────┘        └────────────────┐
    ┌──────────────────┐           ┌──────────────────────┐
    │  Income > 50K?   │           │  Student?             │
    └────────┬─────────┘           └─────────┬────────────┘
         YES │      NO                   YES │      NO
      ┌──────┴──┐  ┌──────┐          ┌──────┴──┐  ┌──────┐
      │ Buy=Yes │  │Buy=No│          │ Buy=No  │  │Buy=Yes│
      └─────────┘  └──────┘          └─────────┘  └──────┘
```

**Splitting criteria:**

| Criterion | Used For | Formula |
|---|---|---|
| **Gini Impurity** | Classification | `1 - Σpᵢ²` |
| **Entropy / Info Gain** | Classification | `-Σpᵢ log₂(pᵢ)` |
| **MSE / MAE** | Regression | Mean of squared/absolute errors |

```python
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

dt = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,          # prevent overfitting
    min_samples_split=10, # min samples to split a node
    min_samples_leaf=5,   # min samples in a leaf
    random_state=42
)
dt.fit(X_train, y_train)

# Visualise the tree
plt.figure(figsize=(20, 10))
plot_tree(dt, filled=True, feature_names=data.feature_names,
          class_names=data.target_names, fontsize=8)
plt.title("Decision Tree")
plt.tight_layout()
plt.show()

# Text rules
print(export_text(dt, feature_names=list(data.feature_names)))
print(f"Train accuracy: {dt.score(X_train, y_train)*100:.2f}%")
print(f"Test accuracy : {dt.score(X_test, y_test)*100:.2f}%")
```

| ✅ Pros | ❌ Cons |
|---|---|
| Highly interpretable (visual rules) | Prone to overfitting |
| Handles mixed data types | Unstable — small data changes → different tree |
| No feature scaling needed | Biased toward features with more categories |
| Captures non-linear relationships | Deep trees are hard to interpret |

**Best for:** Rule extraction, fraud rules, medical diagnosis explanation

---

## 1.3 Ensemble Models

Ensemble methods **combine multiple base models** to produce a stronger, more robust predictor. They address the weakness of individual models (high variance or high bias).

```
Three ensemble strategies:

BAGGING          │  BOOSTING          │  STACKING
─────────────────┼────────────────────┼─────────────────────
Train many models│  Train models      │  Train diverse base
on random        │  sequentially;     │  models; use a
subsamples;      │  each corrects     │  meta-model to
average/vote     │  the last one's    │  combine their
their outputs    │  errors            │  predictions
                 │                    │
Random Forest    │  AdaBoost, GBM,   │  Mix of models
(parallel)       │  XGBoost, LGBM    │  + meta-learner
                 │  (sequential)      │
```

---

### 1. Random Forest

An ensemble of many **decision trees** trained on random subsets of data (**bagging**) and random subsets of features. Predictions are made by majority vote (classification) or averaging (regression).

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

rf = RandomForestClassifier(
    n_estimators=200,       # number of trees
    max_depth=None,         # grow fully — bagging prevents overfitting
    max_features='sqrt',    # features per split = √(total features)
    min_samples_leaf=2,
    bootstrap=True,         # bagging (sampling with replacement)
    oob_score=True,         # out-of-bag score estimate
    n_jobs=-1,              # use all CPU cores
    random_state=42
)
rf.fit(X_train, y_train)

print(f"OOB Score    : {rf.oob_score_*100:.2f}%")
print(f"Test Accuracy: {rf.score(X_test, y_test)*100:.2f}%")

# Feature importances
feat_imp = pd.Series(rf.feature_importances_,
                     index=data.feature_names).sort_values(ascending=False)
print("\nTop 5 Features:")
print(feat_imp.head())
```

| ✅ Pros | ❌ Cons |
|---|---|
| Robust to overfitting | Less interpretable than single tree |
| Handles missing values well | Slower than a single Decision Tree |
| Provides feature importance | High memory usage (many trees) |
| Works well out-of-the-box | Not ideal for very sparse data |

**Best for:** General-purpose classification/regression, feature selection, fraud detection

---

### 2. Gradient Boosting (GBM)

Builds trees **sequentially**, where each new tree corrects the **residual errors** of the previous ensemble. Uses gradient descent to minimise a loss function.

```
Iteration 1: Model₁ → predictions → errors (residuals)
Iteration 2: Model₂ trains on residuals of Model₁
Iteration 3: Model₃ trains on residuals of Model₁ + Model₂
...
Final:        F(x) = Model₁ + Model₂ + ... + Modelₙ  (weighted sum)
```

```python
from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier(
    n_estimators=200,       # number of boosting stages
    learning_rate=0.05,     # shrinkage — controls contribution of each tree
    max_depth=4,            # depth of each tree (typically 3-5)
    subsample=0.8,          # fraction of samples per tree (stochastic GBM)
    min_samples_leaf=10,
    random_state=42
)
gbm.fit(X_train, y_train)
print(f"GBM Accuracy: {gbm.score(X_test, y_test)*100:.2f}%")
```

---

### 3. XGBoost

**Extreme Gradient Boosting** — the most popular ML algorithm for structured/tabular data in competitions. Adds regularisation, parallel processing, and better handling of missing values on top of GBM.

```python
# pip install xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,   # features per tree
    reg_alpha=0.1,          # L1 regularisation
    reg_lambda=1.0,         # L2 regularisation
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_pred = xgb_model.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
```

---

### 4. LightGBM

**Light Gradient Boosting Machine** — a faster, more memory-efficient alternative to XGBoost. Uses histogram-based splits and leaf-wise tree growth instead of level-wise.

```python
# pip install lightgbm
import lightgbm as lgb

lgbm = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,          # key parameter — controls complexity
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    random_state=42
)
lgbm.fit(X_train, y_train,
         eval_set=[(X_test, y_test)],
         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

print(f"LightGBM Accuracy: {lgbm.score(X_test, y_test)*100:.2f}%")
```

| Model | Speed | Accuracy | Memory | Best For |
|---|---|---|---|---|
| **GBM** | Slow | High | Medium | Baseline boosting |
| **XGBoost** | Fast | Very High | Medium | Kaggle competitions, structured data |
| **LightGBM** | Very Fast | Very High | Low | Large datasets, high-cardinality |
| **CatBoost** | Fast | Very High | Medium | Datasets with many categorical features |

---

### 5. AdaBoost

**Adaptive Boosting** — trains weak learners (usually shallow trees) sequentially; misclassified samples get higher weights in the next round, forcing the next learner to focus on hard examples.

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # stump
    n_estimators=200,
    learning_rate=0.5,
    random_state=42
)
ada.fit(X_train, y_train)
print(f"AdaBoost Accuracy: {ada.score(X_test, y_test)*100:.2f}%")
```

---

### 6. Stacking & Voting

```python
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ── Voting Classifier ─────────────────────────────────────────
# Hard voting: majority vote | Soft voting: average probabilities
voting_clf = VotingClassifier(
    estimators=[
        ('lr',  LogisticRegression(max_iter=1000)),
        ('svm', SVC(probability=True)),
        ('rf',  RandomForestClassifier(n_estimators=100, random_state=42))
    ],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
print(f"Voting Accuracy: {voting_clf.score(X_test, y_test)*100:.2f}%")

# ── Stacking Classifier ───────────────────────────────────────
# Base models → meta-model learns to combine their predictions
stacking_clf = StackingClassifier(
    estimators=[
        ('rf',  RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(probability=True, kernel='rbf')),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ],
    final_estimator=LogisticRegression(),  # meta-model
    cv=5
)
stacking_clf.fit(X_train, y_train)
print(f"Stacking Accuracy: {stacking_clf.score(X_test, y_test)*100:.2f}%")
```

---

## 🔵 Part 2 — Unsupervised Learning

### 📌 What is Unsupervised Learning?

Unsupervised learning works with **unlabelled data** to discover hidden structures, patterns, and groupings — without any pre-defined output variable to learn from.

```
Input: data without labels
         ┌─────────────────────────────────┐
         │  ●  ●    ○  ○     ▲  ▲         │
         │    ●  ●○    ○  ▲     ▲         │
         │  ●      ○      ▲               │
         └─────────────────────────────────┘

Output: discovered structure
         Cluster 1: ●●● | Cluster 2: ○○○ | Cluster 3: ▲▲▲
```

---

## 2.1 Clustering Models

---

### 1. K-Means Clustering

Partitions data into **k clusters** by iteratively assigning points to the nearest centroid and updating centroids until convergence.

```
Algorithm:
  Step 1: Randomly initialise k centroids
  Step 2: Assign each point to nearest centroid (Euclidean distance)
  Step 3: Recompute centroid as mean of assigned points
  Step 4: Repeat Steps 2–3 until centroids stop moving
```

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# ── Determine optimal k via Elbow Method ──────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, 'bo-', linewidth=2)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (within-cluster SSE)')
plt.title('Elbow Method — Optimal k')
plt.show()

# ── Apply K-Means ──────────────────────────────────────────────
km = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
labels = km.fit_predict(X_scaled)

print(f"Cluster centres shape: {km.cluster_centers_.shape}")
print(f"Cluster labels (first 10): {labels[:10]}")
print(f"Inertia: {km.inertia_:.2f}")
```

| ✅ Pros | ❌ Cons |
|---|---|
| Fast and scalable | Must specify k in advance |
| Simple to implement | Sensitive to initialisation and outliers |
| Works well on spherical clusters | Assumes clusters are convex and similar size |

---

### 2. Hierarchical Clustering

Builds a **tree of clusters (dendrogram)** by either merging (agglomerative) or splitting (divisive) clusters step by step. Does not require specifying k in advance.

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# ── Dendrogram to visualise cluster structure ─────────────────
linkage_matrix = linkage(X_scaled[:50], method='ward')

plt.figure(figsize=(12, 5))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Euclidean Distance')
plt.show()

# ── Apply Agglomerative Clustering ────────────────────────────
agg = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'     # ward, complete, average, single
)
labels_agg = agg.fit_predict(X_scaled)
print(f"Cluster labels: {labels_agg[:10]}")
```

---

### 3. DBSCAN

**Density-Based Spatial Clustering of Applications with Noise.** Groups together points that are closely packed, and marks points in low-density regions as outliers. No need to specify k — discovers the number of clusters automatically.

```
Core point:    has ≥ min_samples neighbours within radius ε
Border point:  within ε of a core point but fewer neighbours
Noise point:   not a core or border point → labelled -1 (outlier)
```

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(
    eps=0.5,            # neighbourhood radius
    min_samples=5       # min points to form a dense region
)
labels_db = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise    = list(labels_db).count(-1)

print(f"Clusters found : {n_clusters}")
print(f"Noise points   : {n_noise}")
print(f"Cluster labels : {set(labels_db)}")
```

| Algorithm | k Needed | Shape | Handles Outliers | Best For |
|---|---|---|---|---|
| **K-Means** | Yes | Spherical | No | Well-separated, equal-size clusters |
| **Hierarchical** | No (cut dendrogram) | Any | No | Small datasets, unknown k |
| **DBSCAN** | No | Arbitrary | Yes | Irregular shapes, noisy data |
| **GMM** | Yes | Elliptical | Partial | Probabilistic cluster membership |

---

### 4. Gaussian Mixture Models (GMM)

A **probabilistic clustering model** that assumes data is generated from a mixture of Gaussian distributions. Unlike K-Means, it gives **soft assignments** — each point has a probability of belonging to each cluster.

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
    n_components=3,
    covariance_type='full',  # full, tied, diag, spherical
    random_state=42,
    max_iter=100
)
gmm.fit(X_scaled)

labels_gmm   = gmm.predict(X_scaled)
probabilities = gmm.predict_proba(X_scaled)

print(f"Cluster labels (first 5): {labels_gmm[:5]}")
print(f"Cluster probabilities    :\n{probabilities[:3].round(3)}")
```

---

## 2.2 Dimensionality Reduction

---

### 1. PCA

*(Covered in depth in the [PCA README](../pca_README.md))*

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA(n_components=0.95)   # retain 95% variance
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
print(f"Reduced to {pca.n_components_} components")
```

---

### 2. t-SNE

**t-Distributed Stochastic Neighbour Embedding** — a non-linear technique for **visualising** high-dimensional data in 2D/3D. Preserves local neighbourhood structure.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(StandardScaler().fit_transform(X))

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='Set1', alpha=0.7)
plt.colorbar(scatter)
plt.title("t-SNE 2D Visualisation")
plt.show()
```

> ⚠️ t-SNE is for **visualisation only** — do not use the t-SNE output as features for downstream models.

---

### 3. UMAP

**Uniform Manifold Approximation and Projection** — faster than t-SNE and preserves both local and global structure better. Suitable for large datasets.

```python
# pip install umap-learn
import umap

reducer = umap.UMAP(n_components=2, random_state=42)
X_umap  = reducer.fit_transform(StandardScaler().fit_transform(X))
```

---

### 4. Autoencoders

Neural network-based dimensionality reduction. Learns a compressed representation (latent space) by training to reconstruct its own input.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

# ── Encoder ───────────────────────────────────────────────────
input_dim  = X.shape[1]
latent_dim = 8

inputs  = tf.keras.Input(shape=(input_dim,))
encoded = layers.Dense(32, activation='relu')(inputs)
encoded = layers.Dense(latent_dim, activation='relu')(encoded)   # bottleneck

# ── Decoder ───────────────────────────────────────────────────
decoded = layers.Dense(32, activation='relu')(encoded)
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

# ── Autoencoder ───────────────────────────────────────────────
autoencoder = Model(inputs, decoded)
encoder     = Model(inputs, encoded)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose=0)

X_encoded = encoder.predict(X_scaled)
print(f"Compressed from {input_dim} → {latent_dim} dimensions")
```

---

## 2.3 Association Rule Learning

Discovers **co-occurrence patterns** in transactional data (e.g., market basket analysis).

**Key metrics:**
```
Support    = P(A ∩ B)           = how often A and B appear together
Confidence = P(A ∩ B) / P(A)   = how often B appears when A appears
Lift       = Confidence / P(B)  = strength above chance (Lift > 1 = useful rule)
```

### 1. Apriori

```python
# pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Transaction data (one-hot encoded)
transactions = pd.DataFrame({
    'Bread': [1,1,0,1,1,0],
    'Milk' : [1,1,1,0,1,1],
    'Eggs' : [0,1,1,0,0,1],
    'Butter':[1,0,1,0,1,0]
})

# Find frequent itemsets
frequent = apriori(transactions, min_support=0.5, use_colnames=True)

# Generate rules
rules = association_rules(frequent, metric='lift', min_threshold=1.0)
rules = rules.sort_values('lift', ascending=False)
print(rules[['antecedents','consequents','support','confidence','lift']].head())
```

### 2. FP-Growth

Faster than Apriori — uses a compact **FP-tree** structure instead of candidate generation.

```python
from mlxtend.frequent_patterns import fpgrowth

frequent_fp = fpgrowth(transactions, min_support=0.5, use_colnames=True)
rules_fp    = association_rules(frequent_fp, metric='lift', min_threshold=1.0)
print(rules_fp.head())
```

---

## 🟡 Part 3 — Semi-Supervised Learning

Semi-supervised learning uses a **small amount of labelled data** and a **large amount of unlabelled data** to train models — bridging supervised and unsupervised approaches.

```
Labelled data   :  100 examples  (small — expensive to label)
Unlabelled data : 10,000 examples (large — cheap to collect)
                       │
              Semi-Supervised Model
                       │
            Better accuracy than supervised
            trained on 100 labels alone
```

**Common approaches:**

| Approach | How It Works |
|---|---|
| **Self-Training** | Train on labelled data → predict on unlabelled → add high-confidence predictions to training set → repeat |
| **Label Propagation** | Spread labels from labelled to unlabelled via graph structure |
| **GANs (semi-supervised)** | Discriminator classifies both real/fake AND predicts class |
| **Pseudo-Labelling** | Teacher model generates "pseudo labels" for unlabelled data used by student model |

```python
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import numpy as np

# Create semi-supervised scenario: mark most labels as -1 (unknown)
y_semi = y.copy()
rng    = np.random.RandomState(42)
mask   = rng.rand(len(y)) < 0.9   # 90% unlabelled
y_semi[mask] = -1                  # -1 = unlabelled in sklearn

# Label Propagation
lp = LabelPropagation(kernel='rbf', gamma=20, max_iter=1000)
lp.fit(X_scaled, y_semi)
print(f"Label Propagation Accuracy: {lp.score(X_test, y_test)*100:.2f}%")

# Label Spreading — more robust to noise
ls = LabelSpreading(kernel='rbf', alpha=0.2)
ls.fit(X_scaled, y_semi)
print(f"Label Spreading Accuracy  : {ls.score(X_test, y_test)*100:.2f}%")
```

**Real-world use cases:** Medical image labelling (few labelled scans), NLP document classification, web content categorisation

---

## 🟣 Part 4 — Self-Supervised Learning

Self-supervised learning generates **supervision signals from the data itself** — no human labelling required. The model learns rich representations by solving pretext tasks.

```
Pretext tasks (self-generated supervision):
  ┌────────────────────────────────────────────────────┐
  │  Text:  predict the next word given previous words  │  → GPT
  │  Text:  predict masked words in a sentence          │  → BERT
  │  Image: predict rotation angle of an image          │  → SimCLR
  │  Image: predict relative position of patches        │  → MAE
  │  Audio: predict next audio segment                  │  → wav2vec
  └────────────────────────────────────────────────────┘
           ↓
  Pre-trained model with rich representations
           ↓
  Fine-tune on small labelled downstream task
```

**Key examples:**

| Model | Domain | Pretext Task |
|---|---|---|
| **BERT** | NLP | Masked Language Modelling + Next Sentence Prediction |
| **GPT** | NLP | Next token prediction (autoregressive) |
| **SimCLR** | Computer Vision | Contrastive learning of augmented image pairs |
| **MAE** | Computer Vision | Masked Autoencoder — reconstruct masked image patches |
| **wav2vec 2.0** | Audio | Predict quantised audio representations |

---

## 🔴 Part 5 — Reinforcement Learning

Reinforcement Learning (RL) trains an **agent** to make sequential decisions by interacting with an **environment**. The agent learns to maximise cumulative **reward** through trial and error — no labelled data required.

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   Agent ──── Action ──────────► Environment             │
│     ▲                                │                  │
│     │                         State + Reward            │
│     └────────────────────────────────┘                  │
│                                                         │
│  Agent goal: maximise cumulative reward over time       │
└─────────────────────────────────────────────────────────┘
```

**Core components:**

| Component | Description | Example |
|---|---|---|
| **Agent** | The learner/decision-maker | Robot, trading bot, game AI |
| **Environment** | The world the agent interacts with | Game engine, stock market |
| **State (s)** | Current situation of the environment | Board position, portfolio value |
| **Action (a)** | What the agent can do | Move left, buy/sell, turn |
| **Reward (r)** | Feedback signal from environment | +1 win, -1 loss, profit/loss |
| **Policy (π)** | Agent's strategy: state → action | Neural network mapping |
| **Value Function** | Expected future reward from a state | Q(s,a) or V(s) |

**RL Algorithms:**

| Algorithm | Type | Key Idea |
|---|---|---|
| **Q-Learning** | Model-free, value-based | Learn Q-value table directly |
| **Deep Q-Network (DQN)** | Model-free, value-based | Q-Learning with deep neural network |
| **SARSA** | Model-free, on-policy | Updates Q using next action taken |
| **Policy Gradient (REINFORCE)** | Policy-based | Optimise policy directly |
| **Actor-Critic (A2C/A3C)** | Policy + value | Actor (policy) + Critic (value function) |
| **PPO** | Policy-based | Proximal Policy Optimisation — stable training |
| **DDPG / TD3 / SAC** | Continuous action | Actor-Critic for continuous action spaces |

```python
# Simple Q-Learning example (grid world)
import numpy as np

# Q-table: states × actions
n_states  = 16   # 4×4 grid
n_actions = 4    # up, down, left, right
Q = np.zeros((n_states, n_actions))

# Hyperparameters
alpha   = 0.1    # learning rate
gamma   = 0.99   # discount factor
epsilon = 0.1    # exploration rate (ε-greedy)

def choose_action(state):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)   # explore
    return np.argmax(Q[state])                # exploit

def update_q(s, a, r, s_next, done):
    target = r if done else r + gamma * np.max(Q[s_next])
    Q[s, a] += alpha * (target - Q[s, a])    # Bellman equation

print("Q-Learning structure initialised.")
print(f"Q-table shape: {Q.shape}")
```

**Real-world applications:**
- 🎮 Game playing — AlphaGo, OpenAI Five (Dota 2), AlphaStar (StarCraft II)
- 🤖 Robotics — manipulation, locomotion, navigation
- 🚗 Autonomous driving — traffic signal control, path planning
- 💹 Algorithmic trading — portfolio management
- 💊 Healthcare — personalised treatment policies

---

## 🧠 Part 6 — Deep Learning Models

Deep Learning uses **multi-layer neural networks** to learn hierarchical feature representations automatically from raw data.

```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → ... → Output Layer
  (features)    (low-level        (high-level         (predictions)
                 patterns)         patterns)
```

---

### 1. Artificial Neural Network (ANN)

The foundational deep learning model. Layers of interconnected neurons learn complex, non-linear mappings from input to output via **backpropagation**.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# ── Build ANN ─────────────────────────────────────────────────
model_ann = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64,  activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32,  activation='relu'),
    layers.Dense(1,   activation='sigmoid')   # binary classification
])

model_ann.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_ann.summary()

# Train
history = model_ann.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

loss, acc = model_ann.evaluate(X_test, y_test, verbose=0)
print(f"ANN Test Accuracy: {acc*100:.2f}%")
```

---

### 2. Convolutional Neural Network (CNN)

Specialised for **grid-structured data** (images, time-series). Uses convolutional filters to learn spatial patterns — edges, textures, objects — hierarchically.

```
Input Image (28×28)
    │
Conv Layer → detects edges and textures
    │
Pooling Layer → reduces spatial size, retains key features
    │
Conv Layer → detects higher-level patterns (shapes, objects)
    │
Flatten → Dense Layers → Output (class probabilities)
```

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# ── CNN for image classification ──────────────────────────────
model_cnn = models.Sequential([
    layers.Input(shape=(28, 28, 1)),          # height, width, channels
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')    # 10 classes
])

model_cnn.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
model_cnn.summary()
```

**Applications:** Image classification, object detection, face recognition, medical imaging

---

### 3. Recurrent Neural Network (RNN)

Designed for **sequential data** (text, time-series, audio). Has a hidden state that carries information from previous time steps — enabling memory of past context.

```
Sequence: [x₁, x₂, x₃, x₄, x₅]

RNN unfolds over time:
  h₁ = f(x₁, h₀)
  h₂ = f(x₂, h₁)
  h₃ = f(x₃, h₂)  ← hidden state carries past context
  ...
  y  = output from final hidden state
```

> ⚠️ Vanilla RNNs suffer from **vanishing gradients** on long sequences — use LSTM or GRU instead.

---

### 4. LSTM & GRU

**Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)** solve the vanishing gradient problem with gating mechanisms that control what information to remember or forget.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# ── LSTM for sequence classification ──────────────────────────
model_lstm = models.Sequential([
    layers.Input(shape=(100, 50)),           # (timesteps, features)
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model_lstm.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])

# ── GRU — lighter alternative to LSTM ─────────────────────────
model_gru = models.Sequential([
    layers.Input(shape=(100, 50)),
    layers.GRU(128, return_sequences=True),
    layers.GRU(64),
    layers.Dense(1, activation='sigmoid')
])
```

| Model | Gates | Parameters | Speed | Best For |
|---|---|---|---|---|
| **RNN** | None | Few | Fast | Very short sequences |
| **LSTM** | 3 (input, forget, output) | Many | Slower | Long sequences, NLP |
| **GRU** | 2 (reset, update) | Medium | Medium | Good LSTM alternative |

---

### 5. Transformer

The **dominant architecture in modern AI**. Uses self-attention mechanisms to capture relationships between all positions in a sequence simultaneously — no recurrence needed. Foundation of BERT, GPT, T5, and all large language models.

```
Input tokens: [The, cat, sat, on, mat]
                │
        Self-Attention:
        each token attends to ALL other tokens simultaneously
                │
        "cat" strongly attends to "sat" and "mat"
        "on"  attends most to "sat" and "mat"
                │
        Multi-Head Attention (parallel attention heads)
                │
        Feed Forward Layer
                │
        Output: context-rich representations
```

```python
import tensorflow as tf
from tensorflow.keras import layers

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att   = layers.MultiHeadAttention(num_heads=num_heads,
                                               key_dim=embed_dim)
        self.ffn   = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.att(x, x)
        x = self.norm1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out, training=training))
        return x
```

**Famous Transformer models:** BERT, GPT-4, T5, LLaMA, Claude, Gemini, Vision Transformer (ViT), Whisper (audio), DALL-E (image)

---

### 6. Generative Adversarial Network (GAN)

Two networks compete: a **Generator** tries to create fake data indistinguishable from real data; a **Discriminator** tries to tell real from fake. Through this adversarial training, the generator learns to produce highly realistic synthetic data.

```
GENERATOR:          Random noise → Fake data (images, text, etc.)
DISCRIMINATOR:      Real/Fake data → Real or Fake?

Training loop:
  1. Generator creates fake sample
  2. Discriminator sees real + fake samples, outputs probability
  3. Generator trained to fool Discriminator
  4. Discriminator trained to detect fakes
  5. Repeat → Generator improves until outputs are photo-realistic
```

```python
import tensorflow as tf
from tensorflow.keras import layers

def build_generator(latent_dim=100, output_dim=784):
    return tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(latent_dim,)),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(output_dim, activation='tanh')    # image pixels
    ])

def build_discriminator(input_dim=784):
    return tf.keras.Sequential([
        layers.Dense(256, activation='leaky_relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='leaky_relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')     # real (1) or fake (0)
    ])

generator     = build_generator()
discriminator = build_discriminator()
print("Generator    :", generator.count_params(), "parameters")
print("Discriminator:", discriminator.count_params(), "parameters")
```

**Applications:** Image synthesis, data augmentation, super-resolution, style transfer, deepfakes (detection)

---

## 🗺️ Model Selection Guide

```
WHAT KIND OF OUTPUT DO YOU NEED?
│
├── Continuous number (price, score, temperature)?
│       └──► REGRESSION
│               ├── Linear data?       → Linear Regression
│               ├── Many features?     → Ridge / Lasso
│               ├── Non-linear?        → SVR, GBM, XGBoost, Random Forest
│               └── Complex patterns?  → ANN, Deep Learning
│
├── Category / Class (yes/no, A/B/C)?
│       └──► CLASSIFICATION
│               ├── Linear boundary?   → Logistic Regression, SVM (linear)
│               ├── Probabilistic?     → Naive Bayes
│               ├── Instance-based?    → KNN
│               ├── Rule-based?        → Decision Tree
│               ├── Best accuracy?     → XGBoost, LightGBM, Random Forest
│               └── Image/text/seq?    → CNN, LSTM, Transformer
│
├── No labels — discover structure?
│       └──► UNSUPERVISED
│               ├── Group similar items?      → K-Means, DBSCAN, GMM
│               ├── Visualise in 2D?          → t-SNE, UMAP
│               ├── Compress features?        → PCA, Autoencoder
│               └── Find item associations?   → Apriori, FP-Growth
│
├── Learn from interaction / decisions?
│       └──► REINFORCEMENT LEARNING
│               ├── Discrete actions?  → DQN, Q-Learning
│               └── Continuous actions?→ PPO, SAC, DDPG
│
└── Sequential / text / image / audio data?
        └──► DEEP LEARNING
                ├── Images           → CNN
                ├── Sequences, text  → LSTM, GRU, Transformer
                ├── Tabular data     → ANN / XGBoost (often better)
                └── Generation       → GAN, Diffusion Models
```

---

## 📋 Algorithms by Problem Type

| Problem | Recommended Algorithms |
|---|---|
| **House price prediction** | Linear Regression, Ridge, XGBoost, Random Forest |
| **Email spam detection** | Naive Bayes, Logistic Regression, SVM |
| **Image classification** | CNN, ResNet, EfficientNet, Vision Transformer |
| **Customer segmentation** | K-Means, DBSCAN, Hierarchical Clustering, GMM |
| **Fraud detection** | Isolation Forest, XGBoost, Random Forest, Autoencoders |
| **Sentiment analysis** | Logistic Regression, LSTM, BERT (Transformer) |
| **Recommendation system** | Collaborative Filtering, Matrix Factorisation, KNN |
| **Time-series forecasting** | LSTM, GRU, Transformer, XGBoost, Prophet |
| **Object detection** | YOLO, Faster R-CNN (CNN-based) |
| **Text generation** | GPT (Transformer), LSTM |
| **Anomaly detection** | DBSCAN, Isolation Forest, Autoencoders, LOF |
| **Game playing** | DQN, PPO, AlphaZero (RL) |
| **Market basket analysis** | Apriori, FP-Growth |
| **Dimensionality reduction** | PCA, t-SNE, UMAP, Autoencoder |

---

## 📊 Comparison of All Models

| Model | Type | Output | Training Speed | Prediction Speed | Interpretable | Handles Non-linearity |
|---|---|---|---|---|---|---|
| Linear Regression | Supervised | Continuous | Very Fast | Very Fast | ✅ Yes | ❌ No |
| Ridge / Lasso | Supervised | Continuous | Fast | Fast | ✅ Yes | ❌ No |
| Logistic Regression | Supervised | Class | Fast | Fast | ✅ Yes | ❌ No |
| KNN | Supervised | Class/Cont. | None (lazy) | Slow | Partial | ✅ Yes |
| Naive Bayes | Supervised | Class | Very Fast | Very Fast | ✅ Yes | ❌ No |
| SVM | Supervised | Class/Cont. | Slow | Medium | ❌ No | ✅ Yes (kernel) |
| Decision Tree | Supervised | Class/Cont. | Fast | Very Fast | ✅ Yes | ✅ Yes |
| Random Forest | Supervised | Class/Cont. | Medium | Medium | Partial | ✅ Yes |
| XGBoost | Supervised | Class/Cont. | Fast | Fast | Partial | ✅ Yes |
| LightGBM | Supervised | Class/Cont. | Very Fast | Very Fast | Partial | ✅ Yes |
| K-Means | Unsupervised | Clusters | Fast | Fast | ✅ Yes | ❌ No |
| DBSCAN | Unsupervised | Clusters | Medium | Slow | Partial | ✅ Yes |
| PCA | Unsupervised | Components | Fast | Fast | Partial | ❌ No |
| t-SNE | Unsupervised | 2D/3D | Slow | — | ❌ No | ✅ Yes |
| ANN / MLP | Deep Learning | Any | Slow | Fast | ❌ No | ✅ Yes |
| CNN | Deep Learning | Class/Det. | Very Slow | Fast | ❌ No | ✅ Yes |
| LSTM / GRU | Deep Learning | Sequence | Very Slow | Medium | ❌ No | ✅ Yes |
| Transformer | Deep Learning | Any | Very Slow | Medium | ❌ No | ✅ Yes |

---

## 📏 Evaluation Metrics Quick Reference

### Regression Metrics

| Metric | Formula | Meaning |
|---|---|---|
| **MAE** | Σ\|y - ŷ\| / n | Average absolute error |
| **MSE** | Σ(y - ŷ)² / n | Penalises large errors more |
| **RMSE** | √MSE | Same unit as target — interpretable |
| **R² Score** | 1 - SS_res/SS_tot | Proportion of variance explained (1 = perfect) |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
print(f"MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
```

### Classification Metrics

| Metric | Formula | When to Use |
|---|---|---|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Balanced classes |
| **Precision** | TP/(TP+FP) | Minimise false positives (spam) |
| **Recall** | TP/(TP+FN) | Minimise false negatives (cancer detection) |
| **F1-Score** | 2×(P×R)/(P+R) | Imbalanced classes |
| **AUC-ROC** | Area under ROC curve | Overall model discrimination ability |

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)

print(f"Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"\n{classification_report(y_test, y_pred)}")
```

### Clustering Metrics

| Metric | Range | Meaning |
|---|---|---|
| **Silhouette Score** | -1 to 1 | Cohesion vs separation (higher = better) |
| **Davies-Bouldin Index** | 0 to ∞ | Cluster compactness (lower = better) |
| **Inertia** | 0 to ∞ | Within-cluster sum of squares (lower = better) |

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

sil = silhouette_score(X_scaled, labels)
dbi = davies_bouldin_score(X_scaled, labels)
print(f"Silhouette Score   : {sil:.4f}")
print(f"Davies-Bouldin Index: {dbi:.4f}")
```

---

## 🔑 Key Takeaways

- ✅ ML is broadly divided into **Supervised, Unsupervised, Semi-Supervised, Self-Supervised, and Reinforcement Learning** — each for a different data and problem context
- 📈 **Regression** predicts continuous values; **Classification** predicts discrete categories
- 🌲 **Ensemble methods** (Random Forest, XGBoost, LightGBM) consistently outperform single models on structured/tabular data
- 🔵 **Clustering** discovers hidden groupings; **Dimensionality Reduction** compresses feature spaces
- 🤖 **Reinforcement Learning** trains agents via rewards without labelled data — ideal for sequential decision-making
- 🧠 **Deep Learning** (CNN, LSTM, Transformer) excels at images, text, and sequential data but requires large datasets
- 🏆 **Transformers** are the dominant architecture in modern AI — powering all large language models
- ⚖️ **No single best model** exists — always validate with cross-validation and compare multiple algorithms
- 🔧 **Always preprocess** — scale features, handle missing values, remove outliers before training
- 📊 Choose **evaluation metrics based on the problem** — accuracy for balanced classes, F1/AUC for imbalanced, RMSE for regression

---

## 🛠️ Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost lightgbm tensorflow mlxtend umap-learn
```

**Core imports used across this guide:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_squared_error, r2_score, silhouette_score)

# Supervised — Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR

# Supervised — Classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Ensemble
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, VotingClassifier, StackingClassifier)
import xgboost as xgb
import lightgbm as lgb

# Unsupervised
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Semi-supervised
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

# Deep Learning
import tensorflow as tf
from tensorflow.keras import layers, models
```

---

## 📚 Further Reading

- [Scikit-learn: Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)
- [Scikit-learn: Unsupervised Learning](https://scikit-learn.org/stable/unsupervised_learning.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [TensorFlow / Keras Documentation](https://www.tensorflow.org/api_docs)
- [Reinforcement Learning — OpenAI Gymnasium](https://gymnasium.farama.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Principal Component Analysis (PCA)](../pca_README.md)
- [Feature Transformation Techniques](../feature_transformation_README.md)
- [Handling Missing Data](../handling_missing_data_README.md)
- [Outlier Detection and Removal](../outlier_detection_README.md)
