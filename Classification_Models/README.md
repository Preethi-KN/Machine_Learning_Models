# 📊 Logistic Regression in Machine Learning
---

## 📌 Table of Contents

1. [What is Logistic Regression?](#what-is-logistic-regression)
2. [Types of Logistic Regression](#types-of-logistic-regression)
3. [Assumptions of Logistic Regression](#assumptions-of-logistic-regression)
4. [The Sigmoid Function](#the-sigmoid-function)
5. [How Logistic Regression Works](#how-logistic-regression-works)
   - [Linear Combination](#step-1-linear-combination)
   - [Sigmoid Activation](#step-2-sigmoid-activation)
   - [Odds and Log-Odds](#step-3-odds-and-log-odds)
   - [Likelihood Function](#step-4-likelihood-function)
   - [Gradient of Log-Likelihood](#step-5-gradient-of-log-likelihood)
6. [Key Terminologies](#key-terminologies)
7. [Python Implementation](#python-implementation)
   - [Binomial Logistic Regression](#1-binomial-logistic-regression)
   - [Multinomial Logistic Regression](#2-multinomial-logistic-regression)
8. [Model Evaluation Metrics](#model-evaluation-metrics)
9. [Linear vs Logistic Regression](#linear-vs-logistic-regression)
10. [Quick Reference](#quick-reference)

---

## What is Logistic Regression?

**Logistic Regression** is a **supervised machine learning algorithm** primarily used for **classification problems**. Unlike linear regression (which predicts continuous values), logistic regression predicts the **probability** that a given input belongs to a specific class.

### Key Characteristics

- Used for **binary classification** (and extended to multi-class problems)
- Output is a **probability value between 0 and 1**
- Utilises the **sigmoid function** to map any real-valued input to the (0, 1) range
- Decision is made based on a **threshold** (commonly 0.5):
  - Probability ≥ 0.5 → Class 1
  - Probability < 0.5 → Class 0

### Common Use Cases

| Domain | Example |
|---|---|
| Healthcare | Disease diagnosis (cancer / no cancer) |
| Finance | Loan default prediction (yes / no) |
| Email | Spam detection (spam / not spam) |
| Marketing | Customer churn prediction |
| Computer Vision | Image classification (cat / dog) |

---

## Types of Logistic Regression

Logistic Regression is classified into **three main types** based on the nature of the dependent (target) variable:

---

### 1. 🔵 Binomial Logistic Regression

**Definition:** Used when the dependent variable has only **two possible categories**.

**Activation Function:** Sigmoid function

**Examples:**
- Yes / No
- Pass / Fail
- 0 / 1
- Spam / Not Spam

**Formula:**

```
P(y=1) = σ(z) = 1 / (1 + e^(-z))
P(y=0) = 1 - σ(z)
```

**Best For:** Binary classification tasks — the most common form of logistic regression.

---

### 2. 🟠 Multinomial Logistic Regression

**Definition:** Used when the dependent variable has **three or more unordered categories** (no natural ranking between them).

**Activation Function:** Softmax function

**Examples:**
- Classifying animals: Cat / Dog / Sheep
- Sentiment: Positive / Negative / Neutral
- Disease type: Type A / Type B / Type C

**Softmax Formula:**

```
softmax(z_i) = e^(z_i) / Σ e^(z_j)   for j = 1 to K

P(Y=c | X=x) = e^(w_c · x + b_c) / Σ e^(w_k · x + b_k)   for k = 1 to K
```

Where **K** is the total number of classes.

**Best For:** Multi-class classification where classes have no inherent order.

---

### 3. 🟢 Ordinal Logistic Regression

**Definition:** Used when the dependent variable has **three or more categories with a natural order or ranking**.

**Examples:**
- Ratings: Low / Medium / High
- Customer satisfaction: Poor / Average / Good / Excellent
- Education level: High School / Undergraduate / Postgraduate

**Key Difference from Multinomial:** This type explicitly **accounts for the order** of categories during modelling.

**Best For:** Situations where the relative ranking between outcomes matters.

---

### Type Comparison Summary

| Feature | Binomial | Multinomial | Ordinal |
|---|---|---|---|
| Number of Classes | 2 | 3+ | 3+ |
| Class Order | N/A | Unordered | Ordered |
| Activation Function | Sigmoid | Softmax | Cumulative logit |
| Example | Pass/Fail | Cat/Dog/Bird | Low/Med/High |

---

## Assumptions of Logistic Regression

For logistic regression to produce valid and reliable results, several assumptions must be satisfied:

| # | Assumption | Description |
|---|---|---|
| 1 | **Independent Observations** | Each data point must be independent — no correlation or dependence between samples |
| 2 | **Binary Dependent Variable** | The target variable must be binary (for standard logistic regression); softmax is used for multiple classes |
| 3 | **Linearity in Log-Odds** | A linear relationship must exist between the independent variables and the **log-odds** of the dependent variable |
| 4 | **No Outliers** | Extreme outliers can distort the estimation of regression coefficients |
| 5 | **Large Sample Size** | Requires a sufficiently large dataset to produce stable and reliable parameter estimates |

---

## The Sigmoid Function

The **sigmoid function** (also called the logistic function) is the core mathematical tool in logistic regression. It maps any real-valued number into the range **(0, 1)**.

### Formula

```
σ(z) = 1 / (1 + e^(-z))
```

### Properties

| Property | Description |
|---|---|
| Output Range | Always between 0 and 1 |
| Shape | S-shaped curve (sigmoid curve) |
| σ(z) → 1 | As z → +∞ |
| σ(z) → 0 | As z → −∞ |
| σ(0) = 0.5 | At z = 0, output is exactly 0.5 |

### Decision Threshold

```
If σ(z) ≥ 0.5  →  Predict Class 1
If σ(z) < 0.5  →  Predict Class 0
```

---

## How Logistic Regression Works

### Step 1: Linear Combination

Logistic regression first computes a linear combination of the input features:

```
z = w · X + b
  = (w₁x₁ + w₂x₂ + ... + wₙxₙ) + b
```

Where:
- `X` = input feature matrix
- `w` = weights / coefficients
- `b` = bias / intercept term
- `z` = raw linear output (can be any real number)

---

### Step 2: Sigmoid Activation

The linear output `z` is passed through the sigmoid function to produce a probability:

```
σ(z) = 1 / (1 + e^(-z))
```

This maps `z` to a probability value in (0, 1).

---

### Step 3: Odds and Log-Odds

Logistic regression models the **odds** of an event occurring:

```
Odds = P(x) / (1 - P(x)) = e^z
```

Taking the natural logarithm of the odds gives the **log-odds** (also called the **logit**):

```
log[ P(x) / (1 - P(x)) ] = z = w · X + b
```

Solving for P(x):

```
P(x; b, w) = e^(w·X + b) / (1 + e^(w·X + b))
           = 1 / (1 + e^(-(w·X + b)))
```

This is the final logistic regression equation — the probability of the input belonging to Class 1.

---

### Step 4: Likelihood Function

The model is trained by **maximising the likelihood** of observing the training data. The likelihood function is:

```
L(b, w) = ∏ P(xᵢ)^yᵢ · (1 - P(xᵢ))^(1 - yᵢ)
```

Taking the natural log gives the **log-likelihood**:

```
log L(b, w) = Σ [ yᵢ · log P(xᵢ) + (1 - yᵢ) · log(1 - P(xᵢ)) ]
```

This is also known as **binary cross-entropy loss** (negated).

---

### Step 5: Gradient of Log-Likelihood

To find the optimal weights `w` and bias `b`, the model uses **gradient ascent** on the log-likelihood. The gradient with respect to weight `wⱼ` is:

```
∂J / ∂wⱼ = Σ (yᵢ - P(xᵢ; b, w)) · xᵢⱼ
```

This update rule is used iteratively (via **gradient descent** on the loss) to minimise the prediction error.

---

## Key Terminologies

| Term | Definition |
|---|---|
| **Independent Variables** | Input features / predictor variables used to make predictions |
| **Dependent Variable** | The target / output variable to be predicted (categorical in logistic regression) |
| **Logistic Function (Sigmoid)** | Transforms continuous input into a probability between 0 and 1 |
| **Odds** | Ratio of the probability of an event happening to it not happening |
| **Log-Odds (Logit)** | Natural logarithm of the odds; modelled as a linear combination of input features |
| **Coefficient (Weight)** | Parameters estimated by the model; indicate the strength and direction of feature influence |
| **Intercept (Bias)** | Constant term representing log-odds when all independent variables equal zero |
| **MLE** | Maximum Likelihood Estimation — method used to estimate model coefficients |
| **Threshold** | Cut-off value (default 0.5) used to assign class labels from probabilities |

---

## Python Implementation

### 1. Binomial Logistic Regression

Uses the **Breast Cancer dataset** to classify tumours as malignant or benign.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# Split into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=23
)

# Train the model
clf = LogisticRegression(max_iter=10000, random_state=0)
clf.fit(X_train, y_train)

# Evaluate
acc = accuracy_score(y_test, clf.predict(X_test)) * 100
print(f"Logistic Regression model accuracy: {acc:.2f}%")
```

**Output:**
```
Logistic Regression model accuracy: 96.49%
```

---

### 2. Multinomial Logistic Regression

Uses the **Digits dataset** to classify handwritten digits (0–9) — 10 classes.

```python
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics

# Load dataset
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Split into train and test sets (60/40 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

# Train the model
reg = linear_model.LogisticRegression(max_iter=10000, random_state=0)
reg.fit(X_train, y_train)

# Evaluate
y_pred = reg.predict(X_test)
print(f"Logistic Regression model accuracy: {metrics.accuracy_score(y_test, y_pred) * 100:.2f}%")
```

**Output:**
```
Logistic Regression model accuracy: 96.66%
```

---

## Model Evaluation Metrics

After training a logistic regression model, these metrics are used to assess its performance:

### 1. Accuracy
Proportion of correctly classified instances out of total.

```
Accuracy = (True Positives + True Negatives) / Total
```

### 2. Precision
How accurate the positive predictions are.

```
Precision = True Positives / (True Positives + False Positives)
```

### 3. Recall (Sensitivity)
Proportion of actual positives correctly identified.

```
Recall = True Positives / (True Positives + False Negatives)
```

### 4. F1 Score
Harmonic mean of Precision and Recall — useful when class distribution is imbalanced.

```
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```

### 5. AUC-ROC
Area under the Receiver Operating Characteristic curve. Plots True Positive Rate vs. False Positive Rate across thresholds. Higher AUC = better classifier.

### 6. AUC-PR
Area under the Precision-Recall curve. Especially useful for imbalanced datasets where negative class dominates.

---

## Linear vs Logistic Regression

| Aspect | Linear Regression | Logistic Regression |
|---|---|---|
| **Purpose** | Predict continuous values | Predict categorical classes |
| **Problem Type** | Regression | Classification |
| **Output** | Continuous value (e.g., price, age) | Probability → categorical label (0 or 1) |
| **Model Curve** | Best-fit straight line | S-shaped sigmoid curve |
| **Estimation Method** | Least Squares Estimation | Maximum Likelihood Estimation (MLE) |
| **Output Example** | House price = $350,000 | Fraud = Yes (1) |
| **Linearity Required** | Yes — between features and output | Only between features and log-odds |
| **Collinearity** | Acceptable | Should be minimal |

---

## Quick Reference

```
Logistic Regression
│
├── Binomial          → 2 classes       → Sigmoid function
├── Multinomial       → 3+ classes      → Softmax function
└── Ordinal           → 3+ ordered      → Cumulative logit
│
Core Formula:   P(x) = 1 / (1 + e^(-(w·X + b)))
│
Training:       Maximise Log-Likelihood via Gradient Descent
│
Evaluation:     Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR
```

---

## 📚 References

- [GeeksforGeeks — Logistic Regression in Machine Learning](https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/)
- [Scikit-learn Logistic Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Sigmoid Function — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/derivative-of-the-sigmoid-function/)
- [Softmax Classifier — GeeksforGeeks](https://www.geeksforgeeks.org/deep-learning/what-is-softmax-classifier/)
- [AUC-ROC Curve — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/auc-roc-curve/)

---