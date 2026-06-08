# 📈 Linear Regression in Machine Learning

---

## 📌 Table of Contents

- [Overview](#-overview)
- [What is Linear Regression?](#-what-is-linear-regression)
- [Key Terminology](#-key-terminology)
- [Best Fit Line](#-best-fit-line)
  - [Goal of the Best-Fit Line](#1-goal-of-the-best-fit-line)
  - [Equation of Best-Fit Line](#2-equation-of-the-best-fit-line)
  - [Least Squares Method](#3-least-squares-method)
  - [Interpretation of Slope and Intercept](#4-interpretation-of-slope-and-intercept)
- [Hypothesis Function](#-hypothesis-function)
- [Assumptions of Linear Regression](#-assumptions-of-linear-regression)
- [Types of Linear Regression](#-types-of-linear-regression)
  - [Type 1 — Simple Linear Regression](#-type-1--simple-linear-regression)
  - [Type 2 — Multiple Linear Regression](#-type-2--multiple-linear-regression)
  - [Type 3 — Polynomial Regression](#-type-3--polynomial-regression)
  - [Type 4 — Ridge Regression (L2)](#-type-4--ridge-regression-l2)
  - [Type 5 — Lasso Regression (L1)](#-type-5--lasso-regression-l1)
  - [Type 6 — ElasticNet Regression](#-type-6--elasticnet-regression)
  - [Type 7 — Bayesian Linear Regression](#-type-7--bayesian-linear-regression)
  - [Type 8 — Quantile Regression](#-type-8--quantile-regression)
  - [Type 9 — Robust Regression](#-type-9--robust-regression)
  - [Type 10 — Stepwise Regression](#-type-10--stepwise-regression)
- [Cost Function](#-cost-function)
- [Gradient Descent](#-gradient-descent)
  - [Types of Gradient Descent](#-types-of-gradient-descent)
- [Full Python Implementation](#-full-python-implementation)
  - [Simple Linear Regression](#1-simple-linear-regression-implementation)
  - [Multiple Linear Regression](#2-multiple-linear-regression-implementation)
- [Evaluation Metrics](#-evaluation-metrics)
- [Regularisation Techniques — Summary](#-regularisation-techniques--summary)
- [Bias-Variance Tradeoff](#-bias-variance-tradeoff)
- [Advantages of Linear Regression](#-advantages-of-linear-regression)
- [Limitations of Linear Regression](#-limitations-of-linear-regression)
- [When to Use Linear Regression](#-when-to-use-linear-regression)
- [Types Comparison Table](#-types-comparison-table)
- [Decision Guide — Which Type to Use?](#-decision-guide--which-type-to-use)
- [Key Takeaways](#-key-takeaways)
- [Prerequisites](#-prerequisites)
- [Further Reading](#-further-reading)

---

## 🧭 Overview

Linear Regression is a **fundamental supervised learning algorithm** that models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting the best possible straight line through the data.

It is the **starting point for almost every ML practitioner** — simple enough to understand deeply, yet powerful enough to solve a wide range of real-world prediction problems.

```
         Salary (₹)
            │                         ●
            │                    ●
       5L   │              ● Best-fit line
            │         ●──────────────────
       3L   │    ●
            │●
            └──────────────────────────── Experience (years)
              1   2   3   4   5   6   7

Goal: Find the line that best explains how salary
      increases with years of experience.
```

---

## 🔍 What is Linear Regression?

Linear Regression is a fundamental supervised learning algorithm used to model the relationship between a **dependent variable** and one or more **independent variables**. It predicts continuous values by fitting a straight line that best represents the data.

- Assumes a **linear relationship** between input and output
- Uses a **best-fit line** to make predictions
- Commonly used in forecasting, trend analysis, and predictive modelling

**Real-world example:**

```
Problem: Predict a student's exam score based on hours studied.

Observation: As students study more hours, their scores go up.

  Independent variable (input)  → Hours studied  (X)
  Dependent variable  (output)  → Exam score     (Y)

As X increases, Y increases → linear positive relationship
```

---

## 📖 Key Terminology

| Term | Symbol | Description |
|---|---|---|
| **Dependent variable** | Y, ŷ | Target / output variable being predicted |
| **Independent variable** | X | Input / predictor / feature variable |
| **Intercept** | β₀ (theta_0) | Value of Y when all X = 0; where line crosses Y-axis |
| **Slope / Coefficient** | β₁ (theta_1) | Change in Y per unit increase in X |
| **Residual / Error** | εᵢ | Difference between actual and predicted: yᵢ − ŷᵢ |
| **Best-fit line** | ŷ = β₀ + β₁x | The line that minimises total prediction error |
| **Cost function (J)** | MSE | Measures how far predictions are from actual values |
| **Gradient Descent** | — | Optimisation algorithm to minimise cost function |
| **R² Score** | R² | Proportion of variance in Y explained by the model |
| **Regularisation** | λ (lambda) | Penalty term added to prevent overfitting |

---

## 📐 Best Fit Line

### 1. Goal of the Best-Fit Line

The goal of linear regression is to find a straight line that **minimises the error** (the difference) between the observed data points and the predicted values. This line helps predict the dependent variable for new, unseen data.

```
Data points vs. best-fit line:

   Y │ ●  ← actual point
     │  \  ← residual (error)
     │   ×  ← predicted point on line
     │    \_____________________
     │                          Line (ŷ = β₀ + β₁x)
     └─────────────────────── X

Best-fit line minimises: Σ(yᵢ − ŷᵢ)²   ← Sum of Squared Residuals
```

---

### 2. Equation of the Best-Fit Line

For **Simple Linear Regression** (one independent variable):

```
ŷ = mx + b

Where:
  ŷ  = predicted value (dependent variable)
  x  = input (independent variable)
  m  = slope  → how much ŷ changes when x changes by 1
  b  = intercept → value of ŷ when x = 0

Also written as:
  ŷ = β₀ + β₁x

  β₀ = intercept
  β₁ = slope (coefficient)
```

For **Multiple Linear Regression** (multiple independent variables):

```
ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + ... + βₙxₙ

Where:
  x₁, x₂, ..., xₙ  = independent variables (features)
  β₁, β₂, ..., βₙ  = coefficients (one per feature)
  β₀               = intercept
```

---

### 3. Least Squares Method

The **Ordinary Least Squares (OLS)** method finds the optimal β values by minimising the **Sum of Squared Residuals (SSR)**:

```
Residual for each point:
  εᵢ = yᵢ − ŷᵢ = yᵢ − (β₀ + β₁xᵢ)

Objective — minimise SSR:
  SSR = Σᵢ(yᵢ − ŷᵢ)²
      = Σᵢ(yᵢ − β₀ − β₁xᵢ)²

Closed-form OLS solution:

         Σ(xᵢ − x̄)(yᵢ − ȳ)
  β₁ = ─────────────────────
              Σ(xᵢ − x̄)²

  β₀ = ȳ − β₁x̄

Where x̄ = mean of X, ȳ = mean of Y
```

```python
import numpy as np

# OLS from scratch
def ols_coefficients(X, y):
    x_mean = np.mean(X)
    y_mean = np.mean(y)

    beta_1 = np.sum((X - x_mean) * (y - y_mean)) / np.sum((X - x_mean) ** 2)
    beta_0 = y_mean - beta_1 * x_mean

    return beta_0, beta_1

X = np.array([1, 2, 3, 4, 5, 6, 7])
y = np.array([1.5, 3.0, 4.5, 5.5, 7.0, 8.5, 9.5])

b0, b1 = ols_coefficients(X, y)
print(f"Intercept (β₀): {b0:.4f}")
print(f"Slope     (β₁): {b1:.4f}")
print(f"Equation : ŷ = {b0:.4f} + {b1:.4f}x")
```

---

### 4. Interpretation of Slope and Intercept

```
Example: ŷ = 1.2 + 5.0x
  (predicting salary ₹L from years of experience)

  Intercept (β₀ = 1.2):
    → A person with 0 years of experience earns ₹1.2L
    → Starting/base salary

  Slope (β₁ = 5.0):
    → For every 1 additional year of experience,
      salary increases by ₹5.0L
    → Rate of change
```

| Slope value | Meaning |
|---|---|
| **β₁ > 0** | Positive relationship — Y increases as X increases |
| **β₁ < 0** | Negative relationship — Y decreases as X increases |
| **β₁ = 0** | No linear relationship between X and Y |
| **\|β₁\| large** | Steep slope — X has strong influence on Y |
| **\|β₁\| small** | Flat slope — X has weak influence on Y |

---

## 🔧 Hypothesis Function

In linear regression, the **hypothesis function** is the equation used to make predictions about the dependent variable based on independent variables.

**Simple Linear Regression (1 feature):**
```
h(x) = β₀ + β₁x

Where:
  h(x) or ŷ = predicted output
  x          = single input feature
  β₀         = intercept (value of ŷ when x = 0)
  β₁         = slope (rate of change of ŷ per unit of x)
```

**Multiple Linear Regression (k features):**
```
h(x₁, x₂, ..., xₖ) = β₀ + β₁x₁ + β₂x₂ + ... + βₖxₖ

In matrix notation:
  ŷ = Xβ

Where:
  X  = feature matrix (n_samples × n_features)
  β  = coefficient vector (n_features × 1)
  ŷ  = prediction vector (n_samples × 1)
```

---

## ✅ Assumptions of Linear Regression

Linear regression relies on **7 key assumptions**. Violating them can make results unreliable.

---

### 1. Linearity

The relationship between each independent variable X and the dependent variable Y must be **linear** — a straight-line relationship.

```
    Valid (linear):       Not valid (non-linear):
    Y│   /                Y│    ●●
     │  /                  │  ●    ●
     │ /                   │●        ●
     └──── X               └──────── X

    Residual plot check: residuals should scatter randomly
    around zero with no curved pattern
```

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Check linearity via scatter plot of each feature vs target
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(['feature1', 'feature2', 'feature3']):
    axes[i].scatter(df[col], df['target'], alpha=0.5)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('target')
    axes[i].set_title(f'{col} vs target')
plt.tight_layout()
plt.show()
```

---

### 2. Independence of Errors

The residuals (errors) for each observation should be **independent** of each other — knowing one error gives no information about another.

```
Valid:                    Not valid (autocorrelation):
Residuals │ ● ●  ●         Residuals │ ●●●●
          │●  ●  ●●                   │       ●●●●
          │  ●●●                      │           ●●●●
          └──────── x                └────────────── x
          (scattered randomly)       (clear pattern = problem)
```

```python
from statsmodels.stats.stattools import durbin_watson

# Durbin-Watson test for autocorrelation
# Value ≈ 2 → no autocorrelation | <1 or >3 → problem
residuals = y_test - y_pred
dw_stat = durbin_watson(residuals)
print(f"Durbin-Watson: {dw_stat:.4f}")
# Near 2.0 = good, <1.5 or >2.5 = autocorrelation present
```

---

### 3. Homoscedasticity (Constant Variance of Errors)

The spread of residuals should remain **constant** across all values of X.

```
Valid (homoscedastic):    Not valid (heteroscedastic):
Residuals │ ● ● ● ●        Residuals │         ●●●
          │● ● ● ●●                  │      ●●●
          │ ● ● ● ●                  │   ●●
          └──────── ŷ                └────────── ŷ
          (equal spread)             (fan shape = problem)
```

```python
# Check with residual vs fitted plot
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.5, color='steelblue')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values (ŷ)')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted — Check for Homoscedasticity')
plt.show()
```

---

### 4. Normality of Errors

The residuals should follow a **normal (bell-curve) distribution** with mean zero.

```
Valid (normal errors):
        ████
      ██████████
    ██████████████
  ████████████████████
─────────────────────── residuals
   -3σ  -2σ  0  +2σ  +3σ
```

```python
from scipy import stats
import matplotlib.pyplot as plt

# Q-Q Plot — points should fall on diagonal line
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Histogram of residuals
ax1.hist(residuals, bins=20, color='steelblue', edgecolor='white')
ax1.set_title('Histogram of Residuals')
ax1.set_xlabel('Residuals')

# Q-Q Plot
stats.probplot(residuals, dist='norm', plot=ax2)
ax2.set_title('Q-Q Plot')
plt.tight_layout()
plt.show()

# Shapiro-Wilk test for normality
stat, p = stats.shapiro(residuals)
print(f"Shapiro-Wilk p-value: {p:.4f}")
print("Normal" if p > 0.05 else "Not Normal — check distribution")
```

---

### 5. No Multicollinearity (Multiple Regression)

Independent variables should **not be highly correlated** with each other. High correlation makes coefficients unstable and hard to interpret.

```python
import pandas as pd
import seaborn as sns

# Correlation heatmap — values close to ±1 between features = problem
corr_matrix = df[feature_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap — Check for Multicollinearity')
plt.show()

# VIF (Variance Inflation Factor)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

X_with_const = sm.add_constant(X)
vif_data = pd.DataFrame({
    'Feature': X.columns,
    'VIF'    : [variance_inflation_factor(X_with_const.values, i+1)
                for i in range(len(X.columns))]
})
print(vif_data)
# VIF > 10 → serious multicollinearity problem
# VIF > 5  → moderate concern
```

---

### 6. No Autocorrelation

Errors should not show **repeating patterns over time** (especially important for time-series data).

---

### 7. Additivity

The total effect on Y is the **sum of individual effects** from each X. No interaction terms are assumed in simple linear regression.

---

### Assumptions Summary Table

| Assumption | Check Method | Fix if Violated |
|---|---|---|
| **Linearity** | Scatter plot, residual plot | Transform features (log, sqrt) |
| **Independence** | Durbin-Watson test | Use time-series models |
| **Homoscedasticity** | Residuals vs fitted plot | Log-transform Y; use WLS |
| **Normality of errors** | Q-Q plot, Shapiro-Wilk | Transform Y; use robust regression |
| **No multicollinearity** | Correlation matrix, VIF | Remove/combine features; use Ridge |
| **No autocorrelation** | Durbin-Watson, ACF plot | Add lag features; use ARIMA |
| **Additivity** | Domain knowledge | Add interaction terms explicitly |

---

## 📚 Types of Linear Regression

---

## 📌 Type 1 — Simple Linear Regression

**The most basic form of linear regression** with exactly **one independent variable** predicting one dependent variable.

```
ŷ = β₀ + β₁x

Use case: Predict salary (y) from years of experience (x)
```

```
     Salary
        │              ●
        │          ●●
        │       ●          Best-fit line: ŷ = β₀ + β₁x
        │   ● ●
        │●
        └──────────────────── Experience
```

### Python Implementation

```python
from sklearn.linear_model import LinearRegression
```

| ✅ Pros | ❌ Cons |
|---|---|
| Extremely simple and interpretable | Only works for one input feature |
| Fast to train and predict | Cannot model complex patterns |
| Easy to visualise | Assumes perfect linear relationship |

**Best for:** Sales vs. advertising spend, temperature vs. ice cream sales, height vs. weight prediction

---

## 📌 Type 2 — Multiple Linear Regression

Extends Simple LR to handle **two or more independent variables** simultaneously.

```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

Use case: Predict house price using size, bedrooms, location, age
```

```
House Price = β₀ + β₁(Size) + β₂(Bedrooms) + β₃(Location_Score) + β₄(Age)

            Features (X)               Target (y)
  ┌──────┬──────────┬──────────────┐    ┌───────────┐
  │ Size │ Bedrooms │ Loc_Score    │ ─► │ Price ($) │
  │  800 │     2    │    7.5       │    │  320,000  │
  │ 1200 │     3    │    8.2       │    │  480,000  │
  │  950 │     2    │    6.0       │    │  350,000  │
  └──────┴──────────┴──────────────┘    └───────────┘
```

### Python Implementation

```python
from sklearn.linear_model import LinearRegression
```

| ✅ Pros | ❌ Cons |
|---|---|
| Handles multiple features | Assumes linear relationship with each feature |
| Captures combined effects of features | Risk of multicollinearity |
| Interpretable per-feature coefficients | Performance degrades with irrelevant features |

**Best for:** Real estate pricing, financial forecasting, crop yield estimation, e-commerce sales analysis

---

## 📌 Type 3 — Polynomial Regression

Extends linear regression to capture **non-linear (curved) relationships** by adding polynomial terms (x², x³, ...) as new features.

```
ŷ = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ

Despite having x², x³ terms — it is still called LINEAR regression
because it is linear with respect to the COEFFICIENTS (β values).
```

```
     Simple (degree=1):        Polynomial (degree=2):
     Y │    /                  Y │    ╭────
       │   /                     │   ╱
       │  /                      │  ╱
       │ /                       │ ╱
       └───── X                  └────── X
       (straight line)           (parabola — degree 2)
```

### Python Implementation

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
```

> ⚠️ **Degree warning:**
> - **Degree 1** = linear (may underfit)
> - **Degree 2–3** = good balance for most curved data
> - **Degree 5+** = high risk of overfitting — always cross-validate

| ✅ Pros | ❌ Cons |
|---|---|
| Captures non-linear patterns | High degrees overfit easily |
| Still uses linear model (interpretable β) | Hard to interpret high-degree coefficients |
| Flexible — any curve shape possible | Extrapolation is unreliable beyond data range |

**Best for:** Growth curves, biological phenomena, physics simulations, seasonal patterns

---

## 📌 Type 4 — Ridge Regression (L2)

Ridge Regression adds an **L2 regularisation penalty** (sum of squared coefficients) to the cost function to prevent overfitting and handle multicollinearity.

```
Ridge Cost Function:

  J(β) = Σ(yᵢ − ŷᵢ)² + λ · Σβⱼ²
          ─────────────   ────────
          OLS loss term   L2 penalty

  λ (alpha) = regularisation strength
    λ = 0  → same as OLS (no regularisation)
    λ → ∞  → all coefficients shrink toward zero (but never exactly zero)
```

```
Effect on coefficients:
  Without Ridge: β = [25.3, -18.7, 42.1, -31.5]   ← large, unstable
  With Ridge:    β = [ 8.2,  -6.1, 14.3,  -9.8]   ← smaller, stable
  All shrunk toward 0, but none reach exactly 0
```

### Python Implementation

```python
from sklearn.linear_model import Ridge, RidgeCV
```

| ✅ Pros | ❌ Cons |
|---|---|
| Reduces overfitting | Does not perform feature selection |
| Stabilises coefficients with multicollinearity | Adds hyperparameter (λ) to tune |
| All features are kept (useful if all matter) | Less interpretable than OLS |

**Best for:** Datasets with many correlated features, genomics, finance with correlated indicators

---

## 📌 Type 5 — Lasso Regression (L1)

Lasso (**L**east **A**bsolute **S**hrinkage and **S**election **O**perator) adds an **L1 penalty** (sum of absolute coefficients). Uniquely, Lasso can shrink coefficients to **exactly zero** — performing automatic **feature selection**.

```
Lasso Cost Function:

  J(β) = Σ(yᵢ − ŷᵢ)² + λ · Σ|βⱼ|
          ─────────────   ──────────
          OLS loss term   L1 penalty (absolute values)

Key difference from Ridge:
  L1 penalty can zero out coefficients completely → feature selection!
```

```
Effect on coefficients:
  Without Lasso: β = [25.3, -18.7, 42.1, -31.5, 0.8, -0.3]
  With Lasso:    β = [12.1,  -8.4, 18.7,  -0.0, 0.0,  0.0]
                                           ↑   eliminated features!
```

### Python Implementation

```python
from sklearn.linear_model import Lasso, LassoCV
```

| ✅ Pros | ❌ Cons |
|---|---|
| Automatic feature selection (zeros out irrelevant) | May arbitrarily drop one of several correlated features |
| Produces sparse, interpretable models | Not ideal when all features are equally relevant |
| Great for high-dimensional data | Less stable than Ridge for multicollinear data |

**Best for:** High-dimensional datasets, genomics, text analysis, sparse data where few features matter

---

## 📌 Type 6 — ElasticNet Regression

ElasticNet is a **hybrid regularisation** technique that **combines both L1 (Lasso) and L2 (Ridge) penalties**. It inherits the feature selection of Lasso and the stability of Ridge.

```
ElasticNet Cost Function:

  J(β) = Σ(yᵢ − ŷᵢ)² + λ₁·Σ|βⱼ| + λ₂·Σβⱼ²
          ─────────────  ──────────  ─────────
          OLS loss       L1 penalty  L2 penalty
                         (Lasso)     (Ridge)

l1_ratio parameter:
  l1_ratio = 1.0 → pure Lasso
  l1_ratio = 0.0 → pure Ridge
  l1_ratio = 0.5 → equal mix (default in sklearn)
```

### Python Implementation

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV
```

| Model | Penalty | Feature Selection | Grouped Features | Best For |
|---|---|---|---|---|
| **Ridge** | L2 (squared) | ❌ No | Shrinks together | Multicollinear, all features relevant |
| **Lasso** | L1 (absolute) | ✅ Yes (zeros out) | Picks one randomly | Sparse, high-dimensional |
| **ElasticNet** | L1 + L2 | ✅ Partial | Groups and selects | Correlated + sparse features |

---

## 📌 Type 7 — Bayesian Linear Regression

A **probabilistic approach** to linear regression that treats coefficients as **random variables with probability distributions** rather than fixed values. Uses Bayes' theorem to update beliefs about parameters as new data arrives.

```
Classical LR:
  Find single "best" β values
  ŷ = β₀ + β₁x  (point estimate)

Bayesian LR:
  Find probability distribution over β values
  P(β | X, y) ∝ P(y | X, β) × P(β)
  ─────────────  ──────────────  ────
  posterior      likelihood      prior
  (updated belief) (data fit)   (initial belief)

Output: not just ŷ, but a distribution with uncertainty bounds
```

### Python Implementation

```python
from sklearn.linear_model import BayesianRidge
```

| ✅ Pros | ❌ Cons |
|---|---|
| Provides uncertainty estimates | Computationally more expensive |
| Robust to overfitting (prior acts as regularisation) | Requires specifying prior distributions |
| Works well with small datasets | Harder to interpret than classical LR |
| Naturally handles missing data | Results depend on prior choice |

**Best for:** Medical decision-making, finance risk models, small datasets, scenarios requiring confidence intervals

---

## 📌 Type 8 — Quantile Regression

Instead of predicting the **conditional mean** (like OLS), Quantile Regression predicts any **conditional quantile** of Y — the median (q=0.5), 90th percentile (q=0.9), etc.

```
OLS Linear Regression:     predicts the MEAN of y given x
Quantile Regression:       predicts any PERCENTILE of y given x

Example: Predicting salaries
  OLS result:         ŷ = $65,000  (average salary)
  Quantile q=0.90:    ŷ = $95,000  (90th percentile salary)
  Quantile q=0.10:    ŷ = $42,000  (10th percentile salary)

Useful when:
  - Data is skewed
  - You care about extremes (risks)
  - Outliers are present
```

### Python Implementation

```python
# pip install statsmodels
import statsmodels.formula.api as smf
```

**Best for:** Income prediction, insurance premium calculation, risk quantification, skewed or heteroscedastic data

---

## 📌 Type 9 — Robust Regression

Standard linear regression is sensitive to **outliers**. Robust Regression methods use **loss functions that down-weight the influence of outliers**, making the model more reliable when extreme values are present.

```
OLS Loss:      L(ε) = ε²          ← small errors: small loss
                                    large errors: VERY large loss (squared!)

Huber Loss:    L(ε) = ε²           if |ε| ≤ δ  (quadratic for small errors)
                     δ(|ε| - δ/2)  if |ε| > δ  (linear for large errors)

Effect: outliers penalised less → they have less influence on the model
```

### Python Implementation

```python
from sklearn.linear_model import HuberRegressor, TheilSenRegressor, RANSACRegressor
---

## 📌 Type 10 — Stepwise Regression

An automated process of **iteratively adding or removing features** based on statistical significance (p-values or AIC/BIC) to find the optimal subset of predictors.

```
Three approaches:
  Forward Selection:   Start empty → add features one by one
                       (keep if p-value < threshold)

  Backward Elimination: Start with all → remove features one by one
                        (drop if p-value > threshold)

  Bidirectional:       Combination of both — add and remove at each step
```

### Python Implementation

```python
import statsmodels.api as sm
import pandas as pd
import numpy as np

def backward_elimination(X, y, significance=0.05):
    """
    Backward elimination: remove features with p-value > significance
    until all remaining features are statistically significant.
    """
    features = list(X.columns)
    while True:
        X_const = sm.add_constant(X[features])
        model   = sm.OLS(y, X_const).fit()
        p_values = model.pvalues.drop('const')
        max_p    = p_values.max()

        if max_p > significance:
            worst_feature = p_values.idxmax()
            features.remove(worst_feature)
            print(f"Removed '{worst_feature}' (p={max_p:.4f})")
        else:
            break

    print(f"\nFinal features: {features}")
    final_model = sm.OLS(y, sm.add_constant(X[features])).fit()
    print(final_model.summary())
    return final_model, features


# Run backward elimination
final_model, selected_features = backward_elimination(X, y)
```

**Best for:** Exploratory analysis, building parsimonious models, identifying the key predictors

---

## 💰 Cost Function

The **cost function J(β)** measures how far the model's predictions are from the actual values. The goal of training is to find β values that **minimise J**.

```
Mean Squared Error (MSE) — most common cost function:

        1   n
J(β) = ─── Σ (ŷᵢ − yᵢ)²
        n  i=1

Where:
  ŷᵢ = β₀ + β₁xᵢ  (predicted value)
  yᵢ               (actual value)
  n                (number of samples)

Why squared?
  → Penalises large errors more heavily
  → Always positive (no cancellation)
  → Differentiable (needed for gradient descent)
```

```python
import numpy as np

def compute_cost(X, y, beta_0, beta_1):
    """Compute MSE cost function for given parameters."""
    n      = len(y)
    y_pred = beta_0 + beta_1 * X
    cost   = (1 / n) * np.sum((y_pred - y) ** 2)
    return cost

# Example
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

print(f"Cost (β₀=0, β₁=1): {compute_cost(X, y, 0, 1):.4f}")
print(f"Cost (β₀=1, β₁=1): {compute_cost(X, y, 1, 1):.4f}")
print(f"Cost (β₀=0, β₁=0): {compute_cost(X, y, 0, 0):.4f}")
```

---

## 📉 Gradient Descent

**Gradient Descent** is the optimisation algorithm that finds the β values minimising the cost function by taking iterative steps in the direction of steepest descent.

```
Algorithm:
  1. Start with random β₀, β₁
  2. Calculate cost J(β)
  3. Compute partial derivatives (gradients):
       ∂J/∂β₀ and ∂J/∂β₁
  4. Update parameters:
       β₀ := β₀ − α · ∂J/∂β₀
       β₁ := β₁ − α · ∂J/∂β₁
  5. Repeat until convergence (cost stops decreasing)

  α = learning rate  (step size)
```

### Gradient Descent Update Rules

```
∂J/∂β₀ = (2/n) · Σ(ŷᵢ − yᵢ)

∂J/∂β₁ = (2/n) · Σ(ŷᵢ − yᵢ) · xᵢ

Update:
  β₀ := β₀ − α · (2/n) · Σ(ŷᵢ − yᵢ)
  β₁ := β₁ − α · (2/n) · Σ(ŷᵢ − yᵢ) · xᵢ
```

### Python — Gradient Descent from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    n    = len(y)
    b0   = 0.0    # intercept
    b1   = 0.0    # slope
    cost_history = []

    for i in range(n_iterations):
        y_pred = b0 + b1 * X
        error  = y_pred - y

        # Gradients
        db0 = (2/n) * np.sum(error)
        db1 = (2/n) * np.sum(error * X)

        # Update
        b0 -= learning_rate * db0
        b1 -= learning_rate * db1

        # Track cost
        cost = (1/n) * np.sum(error ** 2)
        cost_history.append(cost)

    return b0, b1, cost_history

# ── Example ───────────────────────────────────────────────────
np.random.seed(42)
X = np.random.rand(100) * 10
y = 2.5 * X + 1.5 + np.random.randn(100) * 2.0

b0, b1, history = gradient_descent(X, y,
                                    learning_rate=0.005,
                                    n_iterations=2000)

print(f"Learned β₀ (intercept): {b0:.4f}")
print(f"Learned β₁ (slope)    : {b1:.4f}")
print(f"Final cost (MSE)      : {history[-1]:.6f}")

# Plot cost convergence
plt.figure(figsize=(9, 4))
plt.plot(history, color='steelblue', linewidth=1.5)
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Gradient Descent — Cost Convergence')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

### 📌 Types of Gradient Descent

| Type | Data per Update | Speed | Noise | Best For |
|---|---|---|---|---|
| **Batch GD** | Entire dataset | Slow | Low | Small datasets — smooth convergence |
| **Stochastic GD (SGD)** | 1 random sample | Very fast | High | Very large datasets |
| **Mini-Batch GD** | Small batch (32–256) | Fast | Medium | Standard choice for most problems |

```python
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Mini-batch SGD (via scikit-learn)
sgd = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  SGDRegressor(
                   loss='squared_error',    # OLS loss
                   max_iter=1000,
                   tol=1e-4,
                   learning_rate='invscaling',
                   eta0=0.01,
                   random_state=42
               ))
])
sgd.fit(X_train, y_train)
print(f"SGD R²: {sgd.score(X_test, y_test):.4f}")
```

---
---

## 📊 Evaluation Metrics

| Metric | Formula | Range | Interpretation |
|---|---|---|---|
| **MAE** | Σ\|yᵢ−ŷᵢ\|/n | 0 to ∞ | Average absolute error — same unit as y |
| **MSE** | Σ(yᵢ−ŷᵢ)²/n | 0 to ∞ | Penalises large errors more (squared) |
| **RMSE** | √MSE | 0 to ∞ | Interpretable — same unit as y; sensitive to outliers |
| **R²** | 1 − SS_res/SS_tot | -∞ to 1 | 1 = perfect; 0 = no better than mean; <0 = worse than mean |
| **Adjusted R²** | 1 − (1−R²)(n−1)/(n−k−1) | -∞ to 1 | Penalises for adding irrelevant features |

```python
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
import numpy as np

def regression_metrics(y_true, y_pred, n_features, label="Model"):
    """Print comprehensive regression evaluation metrics."""
    n   = len(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    print(f"\n── {label} Evaluation ─────────────────────")
    print(f"  MAE         : {mae:.4f}")
    print(f"  MSE         : {mse:.4f}")
    print(f"  RMSE        : {rmse:.4f}")
    print(f"  R² Score    : {r2:.4f}  ({r2*100:.1f}% variance explained)")
    print(f"  Adjusted R² : {adj_r2:.4f}")

regression_metrics(y_test, y_pred, n_features=X.shape[1], label="Linear Regression")
```

---

## 🛡️ Regularisation Techniques — Summary

| Technique | Penalty | Formula | Feature Selection | Handles Multicollinearity | sklearn Class |
|---|---|---|---|---|---|
| **OLS (Plain LR)** | None | Σ(y−ŷ)² | ❌ | ❌ | `LinearRegression` |
| **Ridge (L2)** | L2 | Σ(y−ŷ)² + λΣβ² | ❌ | ✅ | `Ridge` |
| **Lasso (L1)** | L1 | Σ(y−ŷ)² + λΣ\|β\| | ✅ | Partial | `Lasso` |
| **ElasticNet** | L1+L2 | Σ(y−ŷ)² + λ₁Σ\|β\|+ λ₂Σβ² | ✅ Partial | ✅ | `ElasticNet` |

---

## ⚖️ Bias-Variance Tradeoff

```
Underfitting (High Bias):          Overfitting (High Variance):
  Model is too simple               Model is too complex

  True curve:  ╭─────╮             True curve:  ╭─────╮
  Model:  ───────────── (flat)      Model: ╭─╮╭─╯╰─╮╰─╮

  Training error: High              Training error: Very low
  Test error:     High              Test error:     High
  Fix: More features/complexity     Fix: Regularisation, simpler model

Sweet spot (Just right):
  Model captures the true pattern
  Training ≈ Test error
  Fix: Cross-validate, tune regularisation λ
```

```python
from sklearn.model_selection import cross_val_score
import numpy as np

# Cross-validation to estimate generalisation
cv_scores = cross_val_score(model, X, y, cv=10, scoring='r2')
print(f"CV R² scores: {cv_scores.round(4)}")
print(f"Mean R²     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

---

## ✅ Advantages of Linear Regression

| Advantage | Description |
|---|---|
| **Simple and Interpretable** | Coefficients directly show the effect of each feature on the target |
| **Computationally Efficient** | Trains fast — OLS has a closed-form solution; works on large datasets |
| **Good Baseline** | Always worth trying first before complex models |
| **Probabilistic Output** | Standard errors and confidence intervals available |
| **Wide Availability** | Implemented in every ML library and statistical software |
| **Feature Importance** | Coefficient magnitudes indicate feature importance |
| **Foundation for Advanced Models** | Logistic regression, neural networks build on these concepts |

---

## ❌ Limitations of Linear Regression

| Limitation | Description | Fix |
|---|---|---|
| **Assumes linearity** | Cannot model curves, interactions | Polynomial, Decision Tree |
| **Sensitive to outliers** | Extreme values pull the line | Robust regression, remove outliers |
| **Multicollinearity** | Correlated features distort coefficients | Ridge, VIF analysis |
| **Requires feature engineering** | Raw data may need transformation | Log, sqrt, binning |
| **Overfitting with many features** | Too many parameters relative to data | Ridge, Lasso, cross-validation |
| **No automatic feature selection** | All features included by default | Lasso, Stepwise |
| **Limited for complex patterns** | Cannot rival deep learning on images/text | ANN, CNN, BERT |

---

## 🎯 When to Use Linear Regression

| ✅ Use Linear Regression When | ❌ Don't Use When |
|---|---|
| Relationship between X and Y is approximately linear | Clear non-linear relationship exists |
| You need an interpretable model | Black-box accuracy is the priority |
| Dataset is small to medium-sized | Input is images, audio, or raw text |
| Fast training and prediction needed | Many complex feature interactions |
| You need confidence intervals | Target is categorical (use logistic regression) |
| Serving as a baseline model | Data has many outliers (use robust regression) |

---

## 📊 Types Comparison Table

| Type | Features | Regularisation | Feature Selection | Handles Multicollinearity | Best Use Case |
|---|---|---|---|---|---|
| **Simple LR** | 1 | None | N/A | N/A | One predictor, teaching |
| **Multiple LR** | 2+ | None | No | No | Multi-feature prediction |
| **Polynomial LR** | 1+ (expanded) | None | No | No | Curved relationships |
| **Ridge (L2)** | 2+ | L2 | No | ✅ Yes | Correlated features |
| **Lasso (L1)** | 2+ | L1 | ✅ Yes | Partial | High-dimensional, sparse |
| **ElasticNet** | 2+ | L1+L2 | ✅ Partial | ✅ Yes | Correlated + sparse |
| **Bayesian LR** | 2+ | Prior | No | Partial | Uncertainty required |
| **Quantile LR** | 2+ | None | No | No | Skewed data, extremes |
| **Robust LR** | 2+ | None | No | No | Data with outliers |
| **Stepwise LR** | 2+ | None | ✅ Auto | No | Feature selection |

---

## 🗺️ Decision Guide — Which Type to Use?

```
START HERE: What is your problem?
│
├── Only ONE input feature?
│       └──► Simple Linear Regression
│
├── TWO OR MORE input features?
│       │
│       ├── Any risk of overfitting / too many features?
│       │       │
│       │       ├── Features are correlated (multicollinearity)?
│       │       │       └──► Ridge Regression (L2)
│       │       │
│       │       ├── Need automatic feature selection?
│       │       │       └──► Lasso Regression (L1)
│       │       │
│       │       └── Both correlated AND need selection?
│       │               └──► ElasticNet Regression
│       │
│       └── No overfitting concern?
│               └──► Multiple Linear Regression (OLS)
│
├── Is the relationship NON-LINEAR / CURVED?
│       └──► Polynomial Regression (degree 2 or 3)
│
├── DATA HAS OUTLIERS?
│       └──► Robust Regression (Huber or RANSAC)
│
├── NEED UNCERTAINTY ESTIMATES or confidence intervals?
│       └──► Bayesian Linear Regression
│
├── DATA IS SKEWED or need to predict extremes?
│       └──► Quantile Regression
│
└── NEED TO IDENTIFY MOST SIGNIFICANT FEATURES automatically?
        └──► Stepwise Regression (Backward Elimination)
```

---

## 🔑 Key Takeaways

- ✅ **Linear Regression** models a straight-line relationship between input features and a continuous output variable
- 📐 The **best-fit line** is found by minimising the **Sum of Squared Residuals** using the Ordinary Least Squares (OLS) method
- 🔢 The **hypothesis function** h(x) = β₀ + β₁x encodes all predictions; β₀ is the intercept, β₁ is the slope
- ⚠️ Linear regression relies on **7 key assumptions** — always verify linearity, homoscedasticity, and no multicollinearity
- 📚 There are **10 main types** — from Simple and Multiple LR to regularised versions (Ridge, Lasso, ElasticNet) and specialised ones (Bayesian, Quantile, Robust, Stepwise)
- 💰 The **cost function (MSE)** measures prediction error; **Gradient Descent** minimises it iteratively
- 🛡️ **Ridge** shrinks all coefficients; **Lasso** can zero them out (feature selection); **ElasticNet** combines both
- 📊 Always evaluate with **multiple metrics** — R², Adjusted R², RMSE, MAE — not just one
- ⚖️ Watch the **bias-variance tradeoff** — use regularisation to prevent overfitting
- 🚀 Linear Regression is the **starting point** for every ML project — always build it as your baseline before trying complex models

---

## 🛠️ Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib seaborn statsmodels scipy
```

**All imports used in this guide:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Scikit-learn
from sklearn.linear_model import (
    LinearRegression,
    Ridge, RidgeCV,
    Lasso, LassoCV,
    ElasticNet, ElasticNetCV,
    BayesianRidge,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor,
    SGDRegressor
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, fetch_california_housing

# Statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

---

## 📚 Further Reading

- [Scikit-learn: Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Gradient Descent in Linear Regression — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/gradient-descent-in-linear-regression/)
- [Ridge Regression from Scratch — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/implementation-of-ridge-regression-from-scratch-using-python/)
- [Lasso Regression from Scratch — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/implementation-of-lasso-regression-from-scratch-using-python/)
- [ElasticNet Regression — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/implementation-of-elastic-net-regression-from-scratch/)
- [Multiple Linear Regression — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/ml-multiple-linear-regression-using-python/)
- [Multicollinearity in Regression — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/multicollinearity-in-regression-analysis/)
- [Statsmodels: Regression](https://www.statsmodels.org/stable/regression.html)
- [ML Models and Types](../ml_models_types_README.md)
- [Feature Transformation Techniques](../feature_transformation_README.md)
- [Outlier Detection and Removal](../outlier_detection_README.md)

---
