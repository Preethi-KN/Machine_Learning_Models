# 🎲 Naive Bayes Classifier in Machine Learning
---

## 📌 Table of Contents

- [Overview](#-overview)
- [What is Naive Bayes?](#-what-is-naive-bayes)
- [Why is it Called "Naive"?](#-why-is-it-called-naive)
- [Key Terminology](#-key-terminology)
- [Bayes Theorem — The Mathematical Foundation](#-bayes-theorem--the-mathematical-foundation)
  - [Theorem Formula](#1-theorem-formula)
  - [Prior Posterior Likelihood](#2-prior-posterior-and-likelihood)
  - [From Bayes to Naive Bayes](#3-from-bayes-theorem-to-naive-bayes)
- [The Naive Independence Assumption](#-the-naive-independence-assumption)
- [Assumptions of Naive Bayes](#-assumptions-of-naive-bayes)
- [How Naive Bayes Works — Step by Step](#-how-naive-bayes-works--step-by-step)
  - [Weather Dataset Example](#-weather-dataset-example)
  - [Complete Manual Calculation](#-complete-manual-calculation)
- [Types of Naive Bayes](#-types-of-naive-bayes)
  - [Type 1 — Gaussian Naive Bayes](#-type-1--gaussian-naive-bayes)
  - [Type 2 — Multinomial Naive Bayes](#-type-2--multinomial-naive-bayes)
  - [Type 3 — Bernoulli Naive Bayes](#-type-3--bernoulli-naive-bayes)
  - [Type 4 — Complement Naive Bayes](#-type-4--complement-naive-bayes)
  - [Type 5 — Categorical Naive Bayes](#-type-5--categorical-naive-bayes)
- [Laplace Smoothing — Handling Zero Probability](#-laplace-smoothing--handling-zero-probability)
- [Evaluation Metrics](#-evaluation-metrics)
- [Visualising Naive Bayes](#-visualising-naive-bayes)
- [Advantages of Naive Bayes](#-advantages-of-naive-bayes)
- [Limitations of Naive Bayes](#-limitations-of-naive-bayes)
- [Real-World Applications](#-real-world-applications)
- [Types Comparison Table](#-types-comparison-table)
- [Decision Guide — Which Type to Use?](#-decision-guide--which-type-to-use)
- [Key Takeaways](#-key-takeaways)
- [Prerequisites](#-prerequisites)
- [Further Reading](#-further-reading)

---

## 🧭 Overview

Naive Bayes is a **fast, probabilistic supervised learning algorithm** used for classification. It applies **Bayes' Theorem** with a strong (naive) assumption that all features are **conditionally independent** given the class label.

```
Input Features (X)              Bayes' Theorem               Prediction
──────────────────    ──────────────────────────────────    ────────────
  feature_1 = x₁                                            Class A
  feature_2 = x₂   →   P(y|X) ∝ P(y) × ∏ P(xᵢ|y)   →   Class B  ← winner
  feature_3 = x₃                                            Class C
  ...

Pick the class y with the highest posterior probability P(y|X)
```

Despite the overly simplified independence assumption, Naive Bayes classifiers work remarkably well in practice — especially for **text classification**, **spam filtering**, and **medical diagnosis** — and are often the first algorithm to try for NLP tasks.

---

## 🔍 What is Naive Bayes?

Naive Bayes is a machine learning classification algorithm that predicts the category of a data point using probability. It assumes that all features are independent of each other.

- Naive Bayes performs well in many real-world applications such as **spam filtering, document categorisation and sentiment analysis**
- It has very few parameters and can predict at a **faster speed** than most other classification algorithms
- It is a **probabilistic classifier** — it models the probability of each class directly

**The algorithm in plain English:**

```
"Given these features (email words, symptoms, weather conditions),
 what is the most PROBABLE class?"

For each class → compute probability using Bayes' Theorem
Pick the class with the HIGHEST probability → that is the prediction
```

---

## 🤔 Why is it Called "Naive"?

It is named **"Naive"** because it assumes the presence of one feature does **not affect any other feature**. This is a naively strong assumption — in reality, features are often correlated (e.g., the word "free" and "money" often appear together in spam). The **"Bayes"** part refers to its basis in Bayes' Theorem.

```
Real world:
  Word "free" in an email DOES affect the probability of "money"
  They are correlated — knowing one tells you something about the other

Naive Bayes assumes:
  P("free" AND "money" | spam) = P("free" | spam) × P("money" | spam)
  ← treats them as independent — this is the naive assumption

Despite being wrong, it works well in practice because:
  ✅ Even if features are correlated, relative probability ranking stays correct
  ✅ The classifier only needs the ORDERING of probabilities, not exact values
```

---

## 📖 Key Terminology

| Term | Symbol | Description |
|---|---|---|
| **Prior Probability** | P(y) | Probability of a class before seeing any data |
| **Likelihood** | P(X\|y) | Probability of observing features X given class y |
| **Posterior Probability** | P(y\|X) | Updated probability of class y after seeing features X |
| **Evidence** | P(X) | Overall probability of the features — constant for all classes |
| **Feature Independence** | — | Core naive assumption: all features are independent given y |
| **Conditional Probability** | P(A\|B) | Probability of A given that B has occurred |
| **Argmax** | argmax | Select the class with the maximum posterior probability |
| **Laplace Smoothing** | α | Technique to avoid zero probabilities for unseen feature values |
| **Log Probability** | log P(y\|X) | Log-space computation to prevent numerical underflow |

---

## 📐 Bayes Theorem — The Mathematical Foundation

### 1. Theorem Formula

**Bayes' Theorem** provides a principled way to reverse conditional probabilities. It defines how to update our belief about a hypothesis given new evidence:

```
         P(X|y) · P(y)
P(y|X) = ──────────────
               P(X)

Where:
  P(y|X)  = Posterior  — probability of class y given features X  ← WHAT WE WANT
  P(X|y)  = Likelihood — probability of features X given class y   ← FROM TRAINING DATA
  P(y)    = Prior      — prior probability of class y              ← FROM TRAINING DATA
  P(X)    = Evidence   — marginal probability of features X        ← CONSTANT (same for all y)
```

---

### 2. Prior, Posterior, and Likelihood

```
PRIOR P(y): What we believe about the class BEFORE seeing the data

  From training data:
  P(Spam)     = 0.35  (35% of training emails were spam)
  P(Not Spam) = 0.65  (65% were not spam)

LIKELIHOOD P(X|y): How probable the features are GIVEN the class

  P("free" appears | Spam)     = 0.80  (80% of spam emails contain "free")
  P("free" appears | Not Spam) = 0.10  (10% of non-spam contain "free")

POSTERIOR P(y|X): What we believe about the class AFTER seeing the features

  P(Spam | "free" in email) = ?   ← this is what we compute
```

---

### 3. From Bayes' Theorem to Naive Bayes

Since P(X) is constant across all classes, we can drop it for classification:

```
P(y|X) ∝ P(X|y) · P(y)      ← proportional to

Naive Bayes independence assumption:
  P(X|y) = P(x₁|y) · P(x₂|y) · ... · P(xₙ|y)  = ∏ᵢ P(xᵢ|y)

Full Naive Bayes classifier:
  P(y|X) ∝ P(y) · ∏ᵢ P(xᵢ|y)

Final prediction:
  ŷ = argmax_y [ P(y) · ∏ᵢ P(xᵢ|y) ]

  Choose the class y that maximises this product
```

---

## 🔗 The Naive Independence Assumption

This is the **most critical concept** in Naive Bayes. Let's see what it means in practice:

```
Without independence assumption (full joint probability):
  P(x₁, x₂, x₃ | y) = P(x₁|y) · P(x₂|x₁, y) · P(x₃|x₁, x₂, y)
  ← need to track all feature combinations → exponential complexity

With Naive Bayes independence assumption:
  P(x₁, x₂, x₃ | y) = P(x₁|y) · P(x₂|y) · P(x₃|y)
  ← just multiply individual likelihoods → linear complexity

For a document with 10,000 words:
  Without assumption: astronomically many parameters to estimate
  With assumption:    just 10,000 per-word probabilities per class
```

**Practical analogy:**

```
Diagnosing a patient with symptoms: fever, cough, fatigue

Without independence:
  "Given fever, does the presence of cough change the probability of fatigue?"
  → Would need every combination of symptoms for every disease

With Naive Bayes:
  P(flu | fever, cough, fatigue) ∝ P(flu) × P(fever|flu) × P(cough|flu) × P(fatigue|flu)
  → Just need each symptom's probability given each disease separately
```

---

## ✅ Assumptions of Naive Bayes

| # | Assumption | Description |
|---|---|---|
| **1** | **Feature Independence** | Each feature contributes independently to predictions — no relations between features |
| **2** | **Continuous features are Gaussian** | If a feature is continuous, it is assumed to follow a normal distribution within each class |
| **3** | **Discrete features are Multinomial** | If a feature is discrete/count-based, it follows a multinomial distribution within each class |
| **4** | **Features are equally important** | All features contribute equally to the prediction — no feature is inherently more valuable |
| **5** | **No missing data** | Data should not contain missing values (or must be handled before fitting) |

---

## 🔢 How Naive Bayes Works — Step by Step

### 📌 Weather Dataset Example

A classic dataset: predict whether to **play golf** based on weather conditions.

| # | Outlook | Temperature | Humidity | Windy | Play Golf |
|---|---|---|---|---|---|
| 0 | Rainy | Hot | High | False | Yes |
| 1 | Rainy | Hot | High | True | No |
| 2 | Overcast | Hot | High | False | Yes |
| 3 | Sunny | Mild | High | False | No |
| 4 | Sunny | Cool | Normal | False | Yes |
| 5 | Sunny | Cool | Normal | True | No |
| 6 | Overcast | Cool | Normal | True | Yes |
| 7 | Rainy | Mild | High | False | No |
| 8 | Rainy | Cool | Normal | False | Yes |
| 9 | Sunny | Mild | Normal | False | Yes |
| 10 | Rainy | Mild | Normal | True | Yes |
| 11 | Overcast | Mild | High | True | Yes |
| 12 | Overcast | Hot | Normal | False | Yes |
| 13 | Sunny | Mild | High | True | No |

**Goal:** Given X = (Sunny, Hot, Normal, False) — will golf be played?

---

### 📌 Complete Manual Calculation

#### Step 1 — Compute Prior Probabilities

```
Total samples = 14

P(Yes) = 9/14 ≈ 0.643    (9 rows where Play Golf = Yes)
P(No)  = 5/14 ≈ 0.357    (5 rows where Play Golf = No)
```

#### Step 2 — Build Conditional Probability Tables

```
── Outlook ──────────────────────────────────────
              Yes (9)      No (5)
Sunny           2/9         3/5
Rainy           3/9         2/5
Overcast        4/9         0/5

── Temperature ──────────────────────────────────
              Yes (9)      No (5)
Hot             2/9         2/5
Mild            4/9         2/5
Cool            3/9         1/5

── Humidity ─────────────────────────────────────
              Yes (9)      No (5)
High            3/9         4/5
Normal          6/9         1/5

── Windy ────────────────────────────────────────
              Yes (9)      No (5)
False           6/9         2/5
True            3/9         3/5
```

#### Step 3 — Compute Posterior for X = (Sunny, Hot, Normal, False)

```
P(Yes | today) ∝ P(Yes) × P(Sunny|Yes) × P(Hot|Yes) × P(Normal|Yes) × P(False|Yes)
               ∝ (9/14) × (2/9) × (2/9) × (6/9) × (6/9)
               ∝ 0.6429 × 0.2222 × 0.2222 × 0.6667 × 0.6667
               ≈ 0.02116

P(No | today)  ∝ P(No) × P(Sunny|No) × P(Hot|No) × P(Normal|No) × P(False|No)
               ∝ (5/14) × (3/5) × (2/5) × (1/5) × (2/5)
               ∝ 0.3571 × 0.6000 × 0.4000 × 0.2000 × 0.4000
               ≈ 0.00686
```

#### Step 4 — Normalise Probabilities

```
Total = 0.02116 + 0.00686 = 0.02802

P(Yes | today) = 0.02116 / 0.02802 ≈ 0.755  (75.5%)
P(No  | today) = 0.00686 / 0.02802 ≈ 0.245  (24.5%)
```

#### Step 5 — Final Prediction

```
P(Yes | today) > P(No | today)
    0.755      >     0.245

Prediction: YES — Play Golf ✅
```

```python
# Full manual Naive Bayes for the golf dataset
import pandas as pd
import numpy as np

data = {
    'Outlook'    : ['Rainy','Rainy','Overcast','Sunny','Sunny','Sunny',
                    'Overcast','Rainy','Rainy','Sunny','Rainy','Overcast','Overcast','Sunny'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool',
                    'Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity'   : ['High','High','High','High','Normal','Normal','Normal',
                    'High','Normal','Normal','Normal','High','Normal','High'],
    'Windy'      : [False,True,False,False,False,True,True,
                    False,False,False,True,True,False,True],
    'PlayGolf'   : ['Yes','No','Yes','No','Yes','No','Yes',
                    'No','Yes','Yes','Yes','Yes','Yes','No']
}
df = pd.DataFrame(data)

# Prior probabilities
priors = df['PlayGolf'].value_counts(normalize=True)
print("── Prior Probabilities ───────────────────────")
print(priors)

# Conditional probabilities for each feature
features = ['Outlook', 'Temperature', 'Humidity', 'Windy']
print("\n── Conditional Probability Tables ───────────")
for feat in features:
    ct = pd.crosstab(df[feat], df['PlayGolf'], normalize='columns')
    print(f"\n{feat}:\n{ct.round(3)}")

# Predict for X = (Sunny, Hot, Normal, False)
X_new = {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'Normal', 'Windy': False}

print("\n── Prediction for:", X_new, "────────────────")
posteriors = {}
for cls in ['Yes', 'No']:
    prob = priors[cls]
    for feat, val in X_new.items():
        ct = pd.crosstab(df[feat], df['PlayGolf'], normalize='columns')
        prob *= ct.loc[val, cls]
    posteriors[cls] = prob

# Normalise
total = sum(posteriors.values())
for cls in posteriors:
    print(f"  P({cls} | features) = {posteriors[cls]/total:.4f} ({posteriors[cls]/total*100:.1f}%)")

print(f"\n  Prediction: {max(posteriors, key=posteriors.get)}")
```

---

## 📚 Types of Naive Bayes

There are **5 types** of Naive Bayes classifiers in scikit-learn, each assuming a different probability distribution for features.

```
Naive Bayes Family
│
├── Gaussian NB       → Continuous features → Normal distribution
├── Multinomial NB    → Count features     → Multinomial distribution
├── Bernoulli NB      → Binary features    → Bernoulli distribution
├── Complement NB     → Imbalanced text    → Complement of Multinomial
└── Categorical NB    → Categorical feats  → Categorical distribution
```

---

## 📌 Type 1 — Gaussian Naive Bayes

Used when features are **continuous and approximately normally distributed**. For each class, it estimates the **mean (μ)** and **variance (σ²)** of each feature, then uses the Gaussian PDF to compute likelihoods.

```
                    1                  (xᵢ − μᵧ)²
P(xᵢ | y) = ─────────────── × exp( − ─────────── )
              √(2π × σ²ᵧ)               2σ²ᵧ

Where:
  μᵧ  = mean of feature xᵢ for class y  (estimated from training data)
  σ²ᵧ = variance of feature xᵢ for class y (estimated from training data)

Bell curve — the Gaussian PDF:

   P(x|y) │    ╭──────╮
           │   ╱        ╲
           │  ╱          ╲
           │ ╱            ╲
           │╱              ╲
           └─────────────────── x
                   μ
                   ↑ centre of distribution for class y
```

### Python Implementation

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load Iris dataset ─────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train Gaussian NB ─────────────────────────────────────────
gnb = GaussianNB(
    var_smoothing=1e-9    # tiny variance added for numerical stability
)
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
y_prob = gnb.predict_proba(X_test)

print("── Gaussian Naive Bayes (Iris) ─────────────────")
print(f"Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
cv = cross_val_score(gnb, X, y, cv=5, scoring='accuracy')
print(f"CV (5-fold): {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
print(f"\n{classification_report(y_test, y_pred, target_names=iris.target_names)}")

# ── Learned parameters: mean and variance per class ───────────
print("── Learned Class Statistics ────────────────────")
for i, cls in enumerate(iris.target_names):
    print(f"\nClass: {cls}")
    for j, feat in enumerate(iris.feature_names):
        print(f"  {feat}: mean={gnb.theta_[i, j]:.3f}, var={gnb.var_[i, j]:.5f}")

# ── Confusion Matrix ──────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.title('Gaussian NB — Confusion Matrix (Iris)')
plt.tight_layout(); plt.show()

# ── Breast Cancer dataset ─────────────────────────────────────
cancer = load_breast_cancer()
X_c, y_c = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

gnb_c = GaussianNB()
gnb_c.fit(X_tr, y_tr)
print(f"\nBreast Cancer Accuracy: {accuracy_score(y_te, gnb_c.predict(X_te))*100:.2f}%")
```

**How it handles the Gaussian PDF computation:**

```python
# Manual Gaussian PDF computation
def gaussian_pdf(x, mean, var):
    """Compute Gaussian probability density for a single value."""
    coeff     = 1.0 / np.sqrt(2 * np.pi * var)
    exponent  = np.exp(-((x - mean) ** 2) / (2 * var))
    return coeff * exponent

# Example: P(sepal_length=5.1 | class=Setosa)
mean_setosa_sepal = 5.006   # learned from training
var_setosa_sepal  = 0.1219

prob = gaussian_pdf(5.1, mean_setosa_sepal, var_setosa_sepal)
print(f"P(sepal_length=5.1 | Setosa) = {prob:.6f}")
```

| ✅ Pros | ❌ Cons |
|---|---|
| Handles continuous features naturally | Assumes normal distribution (may not hold) |
| No feature scaling required | Poor if features are multimodal or skewed |
| Works well with small datasets | Sensitive to outliers (distorts mean/variance) |
| Online learning via `partial_fit` | Not suitable for discrete/text features |

**Best for:** Medical diagnosis, sensor data, physical measurements, image features

---

## 📌 Type 2 — Multinomial Naive Bayes

Used when features represent **counts or frequencies** — especially for **text classification** where features are word counts or TF-IDF values.

```
Feature representation for text:
  Document: "the cat sat on the mat"

  As word counts: {"the": 2, "cat": 1, "sat": 1, "on": 1, "mat": 1}

Multinomial Likelihood:
                (word count of w in class y) + α (smoothing)
  P(w | y) = ──────────────────────────────────────────────────
              (total words in class y) + α × |vocabulary|

Where:
  α = Laplace smoothing (default = 1)
  |vocabulary| = total unique words across all classes
```

### Python Implementation

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

# ── Simple text example from scratch ─────────────────────────
texts = [
    "I love this movie it is great",           # Positive
    "This film is absolutely fantastic",        # Positive
    "Amazing story wonderful acting",           # Positive
    "Brilliant film loved every moment",        # Positive
    "This movie was terrible and boring",       # Negative
    "Awful film complete waste of time",        # Negative
    "Hated it very disappointing movie",        # Negative
    "Worst film I have ever seen",              # Negative
]
labels = [1, 1, 1, 1, 0, 0, 0, 0]   # 1=Positive, 0=Negative

# ── Pipeline: Vectorise → Classify ───────────────────────────
mnb_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1, 2))),   # unigrams + bigrams
    ('clf',        MultinomialNB(alpha=1.0))                # alpha = Laplace smoothing
])
mnb_pipeline.fit(texts, labels)

# Test predictions
test_texts = [
    "This is a wonderful brilliant film",
    "I hated this terrible boring movie",
    "Fantastic story great acting loved it"
]
probs = mnb_pipeline.predict_proba(test_texts)
preds = mnb_pipeline.predict(test_texts)

print("── Multinomial NB — Sentiment Analysis ─────────")
for txt, pred, prob in zip(test_texts, preds, probs):
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"  Text    : '{txt[:40]}...'")
    print(f"  Pred    : {sentiment} | P(Pos)={prob[1]:.3f}, P(Neg)={prob[0]:.3f}\n")

# ── Feature log probabilities — top words per class ───────────
vocab = mnb_pipeline.named_steps['vectorizer'].get_feature_names_out()
clf   = mnb_pipeline.named_steps['clf']

print("── Top 10 Most Important Words per Class ───────")
for i, cls_name in enumerate(['Negative', 'Positive']):
    top_idx = np.argsort(clf.feature_log_prob_[i])[-10:][::-1]
    top_words = [(vocab[j], np.exp(clf.feature_log_prob_[i][j])) for j in top_idx]
    print(f"\n{cls_name}:")
    for word, prob in top_words:
        print(f"  P({word!r} | {cls_name}) = {prob:.4f}")

# ── 20 Newsgroups Dataset (real-world) ───────────────────────
categories = ['sci.med', 'sci.space', 'rec.sport.hockey', 'talk.politics.misc']
train_data = fetch_20newsgroups(subset='train', categories=categories)
test_data  = fetch_20newsgroups(subset='test',  categories=categories)

news_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
    ('clf',   MultinomialNB(alpha=0.1))
])
news_pipeline.fit(train_data.data, train_data.target)
y_pred_news = news_pipeline.predict(test_data.data)

print(f"\n── 20 Newsgroups Accuracy: {accuracy_score(test_data.target, y_pred_news)*100:.2f}% ──")
print(classification_report(test_data.target, y_pred_news, target_names=categories))
```

**Understanding alpha (Laplace Smoothing):**

```python
# Effect of alpha on MultinomialNB
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score

alphas   = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
cv_means = []
cv_stds  = []

for alpha in alphas:
    pipe = Pipeline([
        ('vec', CountVectorizer()),
        ('clf', MultinomialNB(alpha=alpha))
    ])
    scores = cross_val_score(pipe, train_data.data, train_data.target, cv=5)
    cv_means.append(scores.mean())
    cv_stds.append(scores.std())

plt.figure(figsize=(9, 5))
plt.errorbar(alphas, cv_means, yerr=cv_stds, fmt='o-',
             color='steelblue', linewidth=2, capsize=4)
plt.xscale('log')
plt.xlabel('Alpha (Laplace Smoothing)')
plt.ylabel('CV Accuracy')
plt.title('Multinomial NB — Effect of Alpha on Accuracy')
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

| ✅ Pros | ❌ Cons |
|---|---|
| Excellent for text classification | Features must be non-negative counts |
| Handles high-dimensional sparse data | Cannot handle continuous features |
| Fast with large vocabularies | Requires Laplace smoothing (alpha tuning) |
| Interpretable word probabilities | Sensitive to text preprocessing |

**Best for:** Spam filtering, document classification, topic modelling, sentiment analysis

---

## 📌 Type 3 — Bernoulli Naive Bayes

Used when features are **binary (0 or 1)** — each feature indicates whether something is **present or absent**. For text, this means whether a word appears in the document (not how many times).

```
Feature representation:
  Document: "the cat sat on the mat"
  Vocabulary: ["cat", "dog", "mat", "sat", "ran", "the"]

  Multinomial (counts):   [1, 0, 1, 1, 0, 2]
  Bernoulli (binary):     [1, 0, 1, 1, 0, 1]  ← 1 if present, 0 if absent

Bernoulli Likelihood:
  P(xᵢ | y) = P(xᵢ=1 | y)^xᵢ  × (1 − P(xᵢ=1 | y))^(1-xᵢ)

  When xᵢ = 1 (word present):  P(word appears | class)
  When xᵢ = 0 (word absent):   P(word absent  | class) = 1 − P(word | class)

Key difference from Multinomial:
  Bernoulli penalises for ABSENT features (missing words also matter)
  Multinomial ignores absent features
```

### Python Implementation

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# ── Text classification — Bernoulli (binary) ──────────────────
texts_train = [
    "I love this movie great acting",         # Positive
    "Fantastic film brilliant story",          # Positive
    "Amazing movie wonderful experience",      # Positive
    "Absolutely terrible boring film",         # Negative
    "Worst movie ever hated it completely",    # Negative
    "Awful boring terrible waste time",        # Negative
]
labels_train = [1, 1, 1, 0, 0, 0]

texts_test = [
    "Great film loved the story",
    "Boring and terrible movie",
    "Wonderful brilliant acting"
]
labels_test = [1, 0, 1]

# ── Bernoulli NB pipeline ─────────────────────────────────────
bnb_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(binary=True)),   # binary=True → 0/1 features
    ('clf',        BernoulliNB(alpha=1.0,
                               binarize=None))       # already binary from vectorizer
])
bnb_pipeline.fit(texts_train, labels_train)
y_pred_bnb = bnb_pipeline.predict(texts_test)
y_prob_bnb = bnb_pipeline.predict_proba(texts_test)

print("── Bernoulli Naive Bayes ────────────────────────")
for txt, pred, prob in zip(texts_test, y_pred_bnb, y_prob_bnb):
    print(f"  '{txt}'")
    print(f"  → {'Positive' if pred==1 else 'Negative'} | P(Pos)={prob[1]:.3f}\n")

# ── Continuous → Binary (using binarize threshold) ────────────
from sklearn.naive_bayes import BernoulliNB
import numpy as np

# BernoulliNB can binarise continuous data internally
X_cont = np.array([[1.5, 2.3, 0.1], [0.3, 1.8, 2.4], [2.1, 0.2, 1.7]])
y_cont = np.array([0, 1, 1])

bnb_thresh = BernoulliNB(binarize=1.0)   # threshold: values > 1.0 become 1, else 0
bnb_thresh.fit(X_cont, y_cont)
print(f"Bernoulli NB with threshold 1.0: {bnb_thresh.predict([[1.2, 2.5, 0.3]])}")
```

**Bernoulli vs Multinomial — Key Differences:**

```
Feature         Bernoulli NB              Multinomial NB
────────────────────────────────────────────────────────
Input type      Binary (0 or 1)           Counts or frequencies
Text encoding   Word present/absent       Word frequency
Absent words    Penalised explicitly       Ignored
Short docs      Better                    Better for longer docs
Parameter       binarize threshold        alpha smoothing
Use case        Short text, email flags   Documents, long text
```

| ✅ Pros | ❌ Cons |
|---|---|
| Explicitly models feature absence | Loses frequency information |
| Works with very short documents | Binary encoding loses nuance |
| Good for sparse, high-dimensional data | Word count information discarded |

**Best for:** Short document classification, email spam (presence/absence of words), topic tagging

---

## 📌 Type 4 — Complement Naive Bayes

**An improvement over Multinomial NB for imbalanced datasets.** Instead of modelling the probability of features **belonging to** a class, it models the probability of features belonging to the **complement** (everything except that class). This corrects for the poor assumptions of standard MultinomialNB when class distribution is skewed.

```
Standard Multinomial NB:
  Estimate P(xᵢ | y=k)     ← feature probability WITHIN class k

Complement NB:
  Estimate P(xᵢ | y ≠ k)   ← feature probability in ALL OTHER classes

Why it works better for imbalanced data:
  If Class A has 90% of data and Class B has 10%:
  → Multinomial NB is biased toward Class A (more training data)
  → Complement NB normalises by using the complement counts

Final rule for Complement NB:
  ŷ = argmin_k  Σᵢ [ xᵢ × log P(xᵢ | y ≠ k) ]
               ← pick class where complement probability is LOWEST
```

### Python Implementation

```python
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# ── Load imbalanced text dataset ──────────────────────────────
categories = ['sci.med', 'sci.space', 'rec.sport.hockey', 'talk.politics.misc']
train_news  = fetch_20newsgroups(subset='train', categories=categories)
test_news   = fetch_20newsgroups(subset='test',  categories=categories)

# ── Compare Multinomial vs Complement NB ──────────────────────
for name, clf in [('MultinomialNB', __import__('sklearn.naive_bayes', fromlist=['MultinomialNB']).MultinomialNB()),
                   ('ComplementNB',  ComplementNB())]:
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('clf',   clf)
    ])
    pipe.fit(train_news.data, train_news.target)
    acc = accuracy_score(test_news.target, pipe.predict(test_news.data))
    print(f"{name:20s}: {acc*100:.2f}%")

# ── Complement NB with alpha tuning ───────────────────────────
from sklearn.model_selection import GridSearchCV

comp_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
    ('clf',   ComplementNB())
])

param_grid = {'clf__alpha': [0.01, 0.1, 0.5, 1.0, 2.0]}
gs = GridSearchCV(comp_pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs.fit(train_news.data, train_news.target)

print(f"\nBest alpha : {gs.best_params_['clf__alpha']}")
print(f"Best CV acc: {gs.best_score_*100:.2f}%")
print(f"Test acc   : {gs.score(test_news.data, test_news.target)*100:.2f}%")
print(f"\n{classification_report(test_news.target, gs.predict(test_news.data), target_names=categories)}")
```

| ✅ Pros | ❌ Cons |
|---|---|
| Better than MultinomialNB for imbalanced text | Slightly more complex than standard MultinomialNB |
| More stable parameter estimates | Still requires non-negative features |
| Higher accuracy on text classification | Not suitable for non-text/count features |

**Best for:** Imbalanced text classification, news categorisation, sentiment analysis on skewed datasets

---

## 📌 Type 5 — Categorical Naive Bayes

Used when features are **nominal/categorical** — each feature takes one of a set of discrete, non-numeric categories. Unlike Multinomial NB, it does not require count-based features — it works directly with **categorical values** (e.g., "red", "blue", "green").

```
Feature types Categorical NB handles:
  Colour    → red / blue / green / yellow
  Size      → small / medium / large
  Outlook   → sunny / rainy / overcast
  Grade     → A / B / C / D / F

Categorical Likelihood:
  P(xᵢ = c | y) = (count of (xᵢ=c, y=class) + α) / (count of y=class + α × n_categories_i)

Each feature can have a different number of categories.
```

### Python Implementation

```python
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Golf dataset — pure categorical features ──────────────────
data = {
    'Outlook'    : ['Rainy','Rainy','Overcast','Sunny','Sunny','Sunny',
                    'Overcast','Rainy','Rainy','Sunny','Rainy','Overcast','Overcast','Sunny'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool',
                    'Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity'   : ['High','High','High','High','Normal','Normal','Normal',
                    'High','Normal','Normal','Normal','High','Normal','High'],
    'Windy'      : ['False','True','False','False','False','True','True',
                    'False','False','False','True','True','False','True'],
    'PlayGolf'   : ['Yes','No','Yes','No','Yes','No','Yes',
                    'No','Yes','Yes','Yes','Yes','Yes','No']
}

df_golf = pd.DataFrame(data)
X_golf  = df_golf[['Outlook','Temperature','Humidity','Windy']]
y_golf  = (df_golf['PlayGolf'] == 'Yes').astype(int)

# CategoricalNB needs integer-encoded categories
encoder  = OrdinalEncoder()
X_enc    = encoder.fit_transform(X_golf)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_enc, y_golf, test_size=0.3, random_state=42
)

cat_nb = CategoricalNB(alpha=1.0)   # Laplace smoothing
cat_nb.fit(X_tr, y_tr)

print("── Categorical Naive Bayes (Golf) ───────────────")
print(f"Accuracy: {accuracy_score(y_te, cat_nb.predict(X_te))*100:.2f}%")

# ── Predict for new sample ────────────────────────────────────
new_sample = pd.DataFrame([{
    'Outlook': 'Sunny', 'Temperature': 'Hot',
    'Humidity': 'Normal', 'Windy': 'False'
}])
new_enc  = encoder.transform(new_sample)
pred     = cat_nb.predict(new_enc)
pred_prb = cat_nb.predict_proba(new_enc)

print(f"\nNew sample: Sunny, Hot, Normal, False")
print(f"Prediction : {'Play Golf ✅' if pred[0]==1 else 'No Golf ❌'}")
print(f"P(Yes)={pred_prb[0][1]:.4f}, P(No)={pred_prb[0][0]:.4f}")

# ── Larger categorical dataset ────────────────────────────────
from sklearn.datasets import make_classification

# CategoricalNB on discretised data
from sklearn.preprocessing import KBinsDiscretizer

data_cat = make_classification(n_samples=1000, n_features=5,
                                n_informative=3, random_state=42)
X_raw, y_raw = data_cat

discretiser = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
X_discrete  = discretiser.fit_transform(X_raw)

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_discrete, y_raw, test_size=0.2, random_state=42
)
cat_nb2 = CategoricalNB(alpha=1.0)
cat_nb2.fit(X_tr2, y_tr2)
print(f"\nCategorical NB on discretised data: {cat_nb2.score(X_te2, y_te2)*100:.2f}%")
```

| ✅ Pros | ❌ Cons |
|---|---|
| Works directly with categorical features | Requires integer encoding |
| No assumption of numerical ordering | Features must be ordinal-encoded first |
| Handles different category counts per feature | Not for continuous or count data |

**Best for:** Survey data, medical record categorisation, weather datasets, business rule classification

---

## 🧂 Laplace Smoothing — Handling Zero Probability

Without smoothing, if a feature value never appears with a class in training data, its probability is **zero** — and multiplying by zero collapses the entire posterior to zero, regardless of all other evidence.

```
Problem — Zero Probability:
  Suppose "unicorn" never appears in training data spam emails.
  P("unicorn" | Spam) = 0/total_spam_words = 0

  P(Spam | email with "unicorn") ∝ ... × 0 = 0

  The model refuses to classify the email as spam even if all
  other words are perfectly spam-like → incorrect prediction!

Solution — Laplace Smoothing (add-α smoothing):
           count(w, y) + α
P(w | y) = ─────────────────────────────
           total_words_in_y + α × |V|

Where:
  α = smoothing parameter (default = 1.0)
  |V| = vocabulary size
  count(w, y) = times word w appears in class y

  With α=1:
    P("unicorn" | Spam) = (0 + 1) / (total + vocab) > 0  ✅
```

---

## ✅ Advantages of Naive Bayes

| Advantage | Description |
|---|---|
| **Computationally efficient** | Extremely fast to train and predict — only needs to estimate frequency counts or statistics |
| **Works with limited data** | Performs well even with a small training set |
| **Handles high-dimensional data** | Scales linearly with features — excellent for text with 10,000+ word features |
| **Simple implementation** | Few parameters to tune — mostly just smoothing (alpha) |
| **Works well with categorical features** | Natural fit for categorical data without encoding tricks |
| **Probabilistic output** | Returns calibrated probabilities, not just class labels |
| **Online learning** | `partial_fit()` allows incremental training on streaming data |
| **Multi-class by default** | No modification needed for 3+ classes |
| **Robust to irrelevant features** | Irrelevant features contribute small likelihoods that are easily outweighed |

---

## ❌ Limitations of Naive Bayes

| Limitation | Description | Fix |
|---|---|---|
| **Feature independence assumption violated** | In reality, features are often correlated | Use correlated models (LR, SVM, trees) if correlation is critical |
| **Zero probability problem** | Unseen feature-class combinations → zero probability collapses prediction | Always use Laplace smoothing (alpha > 0) |
| **Continuous feature distribution assumption** | Gaussian NB forces normal distribution | Use transformations or discretise to CategoricalNB |
| **Cannot learn feature interactions** | Treats features independently by design | Use tree-based models or logistic regression with interaction terms |
| **Poor with redundant features** | Correlated features count the same evidence twice | Feature selection; remove redundant features |
| **Probability estimates can be poorly calibrated** | May give extreme probabilities (near 0 or 1) | Use CalibratedClassifierCV from sklearn |


---

## 🌍 Real-World Applications

| Application | Type Used | Features | Why NB Works |
|---|---|---|---|
| **Spam Email Filtering** | Multinomial / Bernoulli | Word frequencies or presence | Fast, text-native, excellent recall on known spam patterns |
| **Sentiment Analysis** | Multinomial | TF-IDF word features | Handles large vocabularies with few parameters |
| **Document Classification** | Multinomial / Complement | Word counts, n-grams | Linear scalability with vocabulary size |
| **Medical Diagnosis** | Gaussian | Lab values, vitals | Works well with continuous medical measurements |
| **Weather Prediction** | Categorical / Gaussian | Outlook, humidity, temperature | Naturally models categorical weather conditions |
| **Credit Scoring** | Gaussian / Categorical | Income, age, employment | Fast real-time predictions |
| **News Categorisation** | Complement NB | TF-IDF features | Handles imbalanced news category sizes |
| **Face Recognition (features)** | Gaussian | Pixel intensities or PCA components | Fast baseline for visual feature vectors |

---

## 📊 Types Comparison Table

| Type | Feature Type | Distribution | Formula | Alpha | Best For | sklearn Class |
|---|---|---|---|---|---|---|
| **Gaussian NB** | Continuous | Normal (Gaussian) | PDF formula | No | Medical, sensor, physical data | `GaussianNB` |
| **Multinomial NB** | Count / TF-IDF | Multinomial | Count ratios | Yes | Text, document classification | `MultinomialNB` |
| **Bernoulli NB** | Binary (0/1) | Bernoulli | Presence probability | Yes | Short text, word presence/absence | `BernoulliNB` |
| **Complement NB** | Count (non-negative) | Complement of Multinomial | Complement ratios | Yes | Imbalanced text datasets | `ComplementNB` |
| **Categorical NB** | Nominal categories | Categorical | Category frequency | Yes | Survey, weather, categorical EHR | `CategoricalNB` |

---

## 🗺️ Decision Guide — Which Type to Use?

```
What type are your features?
│
├── CONTINUOUS (measurements, sensor readings)?
│       └──► Gaussian Naive Bayes (GaussianNB)
│               └── Are features strongly non-normal?
│                       → Transform (log, sqrt) or discretise → CategoricalNB
│
├── COUNT-BASED (word counts, term frequencies)?
│       │
│       ├── Is dataset imbalanced across classes?
│       │       └──► Complement NB (ComplementNB)
│       │
│       └── Dataset is balanced?
│               └──► Multinomial NB (MultinomialNB)
│
├── BINARY (presence/absence, 0 or 1)?
│       └──► Bernoulli NB (BernoulliNB)
│               └── Are you working with short documents or word flags?
│                       → Bernoulli is the best choice
│
└── CATEGORICAL (colours, grades, weather conditions)?
        └──► Categorical NB (CategoricalNB)
                └── Remember to OrdinalEncode first!
```

---

## 🔑 Key Takeaways

- ✅ **Naive Bayes** applies Bayes' Theorem with the naive assumption that all features are **conditionally independent** given the class
- 🎲 The classifier picks the class with the **highest posterior probability** P(y|X) ∝ P(y) × ∏P(xᵢ|y)
- 🏷️ It is called "naive" because the independence assumption is unrealistic — yet it works remarkably well in practice
- 📚 There are **5 types** in scikit-learn: Gaussian (continuous), Multinomial (counts), Bernoulli (binary), Complement (imbalanced text), and Categorical (nominal)
- 🚀 Naive Bayes is **the fastest classifier** to train and predict — ideal for real-time and large-scale applications
- 🧂 **Laplace smoothing** (alpha parameter) is critical — it prevents zero-probability issues for unseen feature-class combinations
- 📝 **MultinomialNB** is the gold standard for text classification; **GaussianNB** for continuous numeric features; **BernoulliNB** for binary features
- ⚠️ **ComplementNB** outperforms MultinomialNB when class distribution is imbalanced
- 🎯 Despite its limitations, Naive Bayes often outperforms or matches complex models for **text classification tasks**
- 📊 Always use **Log probabilities** in implementation to avoid numerical underflow from multiplying many small probabilities

---

## 🛠️ Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy
```

**All imports used in this guide:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Naive Bayes classifiers
from sklearn.naive_bayes import (GaussianNB, MultinomialNB, BernoulliNB,
                                  ComplementNB, CategoricalNB)

# Feature extraction for text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Preprocessing
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                    OrdinalEncoder, KBinsDiscretizer)

# Pipeline
from sklearn.pipeline import Pipeline

# Datasets
from sklearn.datasets import (load_iris, load_breast_cancer, load_digits,
                               fetch_20newsgroups, make_classification)

# Model evaluation
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)

# Probability calibration
from sklearn.calibration import CalibratedClassifierCV
```

---

## 📚 Further Reading

- [Scikit-learn: Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [Scikit-learn: GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
- [Scikit-learn: MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
- [Scikit-learn: BernoulliNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)
- [Scikit-learn: ComplementNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html)
- [Scikit-learn: CategoricalNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html)
- [Bayes' Theorem — GeeksforGeeks](https://www.geeksforgeeks.org/maths/bayes-theorem/)
- [Gaussian NB — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/gaussian-naive-bayes/)ME](../ml_models_types_README.md)

---