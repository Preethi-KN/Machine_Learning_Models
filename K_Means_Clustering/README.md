# 📊 K-Means Clustering Algorithm
---

## 📖 Table of Contents

- [Overview](#overview)
- [Key Concepts & Terminology](#key-concepts--terminology)
- [How K-Means Works — Step by Step](#how-k-means-works--step-by-step)
- [Visual Walkthrough](#visual-walkthrough)
- [Mathematical Foundation](#mathematical-foundation)
  - [Euclidean Distance](#1-euclidean-distance)
  - [Centroid Update Formula](#2-centroid-update-formula)
  - [Objective Function (WCSS / Inertia)](#3-objective-function-wcss--inertia)
- [Choosing the Optimal K — Elbow Method](#choosing-the-optimal-k--elbow-method)
- [Centroid Initialization Strategies](#centroid-initialization-strategies)
- [Python Implementation from Scratch](#python-implementation-from-scratch)
- [Step-by-Step Implementation with NumPy & Matplotlib](#step-by-step-implementation-with-numpy--matplotlib)
- [Implementation with Scikit-Learn](#implementation-with-scikit-learn)
- [Evaluation Metrics for Clustering](#evaluation-metrics-for-clustering)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Variants of K-Means](#variants-of-k-means)
- [K-Means vs Other Clustering Algorithms](#k-means-vs-other-clustering-algorithms)
- [Applications](#applications)
- [Advantages](#advantages)
- [Disadvantages & Challenges](#disadvantages--challenges)
- [K-Means vs Supervised Algorithms](#k-means-vs-supervised-algorithms)
- [Summary Cheat Sheet](#summary-cheat-sheet)
- [References](#references)

---

## Overview

**K-Means Clustering** is an **unsupervised machine learning algorithm** that groups similar data points into `K` distinct clusters without needing labelled data. The algorithm discovers hidden structure in raw, unorganised data by iteratively assigning each data point to its nearest cluster centre (centroid) and refining those centres until they stabilise.

### Key Characteristics

| Property | Description |
|---|---|
| **Type** | Unsupervised Learning |
| **Task** | Clustering (grouping / segmentation) |
| **Requires Labels?** | ❌ No — works on unlabelled data |
| **Parametric?** | No (Non-parametric) |
| **Learning Style** | Iterative Optimisation |
| **Distance Metric** | Euclidean Distance (default) |
| **Output** | Cluster assignments + centroid coordinates |
| **Scalability** | Good for medium to large datasets |

### Core Intuition

```
K-Means answers the question:
  "How can I naturally group N data points into K similar groups?"

Think of it like sorting a bag of mixed fruit by type:
  🍎 🍊 🍋 🍎 🍊 🍋 🍎 🍊

  → Group 1 (Apples)  : 🍎 🍎 🍎
  → Group 2 (Oranges) : 🍊 🍊 🍊
  → Group 3 (Lemons)  : 🍋 🍋

Each fruit goes to the group whose "centre" it most resembles.
K-Means does exactly this — in any number of dimensions.
```

### Supervised vs Unsupervised Learning

```
Supervised Learning       Unsupervised Learning (K-Means)
─────────────────         ──────────────────────────────
Has labelled data   →     No labels needed
Predicts categories →     Discovers hidden groups
e.g. spam / not spam →    e.g. customer segments A, B, C
```

---

## Key Concepts & Terminology

### 🔵 Cluster
A **group of data points** that are more similar to each other than to points in other groups. K-Means partitions the dataset into exactly `K` non-overlapping clusters.

### ⭐ Centroid
The **mean position** of all data points within a cluster. It acts as the "representative" or "centre of gravity" of that cluster.

```
Centroid = Average of all points in the cluster
         = (1/|Cₖ|) · Σ xᵢ    for all xᵢ in cluster Cₖ
```

### 🔢 K
The **number of clusters** the algorithm will form. This is the single most important hyperparameter — it must be specified before training.

### 📏 Inertia (WCSS)
**Within-Cluster Sum of Squares** — the total squared distance from each data point to its assigned centroid. K-Means minimises this value.

```
Lower Inertia  →  More compact, tighter clusters  →  Better
Higher Inertia →  More spread-out clusters         →  Worse
```

### 🔄 Iteration
Each complete cycle of **assign → update** steps. The algorithm repeats until convergence (centroids stop moving) or max iterations are reached.

### ✅ Convergence
The algorithm **converges** when centroid positions no longer change significantly between iterations — the solution is stable.

---

## How K-Means Works — Step by Step

K-Means follows a simple, elegant iterative process called **Expectation-Maximisation (EM)**:

```
┌───────────────────────────────────────────────────────────────────┐
│                   K-Means Algorithm Flow                          │
└───────────────────────────────────────────────────────────────────┘

Step 1: Choose K
   └─► Decide how many clusters to form (e.g., K = 3)

Step 2: Initialise K Centroids
   └─► Randomly place K centroids in the feature space
       (or use smarter initialisation like K-Means++)

Step 3: Assignment Step  ← (E-step)
   └─► For each data point xᵢ:
       Compute distance to ALL K centroids
       Assign xᵢ to the cluster of the NEAREST centroid

Step 4: Update Step  ← (M-step)
   └─► For each cluster k:
       Recompute centroid = mean of all points assigned to cluster k

Step 5: Check for Convergence
   ├─► If centroids DID NOT move (or moved < threshold) → STOP ✅
   └─► If centroids moved → Return to Step 3 and repeat

Step 6: Output
   └─► Final K clusters with their centroid coordinates
       Each data point has a cluster label (0, 1, 2, ..., K-1)
```

### Stopping Criteria

| Criterion | Description |
|---|---|
| **Centroid stability** | Centroids move less than a threshold `tol` between iterations |
| **Max iterations** | Algorithm stops after `max_iter` iterations regardless |
| **No reassignments** | No data point changes its cluster assignment |

---

## Visual Walkthrough

A detailed example with K = 3 clusters:

### Iteration 0 — Random Initialisation

```
Feature 2
  │
  │   o  o            ★ ← Random Centroid 1
  │      o  o
  │
  │       ★ ← Random Centroid 2
  │  x  x
  │   x  x     △  △
  │             △  △
  │   ★ ← Random Centroid 3
  └─────────────────────── Feature 1
```

### Iteration 1 — Assignment Step

```
Each point is coloured by its nearest centroid:

  │   🔵 🔵           ★₁
  │      🔵 🔵
  │
  │        ★₂
  │  🔴 🔴
  │   🔴 🔴    🟢 🟢
  │              🟢 🟢
  │   ★₃
  └─────────────────────── Feature 1
```

### Iteration 1 — Update Step

```
Centroids shift to the mean of their assigned points:

  │   🔵 🔵
  │      🔵★₁ ← moved!
  │
  │  🔴 🔴
  │   🔴★₂ ← moved!    🟢 🟢
  │              🟢★₃ ← moved!
  │
  └─────────────────────── Feature 1
```

### Final — Convergence

```
  │   🔵 🔵
  │      🔵 ★₁ (stable)
  │
  │  🔴 🔴
  │   🔴 ★₂ (stable)   🟢 🟢
  │              🟢 ★₃ (stable)
  │
  └─────────────────────── Feature 1

  Centroids no longer move → Algorithm converges ✅
```

---

## Mathematical Foundation

### 1. Euclidean Distance

The default distance metric used to measure similarity between a data point and a centroid:

```
d(xᵢ, μₖ) = √[ Σⱼ (xᵢⱼ − μₖⱼ)² ]
```

| Symbol | Meaning |
|---|---|
| `xᵢ` | The i-th data point (a vector of features) |
| `μₖ` | Centroid of cluster k |
| `j` | Feature dimension index |

---

### 2. Centroid Update Formula

After assigning points to clusters, each centroid is recomputed as the **mean** of its member points:

```
         1
μₖ  =  ──────  ·  Σ xᵢ
        |Cₖ|     xᵢ∈Cₖ
```

| Symbol | Meaning |
|---|---|
| `μₖ` | New centroid position for cluster k |
| `Cₖ` | Set of all data points assigned to cluster k |
| `\|Cₖ\|` | Number of points in cluster k |

---

### 3. Objective Function (WCSS / Inertia)

K-Means minimises the **Within-Cluster Sum of Squares (WCSS)**, also called **inertia**:

```
         K
J  =    Σ     Σ      ‖xᵢ − μₖ‖²
        k=1  xᵢ∈Cₖ
```

| Symbol | Meaning |
|---|---|
| `K` | Number of clusters |
| `Cₖ` | Set of points in cluster k |
| `μₖ` | Centroid of cluster k |
| `‖xᵢ − μₖ‖²` | Squared Euclidean distance from point to centroid |

**Goal:** Find cluster assignments and centroids that **minimise J**.

> ⚠️ K-Means finds a **local minimum** — not necessarily the global minimum. Results can vary with different initialisations.

---

## Choosing the Optimal K — Elbow Method

One of the biggest challenges in K-Means is selecting the right value of `K`. The **Elbow Method** is the most widely used graphical technique.

### How It Works

1. Run K-Means for a range of K values (e.g., K = 1 to 10).
2. Record the **WCSS (inertia)** for each K.
3. Plot K vs WCSS.
4. Look for the **"elbow" point** — where adding more clusters yields diminishing returns.

### Elbow Curve Shape

```
WCSS
(Inertia)
   │
   │\
   │ \
   │  \
   │   \
   │    \____
   │         \_________ ← Diminishing returns
   │
   └──────────────────── K
        1  2  3  4  5  6  7
             ^
           Elbow = Optimal K = 3
```

### Python — Elbow Method

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++',
                    n_init=10, max_iter=300, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o', linewidth=2, color='steelblue')
plt.axvline(x=3, color='red', linestyle='--', label='Optimal K = 3')
plt.title('Elbow Method — Optimal Number of Clusters')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

### Alternative Methods for Choosing K

| Method | Description | Best For |
|---|---|---|
| **Elbow Method** | Plot WCSS vs K; look for the bend | General-purpose, intuitive |
| **Silhouette Score** | Measures how well each point fits its cluster | Validating cluster quality |
| **Gap Statistic** | Compares WCSS to a random reference | Statistical rigour |
| **Davies-Bouldin Index** | Ratio of within-cluster to between-cluster distances | Compact, well-separated clusters |
| **Calinski-Harabasz Index** | Ratio of between-cluster to within-cluster dispersion | Large, well-defined clusters |
| **Domain Knowledge** | Prior business or research knowledge | When context defines K |

---

## Centroid Initialisation Strategies

Poor initialisation can trap K-Means in a bad local minimum. The choice of initial centroid placement significantly impacts the quality of the final clusters.

### 1. Random Initialisation

Randomly selects K data points as initial centroids.

```
Pros: Simple, fast
Cons: Can converge to poor local minima; sensitive to random seed
```

### 2. K-Means++ (Recommended ✅)

Smarter initialisation that spreads centroids apart:

```
Step 1: Pick the first centroid uniformly at random
Step 2: For each remaining centroid:
         Compute distance D(x) from each point to the nearest
         already-chosen centroid
         Choose the next centroid with probability ∝ D(x)²
         (points farther away are more likely to be chosen)
Step 3: Repeat Step 2 until K centroids are chosen
```

```
Benefits:
  ✅ Better initial spread
  ✅ Faster convergence
  ✅ More consistent results
  ✅ Lower final WCSS
  ✅ Default in scikit-learn (init='k-means++')
```

### 3. Manual / Domain-Driven Initialisation

Set centroids based on domain knowledge when group centres are approximately known.

```python
# Manual initialisation
initial_centers = np.array([[1.0, 2.0], [5.0, 5.0], [9.0, 1.0]])
kmeans = KMeans(n_clusters=3, init=initial_centers, n_init=1)
```

### Initialisation Comparison

```
                 Random Init          K-Means++ Init
Convergence:     Slower               Faster
Consistency:     Variable             More stable
WCSS Quality:    Higher (worse)       Lower (better)
Risk:            Local minima         Reduced risk
Sklearn Default: No                   Yes
```

---

## Python Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k=3, max_iter=300, tol=1e-4, random_state=None):
        self.k           = k
        self.max_iter    = max_iter
        self.tol         = tol
        self.random_state = random_state
        self.centroids   = None
        self.labels_     = None
        self.inertia_    = None

    def _euclidean(self, a, b):
        """Compute Euclidean distance between point a and point b."""
        return np.sqrt(np.sum((a - b) ** 2))

    def _init_centroids(self, X):
        """K-Means++ initialisation."""
        rng = np.random.default_rng(self.random_state)
        # Step 1: Pick first centroid randomly
        idx = rng.integers(0, len(X))
        centroids = [X[idx]]

        # Step 2: Pick remaining centroids with D² weighting
        for _ in range(1, self.k):
            dists = np.array([
                min(self._euclidean(x, c) ** 2 for c in centroids)
                for x in X
            ])
            probs = dists / dists.sum()
            cumulative = np.cumsum(probs)
            r = rng.random()
            for j, p in enumerate(cumulative):
                if r <= p:
                    centroids.append(X[j])
                    break
        return np.array(centroids)

    def _assign_clusters(self, X):
        """Assign each point to the nearest centroid."""
        labels = []
        for x in X:
            distances = [self._euclidean(x, c) for c in self.centroids]
            labels.append(np.argmin(distances))
        return np.array(labels)

    def _update_centroids(self, X, labels):
        """Recompute centroids as the mean of assigned points."""
        new_centroids = []
        for k in range(self.k):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                # Empty cluster: keep old centroid
                new_centroids.append(self.centroids[k])
        return np.array(new_centroids)

    def _compute_inertia(self, X, labels):
        """Compute Within-Cluster Sum of Squares (WCSS)."""
        wcss = 0
        for k in range(self.k):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - self.centroids[k]) ** 2)
        return wcss

    def fit(self, X):
        """Train the K-Means model on dataset X."""
        X = np.array(X)
        self.centroids = self._init_centroids(X)

        for iteration in range(self.max_iter):
            # Assignment step
            labels = self._assign_clusters(X)

            # Update step
            new_centroids = self._update_centroids(X, labels)

            # Check convergence
            shift = np.max(np.linalg.norm(new_centroids - self.centroids, axis=1))
            self.centroids = new_centroids

            if shift < self.tol:
                print(f"  Converged at iteration {iteration + 1}")
                break

        self.labels_  = self._assign_clusters(X)
        self.inertia_ = self._compute_inertia(X, self.labels_)
        return self

    def predict(self, X):
        """Predict cluster labels for new data points."""
        X = np.array(X)
        return self._assign_clusters(X)

    def fit_predict(self, X):
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_
```

### Usage

```python
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train custom K-Means
km = KMeans(k=3, max_iter=300, random_state=42)
labels = km.fit_predict(X_scaled)

print(f"Final WCSS (Inertia): {km.inertia_:.4f}")
print(f"Centroids:\n{km.centroids}")

# Plot result
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=40, alpha=0.7)
plt.scatter(km.centroids[:, 0], km.centroids[:, 1],
            marker='^', s=200, c='red', zorder=5, label='Centroids')
plt.title('Custom K-Means Clustering Result')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## Step-by-Step Implementation with NumPy & Matplotlib

This follows the exact approach from GeeksforGeeks — building K-Means piece by piece:

### Step 1: Import Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
```

### Step 2: Create Synthetic Dataset

```python
X, y = make_blobs(n_samples=500, n_features=2, centers=3, random_state=23)

plt.figure(figsize=(7, 5))
plt.grid(True)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.title('Raw Dataset (Unlabelled)')
plt.show()
```

### Step 3: Feature Scaling

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

### Step 4: Initialise Random Centroids

```python
k = 3
clusters = {}
np.random.seed(23)

for idx in range(k):
    center = 2 * (2 * np.random.random((X.shape[1],)) - 1)  # Values in [-2, 2]
    clusters[idx] = {'center': center, 'points': []}

print("Initial centroids:")
for i, c in clusters.items():
    print(f"  Cluster {i}: {c['center'].round(3)}")
```

### Step 5: Define Distance Function

```python
def distance(p1, p2):
    """Euclidean distance between two points."""
    return np.sqrt(np.sum((p1 - p2) ** 2))
```

### Step 6: Assign and Update Functions

```python
def assign_clusters(X, clusters, k):
    """Assign each data point to the nearest centroid."""
    for idx in range(X.shape[0]):
        curr_x = X[idx]
        dists  = [distance(curr_x, clusters[i]['center']) for i in range(k)]
        nearest = np.argmin(dists)
        clusters[nearest]['points'].append(curr_x)
    return clusters


def update_clusters(X, clusters, k):
    """Recompute each centroid as the mean of its assigned points."""
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            clusters[i]['center'] = points.mean(axis=0)
        clusters[i]['points'] = []          # Reset for next iteration
    return clusters
```

### Step 7: Predict Final Cluster Labels

```python
def pred_cluster(X, clusters, k):
    """Return cluster label for each data point."""
    pred = []
    for i in range(X.shape[0]):
        dists = [distance(X[i], clusters[j]['center']) for j in range(k)]
        pred.append(np.argmin(dists))
    return pred
```

### Step 8: Run K-Means Iterations

```python
n_iterations = 10

for it in range(n_iterations):
    clusters = assign_clusters(X, clusters, k)
    clusters = update_clusters(X, clusters, k)

pred = pred_cluster(X, clusters, k)
```

### Step 9: Plot Final Result

```python
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=pred, cmap='Set1', alpha=0.7, s=30)
plt.grid(True)

for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0], center[1], marker='^', s=250,
                c='black', zorder=5, label=f'Centroid {i}')

plt.title('K-Means Clustering Result (From Scratch)')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## Implementation with Scikit-Learn

### Standard Classification

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Generate data
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)

# Scale features (critical for KMeans!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build model
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',       # Smarter initialisation
    n_init=10,              # Run 10 times, pick best WCSS
    max_iter=300,           # Max iterations per run
    tol=1e-4,               # Convergence tolerance
    random_state=42
)

kmeans.fit(X_scaled)
labels     = kmeans.labels_
centroids  = kmeans.cluster_centers_
inertia    = kmeans.inertia_

print(f"WCSS (Inertia)     : {inertia:.4f}")
print(f"Silhouette Score   : {silhouette_score(X_scaled, labels):.4f}")
print(f"Iterations         : {kmeans.n_iter_}")
print(f"Centroids shape    : {centroids.shape}")

# Visualise
plt.figure(figsize=(9, 6))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
                      c=labels, cmap='tab10', s=40, alpha=0.7)
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='*', s=400, c='black', zorder=10, label='Centroids')
plt.title('K-Means Clustering (Scikit-Learn)')
plt.colorbar(scatter, label='Cluster Label')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

### Predicting New Data Points

```python
# Predict cluster for new points
new_points = np.array([[0.5, 1.2], [-1.0, -2.5], [3.0, 3.0]])
new_scaled = scaler.transform(new_points)
predictions = kmeans.predict(new_scaled)

for point, label in zip(new_points, predictions):
    print(f"Point {point} → Cluster {label}")
```

---

## Evaluation Metrics for Clustering

Unlike supervised learning, clustering has no ground-truth labels to compare against (in practice). These metrics help evaluate cluster quality without labels:

### 1. Silhouette Score

Measures how similar each point is to its own cluster vs other clusters.

```
s(i) = (b(i) − a(i)) / max(a(i), b(i))
```

| Symbol | Meaning |
|---|---|
| `a(i)` | Mean distance from point i to all other points in its cluster |
| `b(i)` | Mean distance from point i to all points in the nearest other cluster |

```
Silhouette Score Range:
  +1.0  →  Point is well clustered (far from neighbouring clusters)
   0.0  →  Point is on the boundary between clusters
  −1.0  →  Point is likely in the wrong cluster
```

```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {score:.4f}")
```

### 2. Davies-Bouldin Index

Average similarity ratio of each cluster to its most similar cluster. **Lower is better.**

```python
from sklearn.metrics import davies_bouldin_score
db_score = davies_bouldin_score(X_scaled, labels)
print(f"Davies-Bouldin Index: {db_score:.4f}")
```

### 3. Calinski-Harabasz Index (Variance Ratio)

Ratio of between-cluster dispersion to within-cluster dispersion. **Higher is better.**

```python
from sklearn.metrics import calinski_harabasz_score
ch_score = calinski_harabasz_score(X_scaled, labels)
print(f"Calinski-Harabasz Index: {ch_score:.4f}")
```

### 4. Inertia (WCSS) — Model's Own Metric

The total within-cluster sum of squares. **Lower is better** but decreases with more clusters.

```python
print(f"Inertia: {kmeans.inertia_:.4f}")
```

### Metrics Comparison

| Metric | Range | Optimal | Formula Basis |
|---|---|---|---|
| **Silhouette Score** | −1 to +1 | Higher is better (+1) | Inter vs intra cluster distance |
| **Davies-Bouldin** | 0 to ∞ | Lower is better (0) | Cluster similarity ratio |
| **Calinski-Harabasz** | 0 to ∞ | Higher is better | Variance ratio |
| **Inertia (WCSS)** | 0 to ∞ | Lower is better | Sum of squared distances |

---

## Hyperparameter Tuning

### Key Parameters in Scikit-Learn's KMeans

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_clusters` | int | `8` | Number of clusters K — most critical parameter |
| `init` | str/array | `'k-means++'` | Centroid initialisation: `'k-means++'` or `'random'` |
| `n_init` | int | `10` | Number of times to run with different seeds; best result kept |
| `max_iter` | int | `300` | Maximum iterations per run |
| `tol` | float | `1e-4` | Convergence threshold for centroid shift |
| `algorithm` | str | `'lloyd'` | Algorithm: `'lloyd'` (default) or `'elkan'` (faster for dense data) |
| `random_state` | int | `None` | Seed for reproducibility |

### Best Practices for Tuning

```python
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score

param_grid = {
    'n_clusters': [2, 3, 4, 5, 6],
    'init':       ['k-means++', 'random'],
    'n_init':     [10, 20],
}

best_score  = -1
best_params = {}

for params in ParameterGrid(param_grid):
    km = KMeans(**params, max_iter=300, random_state=42)
    km.fit(X_scaled)
    score = silhouette_score(X_scaled, km.labels_)
    if score > best_score:
        best_score  = score
        best_params = params

print(f"Best Params : {best_params}")
print(f"Best Score  : {best_score:.4f}")
```

---

## Variants of K-Means

| Variant | Key Difference | Best Use Case |
|---|---|---|
| **K-Means** | Classic; minimises WCSS via EM iterations | General-purpose clustering |
| **K-Means++** | Smarter initialisation (D² weighting) | Better convergence; sklearn default |
| **Mini-Batch K-Means** | Uses random mini-batches per iteration | Very large datasets (faster) |
| **K-Medoids (PAM)** | Uses actual data points as centres | Robust to outliers |
| **Fuzzy C-Means** | Points belong to clusters with probability | Soft/overlapping boundaries |
| **Bisecting K-Means** | Hierarchically splits largest cluster | Hierarchical + flat hybrid |
| **Spherical K-Means** | Cosine similarity instead of Euclidean | Text / high-dimensional data |
| **Kernel K-Means** | Projects data to higher dimensions via kernels | Non-spherical cluster shapes |

### Mini-Batch K-Means (for Large Datasets)

```python
from sklearn.cluster import MiniBatchKMeans

mb_kmeans = MiniBatchKMeans(
    n_clusters=4,
    batch_size=100,        # Process 100 random points per iteration
    n_init=3,
    random_state=42
)
mb_kmeans.fit(X_scaled)
print(f"Mini-Batch Inertia: {mb_kmeans.inertia_:.4f}")
```

---

## K-Means vs Other Clustering Algorithms

| Feature | K-Means | DBSCAN | Hierarchical | Gaussian Mixture |
|---|---|---|---|---|
| **Cluster Shape** | Spherical only | Any shape | Any shape | Elliptical |
| **Needs K** | ✅ Yes | ❌ No | ❌ No (dendrogram) | ✅ Yes |
| **Handles Outliers** | ❌ Sensitive | ✅ Flags as noise | ❌ Sensitive | ⚠️ Partial |
| **Scalability** | ✅ Good | ⚠️ Moderate | ❌ Poor (O(n²)) | ⚠️ Moderate |
| **Soft Assignments** | ❌ Hard only | ❌ Hard only | ❌ Hard only | ✅ Probabilistic |
| **Non-Spherical** | ❌ Struggles | ✅ Excellent | ✅ Good | ✅ Good |
| **Speed** | ⚡ Fast | Moderate | Slow | Moderate |
| **Interpretability** | ✅ High | Moderate | ✅ Visual (tree) | ⚠️ Medium |

---

## Applications

K-Means is applied widely across industries and research domains:

| Domain | Application |
|---|---|
| 🛒 **E-Commerce & Retail** | Customer segmentation by purchase history, RFM analysis |
| 🖼️ **Image Processing** | Image compression by grouping similar pixel colours |
| 🕵️ **Anomaly Detection** | Detecting outliers as points far from all centroids |
| 📰 **NLP & Text Mining** | Grouping similar news articles, reviews, or documents |
| 🏥 **Healthcare** | Patient grouping by symptoms for personalised treatment |
| 📍 **Geospatial** | Identifying geographic hotspots, delivery zone planning |
| 🎬 **Recommendation Systems** | Grouping users with similar preferences |
| 🧬 **Bioinformatics** | Gene expression profiling and DNA sequencing clusters |
| 📡 **Telecommunications** | Network traffic pattern analysis |
| 🏦 **Finance** | Market segmentation, fraud pattern grouping |

---

## Advantages

- ✅ **Simple and intuitive** — easy to understand, implement, and explain to stakeholders.
- ✅ **Computationally efficient** — scales well to large datasets; time complexity O(n·K·I·d).
- ✅ **Guarantees convergence** — always converges to a (local) minimum in finite iterations.
- ✅ **Works well on spherical clusters** — excels when data naturally forms round, well-separated clusters.
- ✅ **Flexible K** — can group data into any desired number of clusters.
- ✅ **Feature importance** — centroid coordinates reveal the "average" characteristics of each group.
- ✅ **Unsupervised** — requires no labelled data.

---

## Disadvantages & Challenges

- ❌ **Must specify K in advance** — choosing the wrong K leads to meaningless clusters; finding the right K requires additional analysis (Elbow, Silhouette).
- ❌ **Sensitive to initial centroids** — random initialisation can lead to poor local minima; mitigated by K-Means++ and multiple runs.
- ❌ **Assumes spherical, equal-sized clusters** — performs poorly when clusters are elongated, irregular, or have different densities/sizes.
- ❌ **Sensitive to outliers** — a single extreme point can pull a centroid far from the true cluster centre.
- ❌ **Sensitive to feature scale** — features with larger ranges dominate; always normalise with `StandardScaler` or `MinMaxScaler`.
- ❌ **Finds local minimum** — not guaranteed to find the global optimal solution.
- ❌ **Poor on high-dimensional data** — Euclidean distance loses meaning in very high dimensions (curse of dimensionality).
- ❌ **Hard assignments only** — every point belongs to exactly one cluster; no support for overlapping or probabilistic membership.

---

## K-Means vs Supervised Algorithms

| Aspect | K-Means (Unsupervised) | KNN / SVM / DT (Supervised) |
|---|---|---|
| **Labels Required** | ❌ No | ✅ Yes |
| **Goal** | Find natural groups | Predict class / value |
| **Output** | Cluster IDs (0, 1, 2…) | Class labels / predictions |
| **Evaluation** | Silhouette, WCSS, DB Index | Accuracy, F1, AUC-ROC |
| **Training** | Iterative (EM loop) | Fits to labelled examples |
| **Use Case** | Exploration, segmentation | Classification, regression |
| **Feature Scaling** | Required | Required (KNN, SVM) |

---

## Summary Cheat Sheet

```
┌─────────────────────────────────────────────────────────────────────┐
│                  K-Means Clustering Quick Reference                 │
├──────────────────────────┬──────────────────────────────────────────┤
│ Algorithm Type           │ Unsupervised, Iterative (EM)            │
│ Task                     │ Clustering / Segmentation               │
│ Key Parameter            │ K (number of clusters)                  │
│ Distance Metric          │ Euclidean Distance (default)            │
│ Centroid Update          │ Mean of assigned points                 │
│ Objective Function       │ Minimise WCSS (Inertia)                │
│ Initialisation (best)    │ K-Means++ (sklearn default)            │
│ Choosing K               │ Elbow Method, Silhouette Score          │
│ Convergence              │ When centroids stop moving (< tol)      │
│ Evaluation (no labels)   │ Silhouette, Davies-Bouldin, CH Index    │
│ Feature Scaling          │ REQUIRED (StandardScaler)               │
│ Handles Outliers?        │ ❌ Sensitive — use K-Medoids instead    │
│ Non-Spherical Clusters?  │ ❌ Struggles — use DBSCAN instead       │
│ Large Datasets?          │ Use Mini-Batch K-Means                  │
│ Time Complexity          │ O(n · K · I · d)                       │
│ Sklearn Class            │ sklearn.cluster.KMeans                 │
│ Key Strength             │ Simple, fast, scalable                  │
│ Key Weakness             │ Must specify K; sensitive to init       │
└──────────────────────────┴──────────────────────────────────────────┘

 n = samples  |  K = clusters  |  I = iterations  |  d = features
```

---

## References

- [GeeksforGeeks — K-Means Clustering Introduction](https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/)
- [GeeksforGeeks — Elbow Method for Optimal K](https://www.geeksforgeeks.org/machine-learning/elbow-method-for-optimal-value-of-k-in-kmeans/)
- [GeeksforGeeks — K-Means++ Algorithm](https://www.geeksforgeeks.org/machine-learning/ml-k-means-algorithm/)
- [Scikit-Learn KMeans Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Scikit-Learn Clustering User Guide](https://scikit-learn.org/stable/modules/clustering.html)

