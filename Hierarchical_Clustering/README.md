# 🌿 Hierarchical Clustering Algorithm

> A comprehensive guide to understanding, implementing, and applying Hierarchical Clustering — a powerful **unsupervised machine learning** technique that builds a tree-like structure of nested clusters without requiring the number of clusters to be specified in advance.

---

## 📖 Table of Contents

- [Overview](#overview)
- [Key Concepts & Terminology](#key-concepts--terminology)
- [The Dendrogram](#the-dendrogram)
- [Types of Hierarchical Clustering](#types-of-hierarchical-clustering)
  - [Agglomerative Clustering (Bottom-Up)](#1-agglomerative-clustering-bottom-up)
  - [Divisive Clustering (Top-Down)](#2-divisive-clustering-top-down)
  - [Agglomerative vs Divisive — Comparison](#agglomerative-vs-divisive--comparison)
- [Computing the Distance Matrix](#computing-the-distance-matrix)
- [Linkage Methods (Inter-Cluster Distance)](#linkage-methods-inter-cluster-distance)
- [Mathematical Foundation](#mathematical-foundation)
  - [Euclidean Distance](#1-euclidean-distance)
  - [Manhattan Distance](#2-manhattan-distance)
  - [Ward's Method Formula](#3-wards-method-formula)
- [Step-by-Step Worked Example](#step-by-step-worked-example)
- [How to Read a Dendrogram](#how-to-read-a-dendrogram)
- [Cutting the Dendrogram — Choosing Number of Clusters](#cutting-the-dendrogram--choosing-number-of-clusters)
- [Evaluation Metrics](#evaluation-metrics)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Hierarchical Clustering vs K-Means](#hierarchical-clustering-vs-k-means)
- [Hierarchical vs Other Clustering Algorithms](#hierarchical-vs-other-clustering-algorithms)
- [Applications](#applications)
- [Advantages](#advantages)
- [Disadvantages](#disadvantages)
- [Summary Cheat Sheet](#summary-cheat-sheet)
- [References](#references)

---

## Overview

**Hierarchical Clustering** is an **unsupervised machine learning** algorithm that builds a hierarchy of clusters arranged in a tree-like structure called a **dendrogram**. Unlike K-Means, it does **not require the number of clusters to be specified in advance** — the analyst can choose the number of clusters after inspecting the dendrogram.

### Key Characteristics

| Property | Description |
|---|---|
| **Type** | Unsupervised Learning |
| **Task** | Clustering / Grouping / Segmentation |
| **Requires K?** | ❌ No — determined post-hoc from the dendrogram |
| **Requires Labels?** | ❌ No — works on unlabelled data |
| **Output** | Dendrogram + flat cluster assignments |
| **Approach** | Bottom-up (Agglomerative) or Top-down (Divisive) |
| **Distance Metric** | Euclidean, Manhattan, Cosine, etc. |
| **Linkage Criterion** | Single, Complete, Average, Ward's |

### Core Intuition — Fruit Weight Analogy

Suppose we have four fruits and want to group them by weight:

```
  Fruit       Weight
  ─────────   ──────
  Apple        100g
  Banana       120g
  Cherry        50g
  Grape         30g
```

Hierarchical Clustering proceeds as follows:

```
Step 1: Start — each fruit is its own cluster
        [Apple] [Banana] [Cherry] [Grape]

Step 2: Merge the two closest →  Grape(30g) + Cherry(50g)
        [Apple] [Banana] [Cherry + Grape]

Step 3: Merge the next closest → Apple(100g) + Banana(120g)
        [Apple + Banana] [Cherry + Grape]

Step 4: Merge remaining two groups into one
        [Apple + Banana + Cherry + Grape]
```

The result is a complete tree showing how items relate to each other — even before deciding how many clusters you need.

---

## Key Concepts & Terminology

### 🌳 Dendrogram
A **tree-shaped diagram** that records the sequence of merges (or splits) performed by the hierarchical clustering algorithm. Each leaf node is a data point; internal nodes represent cluster merges; the root represents one giant cluster containing all data.

### 📏 Distance Matrix
A symmetric matrix storing **pairwise distances** between all data points (or clusters). It drives the decision of which clusters to merge next.

### 🔗 Linkage
The criterion used to **measure distance between two clusters** (as opposed to two individual points). Different linkage methods lead to different clustering structures.

### ✂️ Cut (Threshold)
A **horizontal line** drawn across the dendrogram to determine the final number of clusters. Branches cut by this line define distinct clusters.

### 📐 Merge Height
The **height (y-axis value)** at which two clusters are joined in the dendrogram. Higher merge height = greater dissimilarity between the merging clusters.

### 🔁 Agglomeration
The **bottom-up** process of successively merging the two most similar clusters, starting from individual points.

### ✂️ Division
The **top-down** process of recursively splitting one large cluster into smaller ones, starting from the whole dataset.

---

## The Dendrogram

A dendrogram is the signature output of hierarchical clustering. Understanding it is central to interpreting results.

```
Distance
(Height)
   │
 5 │              ┌──────────────────────┐
   │              │                      │
 4 │        ┌─────┘              ┌───────┘
   │        │                   │
 3 │   ┌────┘              ┌────┘
   │   │                   │
 2 │   │         ┌─────────┘
   │   │         │
 1 │   │    ┌────┘
   │   │    │
 0 │   │    │
   └───┴────┴────┴────┴─── Data Points
       P    Q    R    S    T
```

### How to Read the Dendrogram

| Visual Element | What It Means |
|---|---|
| **Leaf nodes (bottom)** | Individual data points — each starts as its own cluster |
| **Horizontal bars** | Merger events — two clusters joining into one |
| **Height of a bar** | Distance/dissimilarity between the two clusters being merged |
| **Low merge height** | Points/clusters are very similar (close together) |
| **High merge height** | Points/clusters are very different (far apart) |
| **Horizontal cut line** | Determines how many final clusters to extract |
| **Number of vertical lines cut** | = Number of resulting clusters |

### Dendrogram for the Fruit Example

```
Dissimilarity
     │
  90 │                    ┌────────────────────────────┐
     │                    │                            │
  50 │          ┌─────────┘                  ┌─────────┘
     │          │                            │
  20 │     ┌────┘                   ┌────────┘
     │     │                        │
   0 │     │                        │
     └─────┴────────────────────────┴──────────────────
         [Grape]  [Cherry]      [Apple]   [Banana]
```

---

## Types of Hierarchical Clustering

### 1. Agglomerative Clustering (Bottom-Up)

Also known as **HAC (Hierarchical Agglomerative Clustering)** or the **bottom-up approach**. It starts with each data point as an individual cluster and iteratively merges the two most similar clusters until only one cluster remains.

```
Start:   [A] [B] [C] [D] [E]     ← N clusters (each point alone)
Iter 1:  [A+B] [C] [D] [E]       ← merge closest pair
Iter 2:  [A+B] [C+D] [E]         ← merge next closest pair
Iter 3:  [A+B+C+D] [E]           ← continue merging
Iter 4:  [A+B+C+D+E]             ← one final cluster
```

#### Agglomerative Workflow

```
┌────────────────────────────────────────────────────────────────────┐
│             Agglomerative Clustering (HAC) Flow                    │
└────────────────────────────────────────────────────────────────────┘

Step 1: Initialise
   └─► Each of N data points is its own cluster
       → Start with N clusters

Step 2: Compute Distance Matrix
   └─► Calculate pairwise distance between every cluster
       → N × N symmetric matrix D

Step 3: Find Minimum Distance
   └─► Identify the two clusters (Cᵢ, Cⱼ) with the smallest d(Cᵢ, Cⱼ)

Step 4: Merge Clusters
   └─► Combine Cᵢ and Cⱼ into a single new cluster Cᵢⱼ
       → Record the merge height (distance) in the dendrogram

Step 5: Update Distance Matrix
   └─► Remove rows/columns for Cᵢ and Cⱼ
   └─► Add new row/column for Cᵢⱼ using the chosen linkage criterion

Step 6: Repeat
   └─► Return to Step 3 until only ONE cluster remains

Step 7: Extract Clusters
   └─► Cut the dendrogram at the desired threshold
       → Obtain K flat clusters
```

---

### 2. Divisive Clustering (Top-Down)

Also known as the **top-down approach**. It begins with all data points in a single cluster and recursively **splits** clusters until each point is its own singleton cluster.

```
Start:  [A+B+C+D+E]              ← 1 cluster (all points together)
Iter 1: [A+B] [C+D+E]            ← split most dissimilar subgroup
Iter 2: [A+B] [C+D] [E]          ← split again
Iter 3: [A] [B] [C+D] [E]        ← continue splitting
Iter 4: [A] [B] [C] [D] [E]      ← N singleton clusters
```

#### Divisive Workflow

```
┌────────────────────────────────────────────────────────────────────┐
│               Divisive Clustering (DIANA) Flow                     │
└────────────────────────────────────────────────────────────────────┘

Step 1: Initialise
   └─► Place ALL data points in a single cluster

Step 2: Select Cluster to Split
   └─► Choose the cluster with the highest internal dissimilarity
       (usually the largest or most spread-out cluster)

Step 3: Split the Cluster
   └─► Find the two most dissimilar points in the selected cluster
   └─► Divide data into two sub-clusters based on proximity to those points

Step 4: Repeat
   └─► Apply Step 2 & 3 recursively to each new sub-cluster
   └─► Continue until each point is in its own singleton cluster
       OR the desired number of clusters K is reached

Step 5: Extract Clusters
   └─► Stop at any level of the recursion
       → Obtain K flat clusters
```

---

### Agglomerative vs Divisive — Comparison

```
              Agglomerative              Divisive
              (Bottom-Up)                (Top-Down)
─────────────────────────────────────────────────────────
Start from   N individual singletons     1 giant cluster
Direction    Merge upward  ↑             Split downward ↓
Complexity   O(n² log n)                 O(2ⁿ) — expensive
Common Use   Most widely used            Less common in practice
Algorithm    HAC / sklearn               DIANA, Bisecting K-Means
Sklearn?     ✅ AgglomerativeClustering  ⚠️  Not built-in (use KMeans)
Dendrogram   Built bottom-up            Built top-down
Best For     Small-medium datasets       Well-separated global splits
```

---

## Computing the Distance Matrix

At each merge step, hierarchical clustering needs to know the **distance between every pair of clusters**. The distance matrix is updated after each merge.

### Distance Matrix Example (4 points: A, B, C, D)

```
Initial distances (Euclidean between points):

        A     B     C     D
   A  [ 0    1.2   3.5   4.1 ]
   B  [ 1.2   0    2.8   3.6 ]
   C  [ 3.5  2.8    0    0.9 ]
   D  [ 4.1  3.6   0.9    0  ]

Minimum distance: C–D = 0.9 → Merge C and D first

Updated distances after merging C+D:

        A     B    (C+D)
   A  [ 0    1.2   3.5  ]    ← using single linkage: min(A-C, A-D)
   B  [ 1.2   0    2.8  ]
 (C+D)[ 3.5  2.8    0   ]

Next minimum: A–B = 1.2 → Merge A and B

Final step: Merge (A+B) and (C+D)
```

---

## Linkage Methods (Inter-Cluster Distance)

The **linkage criterion** determines how the distance between two **clusters** is calculated (not just two points). This choice dramatically affects the shape of the resulting dendrogram and clusters.

### 1. 🔹 Single Linkage (MIN)

Distance = **minimum** distance between any two points from the two clusters.

```
d(Cᵢ, Cⱼ) = min { d(xₐ, xᵦ) : xₐ ∈ Cᵢ, xᵦ ∈ Cⱼ }
```

```
Cluster A:  o  o  o
                    \
                     ←── min distance
                    /
Cluster B:  x  x  x
```

| Pros | Cons |
|---|---|
| Detects elongated, chain-like clusters | Prone to **chaining effect** — loosely connected points can bridge clusters |
| Handles non-spherical shapes | Sensitive to noise and outliers |

---

### 2. 🔹 Complete Linkage (MAX)

Distance = **maximum** distance between any two points from the two clusters.

```
d(Cᵢ, Cⱼ) = max { d(xₐ, xᵦ) : xₐ ∈ Cᵢ, xᵦ ∈ Cⱼ }
```

```
Cluster A:  o  o  [o]
                       \
                        ←── max distance
                       /
Cluster B:  [x]  x  x
```

| Pros | Cons |
|---|---|
| Produces compact, well-separated clusters | Biased toward spherical clusters |
| Less sensitive to noise | Sensitive to outliers (single far point inflates distance) |

---

### 3. 🔹 Average Linkage (UPGMA)

Distance = **average** of all pairwise distances between points in the two clusters.

```
           1          Σ
d(Cᵢ, Cⱼ) = ──────── ·    d(xₐ, xᵦ)
           |Cᵢ|·|Cⱼ|   xₐ∈Cᵢ, xᵦ∈Cⱼ
```

| Pros | Cons |
|---|---|
| Good balance between single and complete | More computationally expensive |
| Less sensitive to outliers than MAX | Assumes similar cluster sizes |
| Often a reliable default choice | Slightly harder to interpret |

---

### 4. 🔹 Ward's Method (Minimum Variance)

Merges the two clusters that produce the **smallest increase in total within-cluster variance** (sum of squared deviations from centroid). Most widely used in practice.

```
d(Cᵢ, Cⱼ) = Δ Variance = Σ‖x − μᵢⱼ‖² − Σ‖x − μᵢ‖² − Σ‖x − μⱼ‖²
```

| Pros | Cons |
|---|---|
| Produces compact, equal-sized spherical clusters | Assumes similarly sized clusters |
| Minimises within-cluster variance | Sensitive to outliers |
| Most powerful linkage — sklearn default | Only works with Euclidean distance |
| Gives clean, interpretable dendrograms | |

---

### Linkage Methods Comparison

```
                Single      Complete    Average     Ward's
                (MIN)        (MAX)      (UPGMA)    (Variance)
────────────────────────────────────────────────────────────────
Cluster Shape  Elongated   Compact     Balanced    Spherical
Outlier Risk   High        High        Medium      Medium
Chaining       Yes         No          Rare        No
Best Default?  No          Sometimes   Often       ✅ Yes
Sklearn Name   'single'   'complete'  'average'   'ward'
```

---

## Mathematical Foundation

### 1. Euclidean Distance

The most common distance metric — straight-line distance between two points in n-dimensional space:

```
d(xᵢ, xⱼ) = √[ Σₖ (xᵢₖ − xⱼₖ)² ]
```

**Best for:** Continuous numeric features with similar scales.

---

### 2. Manhattan Distance

Sum of absolute differences along each dimension:

```
d(xᵢ, xⱼ) = Σₖ |xᵢₖ − xⱼₖ|
```

**Best for:** Grid-like data, high-dimensional spaces.

---

### 3. Ward's Method Formula

Ward's linkage computes the increase in WCSS (Within-Cluster Sum of Squares) when merging clusters Cᵢ and Cⱼ:

```
                  nᵢ · nⱼ
d(Cᵢ, Cⱼ) = √(  ─────────  · ‖μᵢ − μⱼ‖² )
                  nᵢ + nⱼ
```

| Symbol | Meaning |
|---|---|
| `nᵢ` | Number of points in cluster i |
| `nⱼ` | Number of points in cluster j |
| `μᵢ` | Centroid (mean) of cluster i |
| `μⱼ` | Centroid (mean) of cluster j |
| `‖μᵢ − μⱼ‖²` | Squared Euclidean distance between centroids |

---

## Step-by-Step Worked Example

### Dataset: 5 points in 1D

```
Points: A=1, B=2, C=5, D=8, E=9
```

### Step 1 — Initial Distance Matrix

```
     A    B    C    D    E
A  [ 0    1    4    7    8 ]
B  [ 1    0    3    6    7 ]
C  [ 4    3    0    3    4 ]
D  [ 7    6    3    0    1 ]
E  [ 8    7    4    1    0 ]
```

### Step 2 — Find & Merge Minimum Distance

```
Minimum: A–B = 1  →  Merge A and B  →  New cluster: {A,B}
```

### Step 3 — Update Distance Matrix (Single Linkage)

```
        {A,B}   C    D    E
{A,B} [   0     3    6    7  ]
C     [   3     0    3    4  ]
D     [   6     3    0    1  ]
E     [   7     4    1    0  ]
```

### Step 4 — Merge Next Minimum

```
Minimum: D–E = 1  →  Merge D and E  →  New cluster: {D,E}

        {A,B}   C   {D,E}
{A,B} [   0     3     6  ]
C     [   3     0     3  ]
{D,E} [   6     3     0  ]
```

### Step 5 — Continue

```
Minimum: C–{D,E} = 3  AND  C–{A,B} = 3  →  Tie; merge C–{A,B} first

        {A,B,C}   {D,E}
{A,B,C}[   0        3  ]
{D,E}  [   3        0  ]

→  Final merge: {A,B,C} + {D,E} = {A,B,C,D,E}
```

### Resulting Dendrogram

```
Distance
   │
 6 │                 ┌──────────────────────────┐
   │                 │                          │
 3 │    ┌────────────┘                ┌─────────┘
   │    │                             │
 1 │ ┌──┘                         ┌──┘
   │ │                             │
 0 │ │                             │
   └─┴──────────────────────────────┴─────────────
     A   B          C              D   E
```

**Cutting at height = 3:** Two clusters → `{A, B, C}` and `{D, E}`
**Cutting at height = 1:** Three clusters → `{A, B}`, `{C}`, `{D, E}`

---

## How to Read a Dendrogram

```
┌─────────────────────────────────────────────────────────────┐
│                 Dendrogram Reading Guide                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Height                                                     │
│    │                                                        │
│  High │      ┌──────────────────────────────┐              │
│       │      │    ← Large gap here!          │              │
│       │      │   (best cut point)            │              │
│       │      │                               │              │
│  Low  │  ┌───┘            ┌──────────────────┘             │
│       │  │                │                                 │
│     0 │  │    │      │    │    │       │                   │
│       └──┴────┴──────┴────┴────┴───────┴────               │
│          P    Q      R    S    T                            │
│                                                             │
│  Rules:                                                     │
│  1. Count VERTICAL lines crossed by a horizontal cut        │
│     = Number of clusters                                    │
│  2. Cut where the LONGEST vertical line exists              │
│     = Most natural number of clusters                       │
│  3. Low horizontal bar = similar items merged early         │
│  4. High horizontal bar = dissimilar items merged last      │
└─────────────────────────────────────────────────────────────┘
```

---

## Cutting the Dendrogram — Choosing Number of Clusters

The dendrogram encodes all possible clusterings. Choosing where to cut determines the final number of clusters.

### Method 1 — Largest Vertical Gap (Most Common)

```
Find the longest vertical line that no horizontal line crosses.
Draw a horizontal cut through it.
Count the number of vertical lines the cut intersects → K clusters.
```

### Method 2 — Inconsistency Coefficient

Measures how different a merge height is relative to its children:

```python
from scipy.cluster.hierarchy import inconsistent
from scipy.cluster.hierarchy import linkage

Z = linkage(X, method='ward')
inc = inconsistent(Z, d=5)   # d = depth to look back
print(inc)
# Rows with high last-column values = natural cut points
```

### Method 3 — Domain Knowledge

Use business understanding: "We want 3 customer segments" → cut at the height that yields 3 clusters.

### Method 4 — Silhouette Score

Try multiple cuts and select the K that maximises the Silhouette Score.

```python
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster

best_k, best_score = 2, -1
for k in range(2, 10):
    labels = fcluster(Z, k, criterion='maxclust')
    score  = silhouette_score(X, labels)
    if score > best_score:
        best_k, best_score = k, score

print(f"Best K: {best_k}, Silhouette: {best_score:.4f}")
```

---


## Evaluation Metrics

Since hierarchical clustering is unsupervised, we use internal metrics to assess cluster quality:

### 1. Silhouette Score

Measures how similar each point is to its own cluster versus other clusters.

```
s(i) = (b(i) − a(i)) / max(a(i), b(i))
```

| Range | Interpretation |
|---|---|
| +1.0 | Point is perfectly clustered |
| 0.0 | Point lies on a cluster boundary |
| −1.0 | Point is likely in the wrong cluster |

```python
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster

Z      = linkage(X_scaled, method='ward')
labels = fcluster(Z, 3, criterion='maxclust')
score  = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {score:.4f}")
```

### 2. Cophenetic Correlation Coefficient

Measures how faithfully the dendrogram preserves the original pairwise distances. A value closer to 1 means the dendrogram is a good representation.

```python
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

Z            = linkage(X_scaled, method='ward')
c, coph_dist = cophenet(Z, pdist(X_scaled))
print(f"Cophenetic Correlation: {c:.4f}")
# Values > 0.75 are generally considered good
```

### 3. Davies-Bouldin Index

Ratio of within-cluster scatter to between-cluster separation. **Lower is better.**

```python
from sklearn.metrics import davies_bouldin_score
db = davies_bouldin_score(X_scaled, labels)
print(f"Davies-Bouldin Index: {db:.4f}")
```

### 4. Calinski-Harabasz Index

Ratio of between-cluster dispersion to within-cluster dispersion. **Higher is better.**

```python
from sklearn.metrics import calinski_harabasz_score
ch = calinski_harabasz_score(X_scaled, labels)
print(f"Calinski-Harabasz Index: {ch:.4f}")
```

### Metrics Summary

| Metric | Optimal | Measures |
|---|---|---|
| **Silhouette Score** | Closer to +1 | How well-separated clusters are |
| **Cophenetic Correlation** | Closer to +1 | Dendrogram fidelity to data |
| **Davies-Bouldin** | Closer to 0 | Cluster compactness & separation |
| **Calinski-Harabasz** | Higher is better | Dispersion ratio |

---

## Hyperparameter Tuning

### Key Parameters — sklearn AgglomerativeClustering

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_clusters` | int / None | `2` | Number of clusters to extract; `None` when using `distance_threshold` |
| `metric` | str | `'euclidean'` | Distance metric: `'euclidean'`, `'manhattan'`, `'cosine'`, `'l1'`, `'l2'` |
| `linkage` | str | `'ward'` | Linkage criterion: `'ward'`, `'complete'`, `'average'`, `'single'` |
| `distance_threshold` | float / None | `None` | Cut the tree at this threshold; `n_clusters` must be `None` |
| `compute_distances` | bool | `False` | Must be `True` to plot the full dendrogram from sklearn |
| `compute_full_tree` | str / bool | `'auto'` | Build the full tree even if early stopping is possible |

### Grid Search Over Linkage Methods and K

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

linkages = ['ward', 'complete', 'average', 'single']
k_values = range(2, 8)
best_score  = -1
best_config = {}

for linkage in linkages:
    for k in k_values:
        # Ward only works with Euclidean
        metric = 'euclidean' if linkage == 'ward' else 'euclidean'
        try:
            agg = AgglomerativeClustering(
                n_clusters=k, linkage=linkage, metric=metric
            )
            labels = agg.fit_predict(X_scaled)
            score  = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score  = score
                best_config = {'n_clusters': k, 'linkage': linkage, 'metric': metric}
        except Exception:
            pass

print(f"Best Config : {best_config}")
print(f"Best Score  : {best_score:.4f}")
```

---

## Hierarchical Clustering vs K-Means

| Feature | Hierarchical Clustering | K-Means |
|---|---|---|
| **Requires K upfront?** | ❌ No — chosen from dendrogram | ✅ Yes — must specify before training |
| **Deterministic?** | ✅ Yes — same result every run | ❌ No — depends on random init |
| **Cluster Shape** | Any shape (linkage-dependent) | Spherical only |
| **Dendrogram** | ✅ Produces visual hierarchy | ❌ None |
| **Scalability** | ❌ O(n² log n) — slow on large data | ✅ O(n·K·I) — fast on large data |
| **Memory** | ❌ High (stores distance matrix) | ✅ Low (only centroids) |
| **Handles Outliers** | ⚠️ Sensitive (linkage-dependent) | ❌ Very sensitive |
| **Interpretability** | ✅ Very high — dendrogram | ✅ High — centroids |
| **Best Dataset Size** | Small to medium (< 10K) | Medium to large (10K+) |
| **Reversible?** | ❌ Merges cannot be undone | ✅ Assignments change each iter |

---

## Hierarchical vs Other Clustering Algorithms

| Feature | Hierarchical | K-Means | DBSCAN | GMM |
|---|---|---|---|---|
| **Requires K?** | ❌ (post-hoc) | ✅ | ❌ | ✅ |
| **Cluster Shape** | Any | Spherical | Any | Elliptical |
| **Noise Handling** | Poor | Poor | ✅ Excellent | Moderate |
| **Scalability** | Poor O(n²) | Good | Moderate | Moderate |
| **Dendrogram** | ✅ | ❌ | ❌ | ❌ |
| **Probabilistic** | ❌ | ❌ | ❌ | ✅ |
| **Deterministic** | ✅ | ❌ | ✅ | ❌ |
| **Best For** | Small, structured data | Large, spherical data | Arbitrary shapes | Overlapping clusters |

---

## Applications

Hierarchical Clustering is widely used across many domains:

| Domain | Application |
|---|---|
| 🧬 **Bioinformatics** | Gene expression profiling; grouping genes/proteins with similar expression patterns |
| 🛒 **Retail & E-Commerce** | Customer segmentation by purchasing behaviour and demographics |
| 🖼️ **Image Processing** | Grouping similar images; pixel-level colour segmentation |
| 📰 **NLP & Text Mining** | Document clustering; topic discovery in news articles or reviews |
| 🏥 **Healthcare** | Patient phenotyping; grouping patients with similar symptoms or lab values |
| 📡 **Social Networks** | Community detection; finding groups of connected users |
| 🌍 **Geospatial Analysis** | Clustering geographic locations into service zones or regions |
| 💹 **Finance** | Grouping stocks by trading behaviour; portfolio diversification |
| 🔬 **Anomaly Detection** | Identifying outliers as small or isolated clusters in the dendrogram |
| 🎓 **Education** | Grouping students by learning patterns and performance profiles |

---

## Advantages

- ✅ **No need to specify K in advance** — the number of clusters is chosen flexibly from the dendrogram after training.
- ✅ **Produces a dendrogram** — a rich visualisation that reveals the full hierarchical structure of the data at all scales.
- ✅ **Deterministic** — given the same data and linkage, it always produces exactly the same result (no random initialisation).
- ✅ **Captures nested structure** — naturally represents hierarchies (e.g., species → genus → family in biology).
- ✅ **Flexible distance metrics** — works with any distance or similarity measure.
- ✅ **No assumptions on cluster shape** — especially with single or average linkage, can find non-spherical clusters.
- ✅ **Works on any data type** — can handle mixed, categorical, or distance-matrix-only data (e.g., using pre-computed distances).
- ✅ **Useful for small datasets** — provides deep insight into data structure when sample sizes are manageable.

---

## Disadvantages

- ❌ **High computational complexity** — O(n² log n) time and O(n²) space; impractical for datasets with more than ~10,000 points.
- ❌ **Merges are irreversible** — once two clusters are merged (agglomerative), they cannot be separated in later steps; a bad early merge permanently affects all subsequent structure.
- ❌ **Sensitive to outliers** — especially with single linkage (chaining effect) and Ward's method; single noisy points can distort the entire hierarchy.
- ❌ **Sensitive to distance metric and linkage choice** — different combinations can produce very different dendrograms from the same data.
- ❌ **No built-in divisive method in sklearn** — divisive clustering requires custom implementation or workarounds.
- ❌ **Interpretability degrades at large scale** — dendrograms with hundreds or thousands of leaves become unreadable.
- ❌ **Cannot handle new data points** — unlike K-Means, there is no straightforward `predict()` for unseen samples without rerunning the entire algorithm.

---

## Summary Cheat Sheet

```
┌─────────────────────────────────────────────────────────────────────┐
│               Hierarchical Clustering Quick Reference               │
├──────────────────────────────┬──────────────────────────────────────┤
│ Algorithm Type               │ Unsupervised, Hierarchical           │
│ Tasks                        │ Clustering / Segmentation            │
│ Requires K?                  │ ❌ No — chosen from dendrogram       │
│ Types                        │ Agglomerative (bottom-up)            │
│                              │ Divisive (top-down)                  │
│ Default Approach             │ Agglomerative (HAC)                  │
│ Key Output                   │ Dendrogram + flat cluster labels     │
│ Distance Metrics             │ Euclidean, Manhattan, Cosine...      │
│ Linkage Methods              │ Single (MIN), Complete (MAX),        │
│                              │ Average (UPGMA), Ward's              │
│ Best Linkage (general)       │ Ward's (compact, balanced)           │
│ Choosing K                   │ Largest dendrogram gap / Silhouette  │
│ Evaluation Metrics           │ Silhouette, Cophenetic Corr,         │
│                              │ Davies-Bouldin, Calinski-Harabasz    │
│ Feature Scaling              │ REQUIRED (use StandardScaler)        │
│ Handles Outliers?            │ ⚠️ Sensitive (use Complete/Ward)     │
│ Time Complexity              │ O(n² log n)                         │
│ Space Complexity             │ O(n²) — distance matrix             │
│ Best Dataset Size            │ Small to medium (< 10,000 points)   │
│ Sklearn Class                │ AgglomerativeClustering             │
│ SciPy Functions              │ linkage(), dendrogram(), fcluster() │
│ Key Strength                 │ No need to pre-specify K;           │
│                              │ rich visual hierarchy (dendrogram)  │
│ Key Weakness                 │ O(n²) — slow on large datasets;     │
│                              │ irreversible merges                 │
└──────────────────────────────┴──────────────────────────────────────┘
```

---

## References

- [GeeksforGeeks — Hierarchical Clustering in Machine Learning](https://www.geeksforgeeks.org/machine-learning/hierarchical-clustering/)
- [GeeksforGeeks — Agglomerative Clustering](https://www.geeksforgeeks.org/machine-learning/agglomerative-clustering/)
- [GeeksforGeeks — Difference between Agglomerative and Divisive Clustering](https://www.geeksforgeeks.org/machine-learning/difference-between-agglomerative-clustering-and-divisive-clustering/)
- [Scikit-Learn AgglomerativeClustering Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- [Scikit-Learn Hierarchical Clustering User Guide](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
- [SciPy Hierarchical Clustering (scipy.cluster.hierarchy)](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
- [Ward, J.H. (1963). Hierarchical Grouping to Optimise an Objective Function. Journal of the American Statistical Association, 58(301), 236–244.](https://www.jstor.org/stable/2282967)
- [Murtagh, F. & Contreras, P. (2012). Algorithms for Hierarchical Clustering: An Overview. WIREs Data Mining and Knowledge Discovery.](https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.53)

---
