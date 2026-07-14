# 🔵 DBSCAN — Density-Based Spatial Clustering of Applications with Noise

> A comprehensive guide to understanding, implementing, and applying DBSCAN — a powerful **density-based unsupervised machine learning** algorithm that discovers arbitrarily shaped clusters and automatically identifies outliers as noise, without requiring the number of clusters to be specified in advance.

---

## 📖 Table of Contents

- [Overview](#overview)
- [Why DBSCAN? Motivation & Problems it Solves](#why-dbscan-motivation--problems-it-solves)
- [Key Concepts & Terminology](#key-concepts--terminology)
- [Three Types of Points in DBSCAN](#three-types-of-points-in-dbscan)
- [Key Parameters](#key-parameters)
  - [eps (ε) — Neighbourhood Radius](#1-eps-ε--neighbourhood-radius)
  - [MinPts — Minimum Points Threshold](#2-minpts--minimum-points-threshold)
  - [Effect of eps and MinPts on Clustering](#effect-of-eps-and-minpts-on-clustering)
- [How DBSCAN Works — Step by Step](#how-dbscan-works--step-by-step)
- [Density Reachability & Connectivity](#density-reachability--connectivity)
- [Visual Walkthrough](#visual-walkthrough)
- [Mathematical Foundation](#mathematical-foundation)
  - [Euclidean Distance](#1-euclidean-distance)
  - [ε-Neighbourhood Definition](#2-ε-neighbourhood-definition)
  - [Core Point Condition](#3-core-point-condition)
  - [Density Reachability](#4-density-reachability)
  - [Density Connectivity](#5-density-connectivity)
- [Choosing eps Using the K-Distance Graph](#choosing-eps-using-the-k-distance-graph)
- [Python Implementation from Scratch](#python-implementation-from-scratch)
- [Step-by-Step Implementation with Scikit-Learn](#step-by-step-implementation-with-scikit-learn)
  - [Step 1 — Import Libraries](#step-1--import-libraries)
  - [Step 2 — Prepare Dataset](#step-2--prepare-dataset)
  - [Step 3 — Apply DBSCAN](#step-3--apply-dbscan)
  - [Step 4 — Visualise Clusters](#step-4--visualise-clusters)
  - [Step 5 — Evaluation Metrics](#step-5--evaluation-metrics)
- [DBSCAN on Real-World Shaped Data](#dbscan-on-real-world-shaped-data)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation Metrics](#evaluation-metrics)
- [Variants of DBSCAN](#variants-of-dbscan)
- [DBSCAN vs K-Means Clustering](#dbscan-vs-k-means-clustering)
- [DBSCAN vs Other Clustering Algorithms](#dbscan-vs-other-clustering-algorithms)
- [When to Use DBSCAN](#when-to-use-dbscan)
- [Applications](#applications)
- [Advantages](#advantages)
- [Disadvantages](#disadvantages)
- [Summary Cheat Sheet](#summary-cheat-sheet)
- [References](#references)

---

## Overview

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) is an unsupervised machine learning algorithm introduced by **Ester et al. in 1996**. It groups data points that are **closely packed together** in dense regions and marks isolated points in low-density areas as **noise (outliers)** — without requiring the number of clusters to be specified beforehand.

### Key Characteristics

| Property | Description |
|---|---|
| **Full Name** | Density-Based Spatial Clustering of Applications with Noise |
| **Type** | Unsupervised Learning |
| **Task** | Clustering + Outlier/Noise Detection |
| **Requires K?** | ❌ No — number of clusters found automatically |
| **Requires Labels?** | ❌ No — works on raw unlabelled data |
| **Cluster Shapes** | ✅ Any arbitrary shape |
| **Outlier Detection** | ✅ Built-in — noise points labelled as −1 |
| **Key Parameters** | `eps` (ε) and `MinPts` |
| **Deterministic?** | ✅ Yes (given same eps and MinPts) |

### Core Intuition

```
Think of a city map viewed from above:
  
  Dense city blocks    → Clusters   (many buildings close together)
  Sparse countryside   → Noise      (isolated buildings far apart)
  City edge buildings  → Border pts (near a cluster but not dense)

DBSCAN finds the "city blocks" automatically — even if they are
shaped like rings, crescents, or interlocking spirals.
```

---

## Why DBSCAN? Motivation & Problems it Solves

Traditional clustering algorithms like K-Means and Hierarchical Clustering assume clusters are **compact and spherical**. Real-world data rarely fits this assumption.

```
K-Means Failure Cases:

  Crescent Shapes         Rings / Donuts         Irregular Blobs
  ──────────────          ─────────────          ───────────────
    )))   (((              ●●●●●●●●●              ●● ●  ●●●
  K-Means splits          ●  ○○○○  ●               ● ●● ●
  these incorrectly       ●  ○○○○  ●               ●●    ●●
                          ●●●●●●●●●            ●●●●  ●●

                          K-Means: FAILS         K-Means: FAILS
                          DBSCAN:  WORKS ✅      DBSCAN:  WORKS ✅
```

### Problems DBSCAN Solves

| Problem | K-Means | Hierarchical | DBSCAN |
|---|---|---|---|
| Arbitrary-shaped clusters | ❌ Fails | ⚠️ Partial | ✅ Handles |
| Unknown number of clusters | ❌ Must specify K | ❌ Must cut dendrogram | ✅ Auto-detected |
| Noise and outlier detection | ❌ None | ❌ None | ✅ Built-in |
| Clusters of varying density | ❌ Struggles | ⚠️ Partial | ⚠️ Single eps |
| Non-convex cluster shapes | ❌ Fails | ⚠️ With single linkage | ✅ Handles |

---

## Key Concepts & Terminology

### 📍 Neighbourhood (ε-Neighbourhood)
The set of all points within a radius of **eps (ε)** from a given point p. Written as Nₑ(p):

```
Nₑ(p) = { q ∈ D  :  dist(p, q) ≤ ε }

  Visualised:
         ε radius
      ┌─────────┐
      │  q  q   │  ← all points inside this circle
      │    p    │     are in the ε-neighbourhood of p
      │  q   q  │
      └─────────┘
```

### 📊 Density
The **number of data points** within the ε-neighbourhood of a point. High density = many points packed closely; low density = few or no nearby points.

### 🔗 Density Reachability
Point q is **directly density-reachable** from core point p if q is within p's ε-neighbourhood.

### ⛓️ Density Connectivity
Two points a and b are **density-connected** if there exists a chain of directly density-reachable points linking them — even if neither a nor b is a core point itself.

### 🏷️ Cluster Label −1
DBSCAN labels all **noise/outlier points** with the special value **−1** (not assigned to any cluster).

---

## Three Types of Points in DBSCAN

DBSCAN categorises every data point into exactly one of three types based on its neighbourhood density:

```
┌────────────────────────────────────────────────────────────────────┐
│                  Three Point Types in DBSCAN                      │
├─────────────────┬──────────────────────────────────────────────────┤
│  Point Type     │  Definition                                      │
├─────────────────┼──────────────────────────────────────────────────┤
│  🔴 Core Point  │  Has ≥ MinPts points within its ε-neighbourhood  │
│                 │  (including itself)                              │
│                 │  → Forms the "heart" of a cluster                │
├─────────────────┼──────────────────────────────────────────────────┤
│  🟡 Border Pt   │  Has < MinPts in ε-neighbourhood                 │
│                 │  BUT falls within the ε-neighbourhood of a       │
│                 │  Core Point                                      │
│                 │  → On the "edge" of a cluster                   │
├─────────────────┼──────────────────────────────────────────────────┤
│  ⚫ Noise Point  │  Has < MinPts in ε-neighbourhood               │
│                 │  AND is NOT within any Core Point's ε-radius     │
│                 │  → Not part of any cluster; labelled −1          │
└─────────────────┴──────────────────────────────────────────────────┘
```

### Visual Illustration (MinPts = 4, ε = radius of circles)

```
                        ε
                   ┌─────────┐
              ●    │  ●  ●   │
                   │    ●    │
          ●        │  🔴 ●   │ ← Core Point (5 points in ε-radius ≥ MinPts=4)
                   │  ●      │
              ●    └─────────┘

  🟡 Border Point: within ε of a core point, but own ε-circle has < 4 points
  ⚫ Noise Point:  not within ε of ANY core point
```

### Detailed Example Table (MinPts = 3)

| Point | Points in ε-radius | Is Core? | In Core's ε-radius? | Type |
|---|---|---|---|---|
| A | 5 | ✅ Yes (≥ 3) | — | 🔴 Core |
| B | 4 | ✅ Yes (≥ 3) | — | 🔴 Core |
| C | 2 | ❌ No (< 3) | ✅ Within A's ε | 🟡 Border |
| D | 1 | ❌ No (< 3) | ❌ Not near any Core | ⚫ Noise |
| E | 3 | ✅ Yes (= 3) | — | 🔴 Core |

---

## Key Parameters

### 1. eps (ε) — Neighbourhood Radius

`eps` defines the **radius of the circular neighbourhood** around each data point. Points within this radius are considered "neighbours."

```
If eps is too SMALL:
  → Very few points qualify as Core Points
  → Most points become Noise
  → Clusters are fragmented or empty

If eps is too LARGE:
  → Almost all points become Core Points
  → Clusters merge into one giant cluster
  → No meaningful separation

Ideal eps:
  → Found using the K-Distance Graph / Elbow method
```

### 2. MinPts — Minimum Points Threshold

`MinPts` is the **minimum number of data points** (including the point itself) that must fall within the ε-radius for a point to be classified as a Core Point.

```
General Rule of Thumb:
  MinPts ≥ D + 1    where D = number of dimensions

  2D data  → MinPts ≥ 3  (minimum recommended: 3)
  3D data  → MinPts ≥ 4
  nD data  → MinPts ≥ n + 1

Practical recommendation:
  MinPts = 2 × D  (for larger, noisier datasets)
  MinPts = 3 or 4  (commonly used default)
```

### Effect of eps and MinPts on Clustering

```
                    eps → Small              eps → Large
                ┌─────────────────┬──────────────────────────┐
  MinPts Small  │ Many clusters,  │ Few large clusters,      │
                │ many noise pts  │ almost no noise          │
                ├─────────────────┼──────────────────────────┤
  MinPts Large  │ Few/no clusters │ Balanced clustering      │
                │ almost all noise│ (usually best zone ✅)   │
                └─────────────────┴──────────────────────────┘
```

---

## How DBSCAN Works — Step by Step

```
┌───────────────────────────────────────────────────────────────────┐
│                    DBSCAN Algorithm Flow                          │
└───────────────────────────────────────────────────────────────────┘

Input:  Dataset D, parameters eps (ε) and MinPts
Output: Cluster assignments (integer ≥ 0) and noise labels (−1)

Step 1: Initialise
   └─► Mark all points as UNVISITED
       Set cluster counter C = 0

Step 2: Iterate Over All Points
   └─► For each UNVISITED point P in D:
       Mark P as VISITED

       Step 2a: Compute ε-neighbourhood
          └─► Find all points Q such that dist(P, Q) ≤ ε
              → This is NeighborhoodPts(P)

       Step 2b: Check Core Point Condition
          ├─► If |NeighborhoodPts(P)| < MinPts:
          │       Mark P as NOISE (label = −1)    ← may be changed later
          │
          └─► If |NeighborhoodPts(P)| ≥ MinPts:
                  P is a CORE POINT
                  Create new cluster C
                  Add P to cluster C

                  Step 2c: Expand Cluster
                  └─► Add all points in NeighborhoodPts(P) to a QUEUE
                      For each point Q in QUEUE:
                        If Q is UNVISITED:
                          Mark Q as VISITED
                          Compute NeighborhoodPts(Q)
                          If |NeighborhoodPts(Q)| ≥ MinPts:
                            Add NeighborhoodPts(Q) to QUEUE
                        If Q is not yet in any cluster:
                          Assign Q to cluster C

Step 3: After All Points Processed
   └─► Points still labelled −1 are confirmed NOISE/OUTLIERS
       All other points have a cluster label 0, 1, 2, ...

Step 4: Output
   └─► labels_ array: −1 = noise, 0,1,2,... = cluster IDs
       core_sample_indices_: indices of all Core Points
```

---

## Density Reachability & Connectivity

These are the two foundational concepts that define how DBSCAN expands clusters:

### Direct Density Reachability

Point **q** is **directly density-reachable** from point **p** if:

```
1. dist(p, q) ≤ ε          ← q is in p's neighbourhood
2. |Nₑ(p)| ≥ MinPts        ← p is a Core Point
```

Note: This is **not symmetric** — q may be reachable from p, but p may not be reachable from q (if q is a Border Point).

### Density Reachability (Chaining)

Point **q** is **density-reachable** from point **p** if there exists a chain of points p₁, p₂, ..., pₙ where:

```
  p₁ = p,  pₙ = q
  Each pᵢ₊₁ is directly density-reachable from pᵢ
```

```
  Core   Core   Core  Border
   p₁ ──► p₂ ──► p₃ ──► q

  q is density-reachable from p₁ via the chain
```

### Density Connectivity

Points **a** and **b** are **density-connected** if there exists a Core Point **o** such that both a and b are density-reachable from o:

```
      a ◄──── o ────► b
              ↑
          Core Point

  a and b are density-connected through o
  → They belong to the SAME cluster
```

### Why This Matters

```
Density Connectivity is SYMMETRIC:
  If a and b are density-connected → same cluster
  Even if neither a nor b is a Core Point

This allows DBSCAN to form:
  ✅ Elongated clusters
  ✅ Crescent/arc clusters
  ✅ Interlocking ring clusters
  ✅ Any arbitrary shape
```

---

## Visual Walkthrough

### Setup: MinPts = 3, ε = 1.5

```
Dataset (2D points labelled A–J):

  Y
  │
6 │        F
5 │     E     G
4 │   D   ●       J  ← isolated
3 │     C
2 │  B
1 │    A                   I  ← isolated
  └──────────────────────── X
     1  2  3  4  5  6  7  8
```

### Step 1 — Compute Neighbourhoods

```
Point  Neighbours (dist ≤ 1.5)    Count   Type
─────  ──────────────────────     ─────   ────────────
A      B                          1       ⚫ Noise
B      A, C                       2       ⚫ Noise (initially)
C      B, D, E                    3       🔴 Core  (≥ MinPts=3)
D      C, E                       2       🟡 Border (near C)
E      C, D, F, G                 4       🔴 Core
F      E, G                       2       🟡 Border (near E)
G      E, F                       2       🟡 Border (near E)
J      (none within 1.5)          0       ⚫ Noise
I      (none within 1.5)          0       ⚫ Noise
```

### Step 2 — Form Clusters

```
Start from C (first Core Point found):
  Cluster 1 = {C}
  C's neighbours: B, D, E → add to queue

  Process B: |Nₑ(B)| = 2 < 3 → Border, add to Cluster 1
  Process D: |Nₑ(D)| = 2 < 3 → Border, add to Cluster 1
  Process E: |Nₑ(E)| = 4 ≥ 3 → Core!  expand: F, G → add to queue
    Process F: Border → Cluster 1
    Process G: Border → Cluster 1

  Cluster 1 = {B, C, D, E, F, G}
```

### Step 3 — Label Noise

```
Final assignments:
  🔵 Cluster 1 : B, C, D, E, F, G
  ⚫ Noise (−1) : A, I, J
```

---

## Mathematical Foundation

### 1. Euclidean Distance

Default distance metric for computing ε-neighbourhoods:

```
dist(p, q) = √[ Σᵢ (pᵢ − qᵢ)² ]
```

Other supported metrics include Manhattan (`'manhattan'`), Chebyshev (`'chebyshev'`), Cosine (`'cosine'`), and any metric from `scipy.spatial.distance`.

---

### 2. ε-Neighbourhood Definition

```
Nₑ(p) = { q ∈ D  |  dist(p, q) ≤ ε }
```

| Symbol | Meaning |
|---|---|
| `Nₑ(p)` | Set of all points within ε-radius of point p |
| `D` | Full dataset |
| `ε (eps)` | User-defined neighbourhood radius |

---

### 3. Core Point Condition

```
p is a Core Point  ⟺  |Nₑ(p)| ≥ MinPts
```

| Symbol | Meaning |
|---|---|
| `\|Nₑ(p)\|` | Number of points in ε-neighbourhood of p (including p) |
| `MinPts` | Minimum density threshold |

---

### 4. Density Reachability

```
q is density-reachable from p  ⟺
  ∃ chain p₁=p, p₂, ..., pₙ=q  such that
  pᵢ₊₁ ∈ Nₑ(pᵢ)  and  |Nₑ(pᵢ)| ≥ MinPts  for all i < n
```

---

### 5. Density Connectivity

```
p and q are density-connected  ⟺
  ∃ Core Point o  such that:
    p is density-reachable from o   AND
    q is density-reachable from o
```

A **cluster** in DBSCAN is the maximal set of mutually density-connected points.

---

## Choosing eps Using the K-Distance Graph

Selecting the right `eps` is the most critical tuning step. The **K-Distance Graph** (also called the elbow/knee method for eps) provides a systematic approach.

### Method

1. For each data point, compute the distance to its **k-th nearest neighbour** (where k = MinPts).
2. Sort these distances in **descending order**.
3. Plot the sorted distances.
4. Look for the **"elbow" or "knee"** — the point of maximum curvature.
5. The distance value at the elbow is the optimal `eps`.

```
k-th Nearest Neighbour Distance
   │
   │\
   │ \
   │  \___
   │       \____
   │             \_________  ← flat region: points inside dense clusters
   │
   └──────────────────────────── Points (sorted by distance, descending)
              ^
           Elbow = optimal eps
```

### Python Code — K-Distance Graph

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Generate data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.5, random_state=0)
X_scaled = StandardScaler().fit_transform(X)

# Compute k-distance (k = MinPts)
min_pts = 5
nbrs = NearestNeighbors(n_neighbors=min_pts).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)

# Sort in descending order and plot
k_distances = np.sort(distances[:, min_pts - 1])[::-1]

plt.figure(figsize=(9, 5))
plt.plot(k_distances, linewidth=2, color='steelblue')
plt.axhline(y=0.3, color='red', linestyle='--', label='eps = 0.3 (elbow)')
plt.title(f'K-Distance Graph (k = {min_pts}) — Finding Optimal eps')
plt.xlabel('Points (sorted by k-distance, descending)')
plt.ylabel(f'{min_pts}-th Nearest Neighbour Distance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## Python Implementation from Scratch

```python
import numpy as np
from collections import deque


class DBSCAN_Scratch:
    """
    DBSCAN implemented from scratch using only NumPy.
    """

    def __init__(self, eps=0.5, min_pts=5):
        self.eps     = eps
        self.min_pts = min_pts
        self.labels_ = None

    def _euclidean(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def _get_neighbours(self, X, idx):
        """Return indices of all points within eps of X[idx]."""
        neighbours = []
        for i, point in enumerate(X):
            if self._euclidean(X[idx], point) <= self.eps:
                neighbours.append(i)
        return neighbours

    def fit_predict(self, X):
        X = np.array(X)
        n = len(X)
        labels = np.full(n, -1, dtype=int)   # −1 = unassigned / noise
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True

            neighbours = self._get_neighbours(X, i)

            if len(neighbours) < self.min_pts:
                # Mark as noise (may be reassigned later as Border Point)
                labels[i] = -1
            else:
                # i is a Core Point — start a new cluster
                self._expand_cluster(X, labels, visited, i, neighbours,
                                     cluster_id)
                cluster_id += 1

        self.labels_ = labels
        return labels

    def _expand_cluster(self, X, labels, visited, core_idx,
                        neighbours, cluster_id):
        """BFS expansion from a core point."""
        labels[core_idx] = cluster_id
        queue = deque(neighbours)

        while queue:
            q = queue.popleft()

            if not visited[q]:
                visited[q] = True
                q_neighbours = self._get_neighbours(X, q)

                if len(q_neighbours) >= self.min_pts:
                    # q is also a Core Point — expand further
                    queue.extend(q_neighbours)

            # Assign to cluster if not yet assigned
            if labels[q] == -1:
                labels[q] = cluster_id
```

### Usage

```python
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Non-linear crescent-shaped data — K-Means fails here!
X, y = make_moons(n_samples=200, noise=0.08, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# Apply custom DBSCAN
db = DBSCAN_Scratch(eps=0.3, min_pts=5)
labels = db.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = list(labels).count(-1)

print(f"Clusters found : {n_clusters}")
print(f"Noise points   : {n_noise}")

# Visualise
plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
            c=labels, cmap='Set1', s=40, edgecolors='k', alpha=0.8)
plt.title(f'Custom DBSCAN — {n_clusters} clusters, {n_noise} noise points')
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## Step-by-Step Implementation with Scikit-Learn

### Step 1 — Import Libraries

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
```

### Step 2 — Prepare Dataset

```python
# Create dataset with 4 visible clusters, 300 points
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.50, random_state=0)

# Feature scaling is CRITICAL for DBSCAN
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Quick look at raw data
plt.figure(figsize=(7, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], s=30, alpha=0.7)
plt.title('Raw Dataset (Before Clustering)')
plt.grid(True)
plt.show()
```

### Step 3 — Apply DBSCAN

```python
# Fit DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X_scaled)

# Extract results
labels             = db.labels_
core_samples_mask  = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# Summary
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = list(labels).count(-1)

print(f"Estimated number of clusters : {n_clusters}")
print(f"Estimated number of noise pts: {n_noise}")
print(f"Core sample indices count    : {len(db.core_sample_indices_)}")
print(f"Unique labels                : {set(labels)}")
```

### Step 4 — Visualise Clusters

```python
plt.figure(figsize=(9, 6))

unique_labels = set(labels)
palette       = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, palette):
    if k == -1:
        col = 'black'    # Noise points always shown in black

    mask = labels == k

    # Core points: large filled circles
    plt.scatter(X_scaled[mask & core_samples_mask, 0],
                X_scaled[mask & core_samples_mask, 1],
                s=60, color=col, edgecolors='k',
                label=f'Cluster {k}' if k != -1 else 'Noise',
                zorder=3)

    # Border points: smaller crosses
    plt.scatter(X_scaled[mask & ~core_samples_mask, 0],
                X_scaled[mask & ~core_samples_mask, 1],
                marker='x', s=40, color=col, linewidths=1.5,
                zorder=2)

plt.title(f'DBSCAN Clustering — {n_clusters} Clusters  '
          f'(● Core  ✕ Border  ● Black=Noise)')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()
```

### Step 5 — Evaluation Metrics

```python
from sklearn import metrics

# Silhouette Score (higher = better, range: −1 to +1)
if n_clusters > 1:
    sc = metrics.silhouette_score(X_scaled, labels)
    print(f"Silhouette Coefficient  : {sc:.4f}")
else:
    print("Silhouette Score not applicable (< 2 clusters found)")

# Adjusted Rand Index — requires ground truth labels
ari = metrics.adjusted_rand_score(y_true, labels)
print(f"Adjusted Rand Index (ARI): {ari:.4f}")

# Interpretation
print("\n── ARI Interpretation ──")
print("  > 0.90  → Excellent cluster recovery")
print("  > 0.80  → Good cluster recovery")
print("  < 0.50  → Poor cluster recovery")
```

---

## DBSCAN on Real-World Shaped Data

One of DBSCAN's greatest strengths is handling non-spherical clusters that defeat K-Means:

```python
from sklearn.datasets import make_moons, make_circles
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

datasets = [
    ("Two Moons",   make_moons(n_samples=300,  noise=0.07, random_state=42),
     {'eps': 0.25, 'min_samples': 5},   {'n_clusters': 2}),
    ("Two Circles", make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42),
     {'eps': 0.18, 'min_samples': 5},   {'n_clusters': 2}),
]

for col, (name, (X, y), db_params, km_params) in enumerate(datasets):
    X = StandardScaler().fit_transform(X)

    # DBSCAN
    db_labels = DBSCAN(**db_params).fit_predict(X)
    axes[0, col * 2].scatter(X[:, 0], X[:, 1], c=db_labels, cmap='Set1', s=20)
    axes[0, col * 2].set_title(f'DBSCAN — {name}')

    # K-Means
    km_labels = KMeans(**km_params, random_state=42).fit_predict(X)
    axes[0, col * 2 + 1].scatter(X[:, 0], X[:, 1], c=km_labels, cmap='Set1', s=20)
    axes[0, col * 2 + 1].set_title(f'K-Means — {name}')

plt.suptitle('DBSCAN vs K-Means on Non-Spherical Data', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

## Hyperparameter Tuning

### Key Parameters in sklearn's DBSCAN

| Parameter | Type | Default | Description |
|---|---|---|---|
| `eps` | float | `0.5` | ε-neighbourhood radius — most critical parameter |
| `min_samples` | int | `5` | Minimum points in ε-radius to form a Core Point (MinPts) |
| `metric` | str / callable | `'euclidean'` | Distance metric: `'euclidean'`, `'manhattan'`, `'cosine'`, etc. |
| `metric_params` | dict | `None` | Extra keyword arguments for the metric function |
| `algorithm` | str | `'auto'` | Algorithm for neighbour search: `'auto'`, `'ball_tree'`, `'kd_tree'`, `'brute'` |
| `leaf_size` | int | `30` | Leaf size passed to BallTree or KDTree (affects speed/memory) |
| `n_jobs` | int | `None` | Parallel jobs for neighbour computation (`−1` = all CPUs) |
| `p` | float | `None` | Power of the Minkowski metric (2 = Euclidean, 1 = Manhattan) |

### Grid Search for Best eps and min_samples

```python
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import warnings
warnings.filterwarnings('ignore')

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.5, random_state=0)
X_scaled = StandardScaler().fit_transform(X)

eps_values      = np.arange(0.1, 1.5, 0.1)
min_pts_values  = range(3, 15)

best_score  = -1
best_params = {}
results     = []

for eps in eps_values:
    for min_pts in min_pts_values:
        db     = DBSCAN(eps=eps, min_samples=min_pts)
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = list(labels).count(-1)

        if n_clusters < 2:
            continue   # Silhouette requires at least 2 clusters

        try:
            score = silhouette_score(X_scaled, labels)
            results.append({'eps': eps, 'min_pts': min_pts,
                            'n_clusters': n_clusters, 'n_noise': n_noise,
                            'silhouette': score})
            if score > best_score:
                best_score  = score
                best_params = {'eps': eps, 'min_pts': min_pts}
        except Exception:
            pass

print(f"Best eps        : {best_params.get('eps'):.1f}")
print(f"Best min_pts    : {best_params.get('min_pts')}")
print(f"Best Silhouette : {best_score:.4f}")
```

---

## Evaluation Metrics

### 1. Silhouette Score (No Ground Truth Needed)

Measures cluster cohesion and separation. Excludes noise points (label = −1).

```
s(i) = (b(i) − a(i)) / max(a(i), b(i))
```

| Score | Interpretation |
|---|---|
| Close to +1 | Well-clustered point, far from other clusters |
| Close to 0 | On or near a cluster boundary |
| Close to −1 | Likely assigned to wrong cluster |

```python
from sklearn.metrics import silhouette_score

# Exclude noise points for Silhouette calculation
mask          = labels != -1
filtered_X    = X_scaled[mask]
filtered_lbls = labels[mask]

if len(set(filtered_lbls)) > 1:
    score = silhouette_score(filtered_X, filtered_lbls)
    print(f"Silhouette Score (excl. noise): {score:.4f}")
```

### 2. Adjusted Rand Index — ARI (Requires Ground Truth)

Compares predicted cluster assignments to true labels. Range: 0 to 1.

```python
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(y_true, labels)
print(f"Adjusted Rand Index: {ari:.4f}")
```

| ARI Range | Interpretation |
|---|---|
| > 0.90 | Excellent cluster recovery |
| 0.80 – 0.90 | Good recovery |
| 0.50 – 0.80 | Moderate recovery |
| < 0.50 | Poor recovery |

### 3. Adjusted Mutual Information — AMI (Requires Ground Truth)

```python
from sklearn.metrics import adjusted_mutual_info_score

ami = adjusted_mutual_info_score(y_true, labels)
print(f"Adjusted Mutual Info: {ami:.4f}")
```

### 4. Noise Ratio

A simple diagnostic — too many noise points signals eps is too small or min_pts too large:

```python
noise_ratio = list(labels).count(-1) / len(labels)
print(f"Noise Ratio: {noise_ratio:.2%}")
# Ideal: < 5–10% for clean datasets; higher is expected in noisy real-world data
```

### Metrics Summary

| Metric | Ground Truth? | Range | Optimal |
|---|---|---|---|
| **Silhouette Score** | ❌ No | −1 to +1 | Closer to +1 |
| **Adjusted Rand Index** | ✅ Yes | 0 to 1 | Closer to 1 |
| **Adjusted Mutual Info** | ✅ Yes | 0 to 1 | Closer to 1 |
| **Noise Ratio** | ❌ No | 0% to 100% | As low as possible |
| **Davies-Bouldin Index** | ❌ No | 0 to ∞ | Closer to 0 |

---

## Variants of DBSCAN

| Variant | Key Difference | Best Use Case |
|---|---|---|
| **DBSCAN** | Original; single global eps | General-purpose density clustering |
| **HDBSCAN** | Hierarchical; auto-selects optimal eps per cluster | Varying-density clusters; most robust |
| **OPTICS** | Orders points by reachability; handles varying density | Variable-density data; produces reachability plot |
| **ST-DBSCAN** | Extends DBSCAN to spatio-temporal data | GPS tracks, event streams |
| **DENCLUE** | Uses kernel density functions | Smooth, continuous density estimation |
| **GDBSCAN** | Generalised for arbitrary neighbourhood shapes | Non-Euclidean or custom distance spaces |
| **δ-DBSCAN** | Adds delta parameter for overlapping cluster detection | Clusters with fuzzy boundaries |

### HDBSCAN — The Modern Upgrade

```python
# pip install hdbscan
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
labels    = clusterer.fit_predict(X_scaled)

print(f"HDBSCAN clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"Noise points    : {list(labels).count(-1)}")
```

---

## DBSCAN vs K-Means Clustering

| Feature | DBSCAN | K-Means |
|---|---|---|
| **Number of clusters** | Auto-detected | Must specify K upfront |
| **Cluster shape** | ✅ Any arbitrary shape | ❌ Spherical/convex only |
| **Outlier handling** | ✅ Built-in (label = −1) | ❌ None — outliers skew centroids |
| **Parameters required** | `eps` and `MinPts` (2 params) | `K` only (1 param) |
| **Sensitivity to outliers** | ✅ Robust | ❌ Very sensitive |
| **Uniform cluster sizes** | ❌ Not required | ⚠️ Assumes equal size |
| **Scalability** | ⚠️ O(n log n) with index | ✅ O(n·K·I) fast |
| **Deterministic** | ✅ Yes | ❌ Depends on init |
| **Works on non-convex data** | ✅ Excellent | ❌ Fails |
| **Memory usage** | ⚠️ High (distance matrix) | ✅ Low (centroids only) |
| **Best for** | Noisy, irregular-shaped data | Large, spherical, clean data |

---

## DBSCAN vs Other Clustering Algorithms

| Feature | DBSCAN | K-Means | Hierarchical | GMM |
|---|---|---|---|---|
| **Requires K?** | ❌ Auto | ✅ Yes | ❌ Post-hoc | ✅ Yes |
| **Cluster Shape** | Any | Spherical | Any | Elliptical |
| **Noise Detection** | ✅ Built-in | ❌ | ❌ | ⚠️ Partial |
| **Scalability** | O(n log n) | O(nKI) | O(n² log n) | O(n·K²) |
| **Varying Density** | ⚠️ Single ε | ❌ | ⚠️ | ✅ |
| **Dendrogram** | ❌ | ❌ | ✅ | ❌ |
| **Probabilistic** | ❌ | ❌ | ❌ | ✅ |
| **High Dimensions** | ⚠️ Struggles | ⚠️ | ❌ | ⚠️ |

---

## When to Use DBSCAN

### ✅ Use DBSCAN When

- Data contains **arbitrary-shaped clusters** (crescents, rings, spirals, S-curves).
- The **number of clusters is unknown** in advance.
- The dataset has **noise or outliers** that should be identified, not forced into clusters.
- Clusters have **different sizes** but relatively **similar densities**.
- You need a **deterministic** result (reproducible without random seed).
- Performing **spatial/geographic analysis** (GPS coordinates, sensor data).

### ❌ Avoid DBSCAN When

- Clusters have **highly varying densities** — use HDBSCAN or OPTICS instead.
- The dataset is **very high-dimensional** — Euclidean distance loses meaning; consider dimensionality reduction first.
- You need to **predict cluster membership for new data points** — DBSCAN has no native `predict()`.
- The dataset is **very large** (millions of rows) — use Mini-Batch K-Means or HDBSCAN.
- Clusters are **compact and spherical** with no noise — K-Means is faster and simpler.

---

## Applications

DBSCAN is widely used across many domains due to its noise-robustness and shape flexibility:

| Domain | Application |
|---|---|
| 📍 **Geospatial Analysis** | Identifying hotspots in GPS trajectories; city region discovery |
| 🛡️ **Anomaly / Fraud Detection** | Flagging unusual transactions, network intrusions as noise points |
| 🧬 **Bioinformatics** | Clustering gene expression data; protein structure analysis |
| 🌌 **Astronomy** | Discovering galaxy clusters and stellar structures in sky surveys |
| 🖼️ **Image Segmentation** | Grouping pixels by colour or texture in irregular-shaped regions |
| 🌐 **Social Network Analysis** | Detecting communities in graphs with noisy, peripheral members |
| 🚗 **Traffic Analysis** | Identifying accident hotspots and congestion zones from GPS data |
| 📡 **Telecommunications** | Network fault detection; signal anomaly grouping |
| 🏭 **Manufacturing** | Quality control — detecting defective product clusters from sensor logs |
| 🌊 **Environmental Science** | Grouping ocean current data; identifying climate zones |

---

## Advantages

- ✅ **No need to specify the number of clusters** — K is discovered automatically from the data structure.
- ✅ **Handles arbitrarily shaped clusters** — not limited to spherical or convex forms; can find rings, spirals, crescents, and any shape.
- ✅ **Built-in outlier/noise detection** — points in low-density regions are automatically labelled as noise (−1) without extra steps.
- ✅ **Robust to outliers** — unlike K-Means, a single extreme point cannot distort an entire cluster since it will simply be marked as noise.
- ✅ **Deterministic** — given the same eps and MinPts, DBSCAN always produces the same result (no random initialisation).
- ✅ **No assumptions about cluster shape or size** — purely data-driven; clusters emerge from density patterns alone.
- ✅ **Works with any distance metric** — Euclidean, Manhattan, cosine, Haversine (for geo-coordinates), or custom metrics.
- ✅ **Efficient with spatial indexing** — O(n log n) with Ball Tree or KD-Tree for low-to-medium dimensions.

---

## Disadvantages

- ❌ **Struggles with varying-density clusters** — a single global eps cannot simultaneously capture both dense and sparse clusters; use HDBSCAN for varying densities.
- ❌ **Sensitive to eps and MinPts selection** — small changes can drastically alter the number of clusters or noise ratio; requires careful K-distance graph analysis.
- ❌ **Poor performance in high dimensions** — the Euclidean distance becomes less meaningful in high-dimensional spaces (curse of dimensionality); consider dimensionality reduction (PCA, UMAP) first.
- ❌ **No native predict() for new data** — DBSCAN cannot classify new, unseen data points without re-running the algorithm on the entire dataset.
- ❌ **Memory-intensive** — computing and storing pairwise distances requires O(n²) memory without spatial indexing.
- ❌ **Border point assignment is non-deterministic** — a Border Point reachable from two different clusters may be assigned to either, depending on processing order.
- ❌ **Not suitable for very large datasets** — without approximate nearest neighbours, scales poorly beyond ~100K points.

---

## Summary Cheat Sheet

```
┌──────────────────────────────────────────────────────────────────────┐
│                    DBSCAN Quick Reference                            │
├────────────────────────────┬─────────────────────────────────────────┤
│ Full Name                  │ Density-Based Spatial Clustering of     │
│                            │ Applications with Noise                 │
│ Algorithm Type             │ Unsupervised, Density-Based             │
│ Task                       │ Clustering + Outlier Detection          │
│ Requires K?                │ ❌ No — discovered automatically        │
│ Key Parameters             │ eps (ε) and MinPts (min_samples)        │
│ Point Types                │ Core, Border, Noise (−1)               │
│ Core Point Condition       │ |Nₑ(p)| ≥ MinPts                      │
│ Cluster Formation          │ Density connectivity via Core Points    │
│ Cluster Shapes             │ ✅ Any arbitrary shape                  │
│ Outlier Handling           │ ✅ Built-in (labelled as −1)            │
│ MinPts Rule of Thumb       │ ≥ D + 1 (D = dimensions); min = 3      │
│ eps Selection              │ K-Distance Graph (elbow/knee method)    │
│ Distance Metric (default)  │ Euclidean (others supported)            │
│ Deterministic?             │ ✅ Yes (no random init)                 │
│ Feature Scaling            │ ✅ REQUIRED (StandardScaler)            │
│ Evaluation (no labels)     │ Silhouette Score, Davies-Bouldin        │
│ Evaluation (with labels)   │ Adjusted Rand Index (ARI), AMI         │
│ Time Complexity            │ O(n log n) with spatial index           │
│ Space Complexity           │ O(n²) without index                     │
│ Sklearn Class              │ sklearn.cluster.DBSCAN                 │
│ Modern Alternative         │ HDBSCAN (handles varying density)       │
│ Key Strength               │ Arbitrary shapes + noise detection      │
│ Key Weakness               │ Single eps fails on varying density     │
│ Best For                   │ Noisy, irregular, geospatial data       │
│ Avoid When                 │ High-dim, very large, or varying-       │
│                            │ density datasets                        │
└────────────────────────────┴─────────────────────────────────────────┘
```

---

## References

- [GeeksforGeeks — DBSCAN Clustering in ML: Density-Based Clustering](https://www.geeksforgeeks.org/machine-learning/dbscan-clustering-in-ml-density-based-clustering/)
- [GeeksforGeeks — Silhouette Algorithm to Determine Optimal K](https://www.geeksforgeeks.org/machine-learning/silhouette-algorithm-to-determine-the-optimal-value-of-k/)
- [GeeksforGeeks — Rand Index in Machine Learning](https://www.geeksforgeeks.org/machine-learning/rand-index-in-machine-learning/)
- [Scikit-Learn DBSCAN Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [Scikit-Learn Clustering User Guide — DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
