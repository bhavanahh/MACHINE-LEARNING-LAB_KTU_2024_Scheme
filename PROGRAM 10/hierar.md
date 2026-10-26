# PROGRAM NO. 13 – ML

**Roll No: 56**

## AIM
To implement and compare Hierarchical (Agglomerative) and K-Means clustering algorithms on the Mall Customers dataset using clustering evaluation metrics and visualization.

## TASKS
1. Load and preprocess the Mall Customers dataset.
2. Apply K-Means clustering with five clusters.
3. Apply Agglomerative Hierarchical Clustering with five clusters.
4. Evaluate clustering using inertia and silhouette score.
5. Visualize and compare the clusters.
6. Discuss the advantages and disadvantages of both algorithms.

## ALGORITHM
1. Import the required libraries.
2. Load the Mall Customers dataset from `Mall_Customers.csv`.
3. Select Annual Income and Spending Score as input features.
4. Standardize the features using `StandardScaler`.
5. Apply K-Means clustering with K = 5.
6. Calculate and display the K-Means inertia and silhouette score.
7. Apply Agglomerative Hierarchical Clustering with five clusters.
8. Calculate and display its silhouette score.
9. Visualize both clustering results using scatter plots.
10. Compare the clustering quality and analyze the strengths and weaknesses of both algorithms.

## PROGRAM

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

data = pd.read_csv("Mall_Customers.csv")

X = data[["Annual Income (k$)", "Spending Score (1-100)"]]
X = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
k_labels = kmeans.fit_predict(X)

print("K-Means Inertia:", round(kmeans.inertia_, 2))
print("K-Means Silhouette Score:", round(silhouette_score(X, k_labels), 3))

agg = AgglomerativeClustering(n_clusters=5)
a_labels = agg.fit_predict(X)

print("\nHierarchical Silhouette Score:",
      round(silhouette_score(X, a_labels), 3))

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=k_labels)
plt.title("K-Means Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=a_labels)
plt.title("Hierarchical Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.show()
```

## OUTPUT

The program displays:
- K-Means inertia.
- K-Means silhouette score.
- Hierarchical clustering silhouette score.
- A scatter plot comparing K-Means and Hierarchical clustering results.

**Advantages and Disadvantages**

| Algorithm | Advantages | Disadvantages |
|---|---|---|
| K-Means | Simple, fast, and efficient for large datasets. | Requires the number of clusters to be specified in advance and is sensitive to initialization. |
| Hierarchical Clustering | Reveals hierarchical relationships between data points and does not require random centroid initialization. | Can be computationally expensive for large datasets and does not easily support reassignment after merging. |

*Note: Actual metric values and cluster visualizations depend on the dataset and execution environment.*

## RESULT

K-Means and Agglomerative Hierarchical Clustering were implemented on the Mall Customers dataset using Annual Income and Spending Score. Their clustering performance was compared using inertia, silhouette score, and scatter plots.
