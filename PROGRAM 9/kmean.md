# PROGRAM NO. 14 – ML

**Roll No: 56**

## AIM
To implement K-Means clustering on the Digits dataset using different numbers of clusters and evaluate the results using inertia and silhouette score.

## TASKS
1. Load and preprocess the Digits dataset.
2. Standardize the 64 pixel intensity features.
3. Implement K-Means clustering with different K values.
4. Evaluate clustering performance using inertia and silhouette score.
5. Analyze the impact of K on clustering quality.

## ALGORITHM
1. Import the required libraries.
2. Load the Digits dataset using `load_digits()`.
3. Extract the 64 pixel features.
4. Standardize the features using `StandardScaler`.
5. Select K values of 5, 8, 10, and 12.
6. Apply K-Means clustering for each K value.
7. Calculate inertia using `model.inertia_`.
8. Calculate the silhouette score using `silhouette_score()`.
9. Display and compare the results for different K values.

## PROGRAM

```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = load_digits()
X = data.data

X = StandardScaler().fit_transform(X)

for k in [5, 8, 10, 12]:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    inertia = model.inertia_
    silhouette = silhouette_score(X, labels)

    print("K =", k)
    print("Inertia:", round(inertia, 2))
    print("Silhouette Score:", round(silhouette, 3))
    print()
```

## OUTPUT

The program displays the inertia and silhouette score for each K value:

```text
K = 5
Inertia: <calculated value>
Silhouette Score: <calculated value>

K = 8
Inertia: <calculated value>
Silhouette Score: <calculated value>

K = 10
Inertia: <calculated value>
Silhouette Score: <calculated value>

K = 12
Inertia: <calculated value>
Silhouette Score: <calculated value>
```

*Note: Replace the placeholders with the actual values obtained when you run the program.*

## RESULT

K-Means clustering was implemented on the Digits dataset using K values of 5, 8, 10, and 12. The clustering results were evaluated using inertia and silhouette score to analyze the effect of the number of clusters on clustering performance.
