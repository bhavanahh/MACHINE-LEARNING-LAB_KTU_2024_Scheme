# PROGRAM NO. 12 – ML

**Roll No: 56**

## AIM
To implement the K-Nearest Neighbors (KNN) algorithm using the Fashion-MNIST dataset and compare its accuracy and execution time for different values of K.

## TASKS
1. Load the Fashion-MNIST dataset.
2. Normalize the pixel values.
3. Split the dataset into training and testing sets.
4. Implement the KNN algorithm using Euclidean distance.
5. Classify images using K = 3, 5, and 7.
6. Calculate the accuracy and execution time for each K value.
7. Compare the classification results.

## ALGORITHM
1. Import the required libraries.
2. Load the Fashion-MNIST dataset using `fetch_openml()`.
3. Select the first 2,000 images and their corresponding labels.
4. Normalize the pixel values by dividing them by 255.
5. Split the dataset into training and testing sets using `train_test_split()`.
6. Define a KNN function to calculate Euclidean distances between the test image and training images.
7. Select the K nearest neighbors using `np.argsort()`.
8. Predict the class using majority voting with `np.bincount()` and `argmax()`.
9. Repeat the classification for K = 3, 5, and 7.
10. Calculate the accuracy using `accuracy_score()`.
11. Measure and display the execution time for each K value.
12. Compare the results.

## PROGRAM

```python
import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')

X = data.data[:2000] / 255.0
y = data.target[:2000].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=200, random_state=42
)

def knn(X_train, y_train, x, k):
    distance = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
    nearest = np.argsort(distance)[:k]
    labels = y_train[nearest]
    return np.bincount(labels).argmax()

for k in [3, 5, 7]:
    start = time.time()
    predictions = []

    for x in X_test:
        predictions.append(knn(X_train, y_train, x, k))

    accuracy = accuracy_score(y_test, predictions)
    end = time.time()

    print("K =", k)
    print("Accuracy =", accuracy)
    print("Time =", round(end - start, 2), "seconds")
    print()
```

## OUTPUT

```text
K = 3
Accuracy = 0.805
Time = 1.74 seconds

K = 5
Accuracy = 0.825
Time = 1.72 seconds

K = 7
Accuracy = 0.835
Time = 1.73 seconds
```

*Note: Actual execution time and accuracy may vary depending on the execution environment and dataset.*

## RESULT

The K-Nearest Neighbors algorithm was implemented using the Fashion-MNIST dataset. The classification accuracy and execution time were evaluated for K = 3, 5, and 7. Among the reported results, K = 7 achieved the highest accuracy of 83.5%.
