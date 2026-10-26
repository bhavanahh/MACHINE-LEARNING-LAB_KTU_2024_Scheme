# PROGRAM NO. 8 – ML

**Roll No: 56**

## Linear Support Vector Machine on Iris Dataset

## AIM

To implement a Linear Support Vector Machine (SVM) to classify the Iris dataset, visualize the decision boundary and margins, and understand how the margin influences classification.

## TASKS

- Load and preprocess the Iris dataset.
- Implement a Linear SVM for binary classification (Setosa vs. Non-Setosa).
- Visualize the decision boundary and margin lines.
- Identify and highlight the support vectors.
- Discuss how the maximum-margin hyperplane separates the two classes.

## ALGORITHM

1. Load the Iris dataset using `load_iris()`.
2. Select Sepal Length and Sepal Width as input features.
3. Convert the target into binary classification:
   - Setosa → `+1`
   - Non-Setosa → `-1`
4. Split the dataset into 80% training data and 20% testing data using `random_state=42`.
5. Create a linear SVM classifier with `C=1`.
6. Train the SVM model using the training data.
7. Predict the classes of the test data.
8. Calculate the classification accuracy.
9. Obtain the weight vector `w` and bias `b` from the trained model.
10. Calculate the decision boundary using:

    `w1*x1 + w2*x2 + b = 0`

11. Calculate the two margin lines using:

    `w1*x1 + w2*x2 + b = +1`

    `w1*x1 + w2*x2 + b = -1`

12. Identify and highlight the support vectors.
13. Plot the Setosa and Non-Setosa training points, decision boundary, margins, and support vectors.
14. Display the graph and observe how the maximum-margin hyperplane separates the two classes.

## PROGRAM

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data[:, :2]
y = iris.target
y = np.where(y == 0, 1, -1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

w = model.coef_[0]
b = model.intercept_[0]

print("Weights:", w)
print("Bias:", b)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_train[y_train == 1, 0],
    X_train[y_train == 1, 1],
    label="Setosa"
)
plt.scatter(
    X_train[y_train == -1, 0],
    X_train[y_train == -1, 1],
    label="Non-Setosa"
)

x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

y_boundary = -(w[0] * x_values + b) / w[1]
y_margin1 = -(w[0] * x_values + b - 1) / w[1]
y_margin2 = -(w[0] * x_values + b + 1) / w[1]

plt.plot(x_values, y_boundary, label="Decision Boundary")
plt.plot(x_values, y_margin1, '--', label="Margin")
plt.plot(x_values, y_margin2, '--')

plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=100,
    facecolors='none',
    edgecolors='black',
    label="Support Vectors"
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Linear SVM - Setosa vs Non-Setosa")
plt.legend()
plt.grid()
plt.show()
```

## OUTPUT

```text
Accuracy: 1.0
Weights: [-2.44999622  2.45000378]
Bias: 5.409971273612131
```

The output graph displays:

- Setosa and Non-Setosa training data points.
- The linear decision boundary separating the classes.
- The two margin lines.
- Support vectors highlighted with unfilled circles.

## RESULT

A Linear Support Vector Machine was implemented on the Iris dataset to classify Setosa and Non-Setosa using Sepal Length and Sepal Width. The model achieved **100% accuracy** on the test set for the shown run. The decision boundary, margin lines, and support vectors were visualized to illustrate maximum-margin classification.
