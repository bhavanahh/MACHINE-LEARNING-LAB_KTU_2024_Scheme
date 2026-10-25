# PROGRAM NO. 9 – ML

**ROLL NO: 56**

## AIM

Implement and train a Multilayer Feed-Forward Network (MLP) on the Wine Quality dataset. Experiment with different numbers of hidden layers and neurons, and analyze how these choices affect the network's performance.

## TASKS

* Load and preprocess the Wine Quality dataset.
* Design and implement an MLP with varying hidden layers and neurons.
* Train and evaluate the network.
* Discuss the impact of architecture choices on performance.

## ALGORITHM

1. Load the Wine Quality (Red Wine) dataset from the CSV file.
2. Separate the dataset into input features `X` and quality labels `y`.
3. Convert the wine quality scores into three classes:
   * 0 → Low Quality
   * 1 → Medium Quality
   * 2 → High Quality
4. Split the dataset into 80% training and 20% testing data using stratified sampling.
5. Standardize the input features using `StandardScaler`.
6. Define different MLP architectures with hidden layers `(5)`, `(10)`, `(20)`, and `(10, 5)`.
7. Create an `MLPClassifier` using the ReLU activation function and Adam optimizer.
8. Train the MLP separately for each architecture.
9. Predict the wine quality classes using the test data.
10. Calculate the classification accuracy for each architecture.
11. Compare the accuracy obtained for different numbers of hidden layers and neurons.
12. Analyze how increasing the number of neurons or hidden layers affects the learning ability and performance of the network.

## PROGRAM

```python
# Import required libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# Task 1: Load Dataset
data = pd.read_csv('winequality-red.csv', sep=';')

X = data.drop('quality', axis=1)
y = data['quality']


# Task 2: Convert Quality Scores into Classes
y = pd.cut(y, bins=[0, 5, 7, 10], labels=[0, 1, 2])

print("Class distribution:")
print(y.value_counts())


# Task 3: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Task 4: Standardize Features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Task 5: Define MLP Architectures
architectures = [
    (5,),
    (10,),
    (20,),
    (10, 5)
]


# Task 6: Train and Evaluate Models
for i, arch in enumerate(architectures):

    model = MLPClassifier(
        hidden_layer_sizes=arch,
        activation='relu',
        solver='adam',
        max_iter=2000,
        random_state=i
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nArchitecture:", arch)
    print("Accuracy:", round(
        accuracy_score(y_test, y_pred) * 100, 2
    ), "%")
```

## RESULT

The Wine Quality dataset was loaded and preprocessed. The wine quality scores were converted into three classes, and the input features were standardized. Multilayer Feed-Forward Networks with different architectures were trained and evaluated. The classification accuracies were compared to analyze the effect of the number of hidden layers and neurons on model performance.
