# PROGRAM NO. 10 – ML

**Roll No: 56**

## AIM
To implement and compare the performance of a neural network using different activation functions (Sigmoid, ReLU, and Tanh) on the handwritten digit dataset and analyze their training time, convergence, and classification accuracy.

## TASKS
1. Load and preprocess the handwritten digit dataset.
2. Normalize the pixel values.
3. Implement neural networks using Sigmoid, ReLU, and Tanh activation functions.
4. Train and evaluate each neural network.
5. Compare their classification accuracy, training time, and number of iterations.

## ALGORITHM
1. Load the handwritten digit dataset using `load_digits()`.
2. Separate the dataset into input features `X` and target labels `y`.
3. Normalize the pixel values by dividing `X` by 16.
4. Split the dataset into 80% training data and 20% testing data.
5. Define the Sigmoid, ReLU, and Tanh activation functions.
6. Create a multilayer perceptron (MLP) with one hidden layer containing 50 neurons.
7. Train separate MLP models using Sigmoid, ReLU, and Tanh activation functions.
8. Record the training time and number of iterations for each model.
9. Predict the digit classes using the test data.
10. Calculate the classification accuracy of each model.
11. Compare the accuracy, training time, and number of iterations.
12. Analyze how the activation function affects neural network performance.

## PROGRAM

```python
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Task 1: Load the dataset
data = load_digits()
X = data.data
y = data.target

# Task 2: Normalize pixel values
X = X / 16.0

# Task 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Task 4: Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

print("Sigmoid:", sigmoid(2))
print("ReLU:", relu(2))
print("Tanh:", tanh(2))

# Task 5: Train and evaluate the models
activations = ['logistic', 'relu', 'tanh']

for act in activations:
    start = time.time()

    model = MLPClassifier(
        hidden_layer_sizes=(50,),
        activation=act,
        max_iter=500,
        tol=0.001,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    training_time = time.time() - start

    print("\nActivation:", act)
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Training Time:", round(training_time, 3), "seconds")
    print("Iterations:", model.n_iter_)
```

## RESULT
The neural network was implemented using Sigmoid, ReLU, and Tanh activation functions on the handwritten digit dataset. Each model was evaluated using classification accuracy, training time, and number of iterations.

The actual results depend on the execution environment and training process. ReLU often trains efficiently, while Sigmoid and Tanh may have different convergence behavior. The printed results can be used to compare the performance of all three activation functions.
