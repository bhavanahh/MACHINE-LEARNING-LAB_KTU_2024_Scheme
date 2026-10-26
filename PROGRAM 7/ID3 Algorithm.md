# PROGRAM NO. 13 – ML

**Roll No: 56**

## AIM
To implement a Decision Tree classifier using the ID3 algorithm on the Online Retail dataset to segment customers based on their purchasing behavior, visualize the tree structure, and analyze feature importance.

## TASKS
1. Load and preprocess the Online Retail dataset.
2. Calculate the total purchase amount for each transaction.
3. Group transactions by CustomerID and calculate customer-wise purchasing features.
4. Segment customers into Low Value and High Value categories.
5. Implement a Decision Tree classifier using entropy.
6. Evaluate the model using accuracy and classification report.
7. Analyze feature importance and visualize the decision tree.

## ALGORITHM
1. Load the Online Retail dataset from the CSV file.
2. Remove records containing missing values.
3. Calculate the total transaction amount using `Quantity × UnitPrice`.
4. Group transactions by CustomerID.
5. Calculate Total Quantity, Total Spent, and Purchase Frequency for each customer.
6. Create the target variable `Segment` based on the median of Total Spent:
   - `0` – Low Value Customer
   - `1` – High Value Customer
7. Select Total Quantity, Purchase Frequency, and Total Spent as input features.
8. Split the dataset into 80% training data and 20% testing data.
9. Create a Decision Tree classifier using entropy with a maximum depth of 3.
10. Train the model using the training data.
11. Predict customer segments for the testing data.
12. Evaluate the model using accuracy and a classification report.
13. Calculate and display feature importance.
14. Visualize the decision tree and analyze its structure.

## PROGRAM

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('OnlineRetail.csv', encoding="ISO-8859-1")

print("Data Columns:", data.columns)

data = data.dropna()
data["TotalAmount"] = data["Quantity"] * data["UnitPrice"]

print("Data Columns:", data.columns)

customer = data.groupby("CustomerID").agg({
    "Quantity": "sum",
    "TotalAmount": "sum",
    "InvoiceNo": "nunique"
}).reset_index()

customer.columns = [
    "CustomerID", "TotalQuantity", "TotalSpent", "PurchaseFrequency"
]

customer["Segment"] = (
    customer["TotalSpent"] > customer["TotalSpent"].median()
).astype(int)

X = customer[["TotalQuantity", "PurchaseFrequency", "TotalSpent"]]
y = customer["Segment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nFeature Importance")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(feature, ":", round(importance, 3))

plt.figure(figsize=(12, 8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Low Value", "High Value"],
    filled=True
)

plt.show()
```

## OUTPUT

The output includes:

1. Dataset column names.
2. Model accuracy.
3. Classification report containing precision, recall, F1-score, and support.
4. Feature importance scores for Total Quantity, Purchase Frequency, and Total Spent.
5. A visual representation of the trained decision tree.

*Note: Exact numerical results depend on the contents of `OnlineRetail.csv` and the execution environment.*

## RESULT

The Decision Tree classifier was implemented using the Online Retail dataset to segment customers based on purchasing behavior. The model was evaluated using accuracy and a classification report. Feature importance and the visualized decision tree were used to examine how purchasing features contribute to customer segmentation.
