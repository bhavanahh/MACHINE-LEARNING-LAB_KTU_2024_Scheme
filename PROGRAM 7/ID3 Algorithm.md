# PROGRAM NO. 13 – ML

**Roll No: 56**

## AIM
To implement a Decision Tree classifier using the ID3 algorithm on the Online Retail dataset to segment customers based on their purchasing behavior, visualize the tree structure, and analyze feature importance.

## TASKS
1. Load and preprocess the Online Retail dataset.
2. Calculate the total purchase amount for each transaction.
3. Group transactions by CustomerID.
4. Calculate total quantity, total spending, and purchase frequency.
5. Classify customers into Low Value and High Value segments.
6. Implement a Decision Tree classifier using entropy.
7. Evaluate the model using accuracy and a classification report.
8. Visualize the decision tree and analyze feature importance.

## ALGORITHM
1. Import the required libraries.
2. Load the `online_retail.csv` dataset.
3. Remove records containing missing values.
4. Calculate the total transaction amount using `Quantity × UnitPrice`.
5. Group transactions by CustomerID.
6. Calculate Total Quantity, Total Spent, and Purchase Frequency for each customer.
7. Create the target variable `Segment` using the median of Total Spent:
   - `0` – Low Value Customer
   - `1` – High Value Customer
8. Select Total Quantity, Purchase Frequency, and Total Spent as input features.
9. Split the dataset into 80% training data and 20% testing data.
10. Create a Decision Tree classifier using entropy with a maximum depth of 3.
11. Train the model and predict customer segments.
12. Evaluate the model using accuracy and a classification report.
13. Calculate feature importance for each input feature.
14. Visualize the decision tree and analyze its structure.

## PROGRAM

```python
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "OnlineRetail.csv"),
    encoding="ISO-8859-1"
)

print("Data Columns:", data.columns)

data = data.rename(columns={
    "Invoice": "InvoiceNo",
    "Price": "UnitPrice",
    "Customer ID": "CustomerID"
})

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

The program displays:
1. Dataset column names.
2. Model accuracy.
3. Classification report with precision, recall, F1-score, and support.
4. Feature importance scores.
5. A visualization of the trained decision tree.

*Note: Exact numerical output depends on the dataset and execution environment.*

## RESULT

The Decision Tree classifier was implemented using the Online Retail dataset to segment customers based on purchasing behavior. The model was evaluated using accuracy and a classification report. Feature importance and the decision tree visualization were used to analyze the purchasing features contributing to customer segmentation.
