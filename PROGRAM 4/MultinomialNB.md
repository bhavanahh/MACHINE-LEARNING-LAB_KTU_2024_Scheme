# PROGRAM NO. 7 – ML

**ROLL NO: 56**

## AIM

Implement a Naïve Bayes classifier to categorize text documents into topics using the 20 Newsgroups dataset. Compare the performance of Multinomial Naïve Bayes with Bernoulli Naïve Bayes.

## TASKS

* Load and preprocess the 20 Newsgroups dataset.
* Implement Multinomial Naïve Bayes and Bernoulli Naïve Bayes classifiers.
* Evaluate and compare the performance of both models using accuracy and F1-score.
* Discuss the strengths and weaknesses of each Naïve Bayes variant for text classification.

## ALGORITHM

1. Import the required libraries.
2. Load the 20 Newsgroups training and testing datasets.
3. Preprocess the text using CountVectorizer.
4. Convert text documents into numerical features using word counts.
5. Train the Multinomial Naïve Bayes classifier.
6. Convert text documents into binary word features using CountVectorizer.
7. Train the Bernoulli Naïve Bayes classifier.
8. Predict the categories of the test documents using both models.
9. Calculate accuracy, weighted F1-score, and classification reports for both models.
10. Compare the performance of Multinomial Naïve Bayes and Bernoulli Naïve Bayes.

## PROGRAM

```python
# Import required libraries
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, f1_score


# Task 1: Load Dataset
train_data = fetch_20newsgroups(
    subset='train',
    remove=('headers', 'footers', 'quotes')
)

test_data = fetch_20newsgroups(
    subset='test',
    remove=('headers', 'footers', 'quotes')
)

X_train = train_data.data
X_test = test_data.data

y_train = train_data.target
y_test = test_data.target


# Task 2: Text Preprocessing
vectorizer = CountVectorizer(
    stop_words='english',
    lowercase=True,
    max_features=10000
)

X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

print("Training Shape:", X_train_counts.shape)
print("Testing Shape:", X_test_counts.shape)


# Task 3: Train Multinomial Naive Bayes
mnb = MultinomialNB()

mnb.fit(X_train_counts, y_train)

mnb_pred = mnb.predict(X_test_counts)


# Task 4: Train Bernoulli Naive Bayes
binary_vectorizer = CountVectorizer(
    stop_words='english',
    lowercase=True,
    binary=True,
    max_features=10000
)

X_train_binary = binary_vectorizer.fit_transform(X_train)
X_test_binary = binary_vectorizer.transform(X_test)

bnb = BernoulliNB()

bnb.fit(X_train_binary, y_train)

bnb_pred = bnb.predict(X_test_binary)


# Task 5: Evaluate Multinomial Naive Bayes
mnb_accuracy = accuracy_score(y_test, mnb_pred)
mnb_f1 = f1_score(y_test, mnb_pred, average='weighted')

print("\nMultinomial Naive Bayes")
print("-----------------------")
print("Accuracy:", mnb_accuracy)
print("Weighted F1:", mnb_f1)

print(classification_report(
    y_test,
    mnb_pred,
    target_names=test_data.target_names
))


# Task 6: Evaluate Bernoulli Naive Bayes
bnb_accuracy = accuracy_score(y_test, bnb_pred)
bnb_f1 = f1_score(y_test, bnb_pred, average='weighted')

print("\nBernoulli Naive Bayes")
print("---------------------")
print("Accuracy:", bnb_accuracy)
print("Weighted F1:", bnb_f1)

print(classification_report(
    y_test,
    bnb_pred,
    target_names=test_data.target_names
))


# Task 7: Compare the Models
comparison = pd.DataFrame({
    "Model": ["Multinomial NB", "Bernoulli NB"],
    "Accuracy": [mnb_accuracy, bnb_accuracy],
    "Weighted F1 Score": [mnb_f1, bnb_f1]
})

print("\nPerformance Comparison:")
print(comparison)
```

## RESULT

The 20 Newsgroups dataset was loaded and preprocessed. Multinomial Naïve Bayes and Bernoulli Naïve Bayes classifiers were implemented to categorize text documents into topics. Both models were evaluated using accuracy, weighted F1-score, and classification reports. Their performance was compared to understand the effectiveness of each classifier for text classification.
