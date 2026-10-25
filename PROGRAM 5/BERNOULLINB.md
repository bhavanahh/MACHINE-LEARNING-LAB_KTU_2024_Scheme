# PROGRAM NO. 11 – ML

**Roll No: 56**

## AIM
To implement Bernoulli Naive Bayes for text classification using binary features and predict whether a message is Spam or Not Spam.

## TASKS
1. Create a training dataset containing text messages and their labels.
2. Convert text messages into binary features using `CountVectorizer`.
3. Train a Bernoulli Naive Bayes classifier.
4. Test the model using a new message.
5. Predict the message category.
6. Calculate and display the class probabilities.

## ALGORITHM
1. Import `CountVectorizer` and `BernoulliNB`.
2. Create a dataset containing Spam and Not Spam messages.
3. Assign the corresponding labels to the messages.
4. Convert the text messages into binary features using `CountVectorizer(binary=True)`.
5. Display the vocabulary and binary feature matrix.
6. Create and train the Bernoulli Naive Bayes model using the training data.
7. Create a test message containing `"free prize"`.
8. Transform the test message using the fitted vectorizer.
9. Predict whether the message is Spam or Not Spam.
10. Calculate and display the probability of each class.

## PROGRAM

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

# 1. Training Dataset
texts = [
    "free money now",
    "win prize now",
    "limited time offer",
    "project meeting tomorrow",
    "schedule the meeting",
    "discuss project details"
]

labels = [
    "Spam",
    "Spam",
    "Spam",
    "Not Spam",
    "Not Spam",
    "Not Spam"
]

# 2. Convert Text to Binary Features
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(texts)

print("Vocabulary:")
print(vectorizer.get_feature_names_out())

print("\nBinary Feature Matrix:")
print(X.toarray())

# 3. Train BernoulliNB
model = BernoulliNB()
model.fit(X, labels)

# 4. Test Message
test_message = ["free prize"]
X_test = vectorizer.transform(test_message)

# 5. Prediction
prediction = model.predict(X_test)

print("\nTest Message:")
print(test_message[0])

print("\nPredicted Class:")
print(prediction[0])

# 6. Class Probabilities
probabilities = model.predict_proba(X_test)

print("\nClass Probabilities:")
print(probabilities)

for cls, prob in zip(model.classes_, probabilities[0]):
    print(f"{cls}: {prob:.4f}")
```

## RESULT
The Bernoulli Naive Bayes classifier was implemented successfully using binary text features. The model predicts whether the test message belongs to the Spam or Not Spam category and displays the estimated probability of each class.
