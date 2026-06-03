import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset/spam.csv")

print(data.head())

# Convert labels
data['label'] = data['label'].map({
    'ham':0,
    'spam':1
})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'],
    data['label'],
    test_size=0.2,
    random_state=42
)

# Convert text into numbers
vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test accuracy
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))