# Write a program to implement the naïve Bayesian classifier for a sample training data set stored 
# as a .CSV file. Compute the accuracy of the classifier, considering few test data sets. 


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("/content/spam_and_ham_classification.csv")

data.head()

data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

X = data['text']
y = data['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

new_messages = [
    "Congratulations! You have won a free ticket!",
    "Are we meeting at 4pm today?"
]

new_vec = vectorizer.transform(new_messages)
predictions = model.predict(new_vec)

for msg, pred in zip(new_messages, predictions):
    label = "Spam" if pred == 1 else "Ham"
    print(f"Text: {msg} → {label}")
















    import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ── LOAD DATA ─────────────────────────────────────────────────
data = pd.read_csv("/content/spam_and_ham_classification.csv.zip")
print(data.head())
print("\nShape:", data.shape)
print("\nMissing Values:\n", data.isnull().sum())

# ── PREPROCESSING ─────────────────────────────────────────────
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

X = data['text']
y = data['label_num']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── TRAIN NAIVE BAYES MODEL ───────────────────────────────────
model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# ── CORRECT PREDICTIONS ───────────────────────────────────────
print("\n── Correct Predictions (first 10) ──")
correct = 0
for i in range(len(y_test)):
    actual    = 'Spam' if y_test.iloc[i] == 1 else 'Ham'
    predicted = 'Spam' if y_pred[i]      == 1 else 'Ham'
    if y_test.iloc[i] == y_pred[i]:
        correct += 1
        if correct <= 10:                            # Print only first 10 to keep output clean
            print(f"  Sample {i+1:4} | Actual: {actual:<6} | Predicted: {predicted}")

# ── WRONG PREDICTIONS ─────────────────────────────────────────
print("\n── Wrong Predictions ──")
wrong = 0
for i in range(len(y_test)):
    actual    = 'Spam' if y_test.iloc[i] == 1 else 'Ham'
    predicted = 'Spam' if y_pred[i]      == 1 else 'Ham'
    if y_test.iloc[i] != y_pred[i]:
        wrong += 1
        print(f"  Sample {i+1:4} | Actual: {actual:<6} | Predicted: {predicted}")

if wrong == 0:
    print("  No wrong predictions!")

# ── ACCURACY ──────────────────────────────────────────────────
accuracy = accuracy_score(y_test, y_pred)
print(f"\n── Results ──")
print(f"  Total Samples : {len(y_test)}")
print(f"  Correct       : {correct}")
print(f"  Wrong         : {wrong}")
print(f"  Accuracy      : {accuracy * 100:.2f}%")
print(f"\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# ── TEST ON NEW MESSAGES ──────────────────────────────────────
print("\n── Custom Message Predictions ──")
new_messages = [
    "Congratulations! You have won a free ticket!",
    "Are we meeting at 4pm today?",
    "Win cash prize now! Click the link",
    "Can you send me the notes please?"
]

new_vec     = vectorizer.transform(new_messages)
predictions = model.predict(new_vec)

for msg, pred in zip(new_messages, predictions):
    label = "Spam" if pred == 1 else "Ham"
    print(f"  {label:<6} ← {msg}")