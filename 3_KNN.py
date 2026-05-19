# Write a program to implement k-Nearest Neighbour algorithm to classify the iris data set. Print 
# both correct and wrong predictions. Python ML library classes can be used for this problem. 


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("iris.csv")

data.head()

data.isnull().sum()

data.describe()

data.dtypes

X = data.iloc[:, 0:4]
y = data.iloc[:, 4].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

distances, indices = knn.kneighbors(X_test)

print("Distances of first test sample:", distances[0])
print("Nearest training indices:", indices[0])

print("Correct Predictions:")
for i in range(len(y_test)):
    if y_test.iloc[i] == y_pred[i]:
        print("Actual:", y_test.iloc[i], "Predicted:", y_pred[i])


print("\nWrong Predictions:")
for i in range(len(y_test)):
    if y_test.iloc[i] != y_pred[i]:
        print("Actual:", y_test.iloc[i], "Predicted:", y_pred[i])

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)













import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("iris.csv")
print(data.head())
print(data.isnull().sum())

# Fixed: drop 'Id' column, use column names directly
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data['Species']                                  # Fixed: use column name not index

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# KNN Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Print Correct Predictions
print("\n── Correct Predictions ──")
correct = 0
for i in range(len(y_test)):
    if y_test.iloc[i] == y_pred[i]:
        correct += 1
        print(f"  Sample {i+1:3} | Actual: {y_test.iloc[i]:<25} Predicted: {y_pred[i]}")

# Print Wrong Predictions
print("\n── Wrong Predictions ──")
wrong = 0
for i in range(len(y_test)):
    if y_test.iloc[i] != y_pred[i]:
        wrong += 1
        print(f"  Sample {i+1:3} | Actual: {y_test.iloc[i]:<25} Predicted: {y_pred[i]}")

if wrong == 0:
    print("  No wrong predictions!")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n── Results ──")
print(f"  Total Samples : {len(y_test)}")
print(f"  Correct       : {correct}")
print(f"  Wrong         : {wrong}")
print(f"  Accuracy      : {accuracy * 100:.2f}%")