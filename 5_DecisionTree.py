# Learn Decision trees for regression and classification problem  
# a. Split the data set into training and test sets.  
# b. Build the decision tree  
# c. Check model performances on training and test data sets. 
# d. Apply cost complexity pruning to overcome overfitting problem 
# e. Apply Random Forest algorithm to overcome overfitting problem.  
# f. Apply Ada-boost ensemble method on Decision stumps. 


import numpy as np
import pandas as pd
import matplotlib as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report


# Load iris dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Feature Names:", iris.feature_names)
print("Target Names:", iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

print("Training size:", X_train.shape)
print("Test size:", X_test.shape)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predictions
train_pred = dt_model.predict(X_train)
test_pred = dt_model.predict(X_test)

print("Decision Tree Training Accuracy:",
      accuracy_score(y_train, train_pred))
print("Decision Tree Test Accuracy:",
      accuracy_score(y_test, test_pred))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12,8))
plot_tree(dt_model,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True)
plt.show()

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42)

rf_model.fit(X_train, y_train)

print("Random Forest Test Accuracy:",
      rf_model.score(X_test, y_test))

models = {
    "Decision Tree": dt_model.score(X_test, y_test),
    "Random Forest": rf_model.score(X_test, y_test),
}

for model, score in models.items():
    print(model, ":", round(score, 4))

print(classification_report(y_test, test_pred))

print("RF Train Accuracy:", rf_model.score(X_train, y_train))
print("RF Test Accuracy:", rf_model.score(X_test, y_test))

y_reg = iris.data[:, 3]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.3, random_state=42)

# Decision Tree Regressor
dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg.fit(X_train_r, y_train_r)
y_pred_dt = dt_reg.predict(X_test_r)
print("\nDecision Tree Regressor MSE:", round(mean_squared_error(y_test_r, y_pred_dt), 3))
print("Decision Tree Regressor R²:", round(r2_score(y_test_r, y_pred_dt), 3))

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_r, y_train_r)
y_pred_rf = rf_reg.predict(X_test_r)
print("Random Forest Regressor MSE:", round(mean_squared_error(y_test_r, y_pred_rf), 3))
print("Random Forest Regressor R²:", round(r2_score(y_test_r, y_pred_rf), 3))














import numpy as np
import pandas as pd
import matplotlib.pyplot as plt                      # Fixed: was 'import matplotlib as plt'

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report

# ── LOAD DATA ─────────────────────────────────────────────────
iris = load_iris()
X    = iris.data
y    = iris.target

print("Feature Names:", iris.feature_names)
print("Target Names :", iris.target_names)

# ── a. SPLIT DATA ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

print("\nTraining size:", X_train.shape)
print("Test size    :", X_test.shape)

# ── b. BUILD DECISION TREE ────────────────────────────────────
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Visualize Tree
plt.figure(figsize=(12, 8))
plot_tree(dt_model,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True)
plt.title("Decision Tree")
plt.show()

# ── c. CHECK MODEL PERFORMANCE ────────────────────────────────
train_pred = dt_model.predict(X_train)
test_pred  = dt_model.predict(X_test)

print("\n── Decision Tree Performance ──")
print(f"  Train Accuracy : {accuracy_score(y_train, train_pred) * 100:.2f}%")
print(f"  Test  Accuracy : {accuracy_score(y_test,  test_pred)  * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, test_pred,
      target_names=iris.target_names))

# ── d. COST COMPLEXITY PRUNING ────────────────────────────────  # Fixed: was missing
print("\n── Cost Complexity Pruning ──")

# Get pruning path
path       = dt_model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]                   # Remove last (trivial tree)

train_scores, test_scores = [], []

for alpha in ccp_alphas:
    dt_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    dt_pruned.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, dt_pruned.predict(X_train)))
    test_scores.append(accuracy_score(y_test,  dt_pruned.predict(X_test)))

# Plot pruning effect
plt.figure(figsize=(8, 4))
plt.plot(ccp_alphas, train_scores, marker='o', label='Train Accuracy')
plt.plot(ccp_alphas, test_scores,  marker='s', label='Test Accuracy')
plt.xlabel("CCP Alpha")
plt.ylabel("Accuracy")
plt.title("Accuracy vs CCP Alpha (Pruning)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Best pruned model
best_alpha   = ccp_alphas[np.argmax(test_scores)]
dt_best      = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
dt_best.fit(X_train, y_train)
print(f"  Best Alpha     : {best_alpha:.4f}")
print(f"  Pruned Train   : {accuracy_score(y_train, dt_best.predict(X_train)) * 100:.2f}%")
print(f"  Pruned Test    : {accuracy_score(y_test,  dt_best.predict(X_test))  * 100:.2f}%")

# ── e. RANDOM FOREST ──────────────────────────────────────────
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("\n── Random Forest Performance ──")
print(f"  Train Accuracy : {rf_model.score(X_train, y_train) * 100:.2f}%")
print(f"  Test  Accuracy : {rf_model.score(X_test,  y_test)  * 100:.2f}%")

# ── f. ADABOOST ON DECISION STUMPS ───────────────────────────  # Fixed: was missing
stump    = DecisionTreeClassifier(max_depth=1, random_state=42) # Decision Stump = depth 1
ada_model = AdaBoostClassifier(estimator=stump,
                                n_estimators=100,
                                random_state=42)
ada_model.fit(X_train, y_train)

print("\n── AdaBoost Performance ──")
print(f"  Train Accuracy : {ada_model.score(X_train, y_train) * 100:.2f}%")
print(f"  Test  Accuracy : {ada_model.score(X_test,  y_test)  * 100:.2f}%")

# ── MODEL COMPARISON ──────────────────────────────────────────
print("\n── Model Comparison ──")
models = {
    "Decision Tree (Unpruned)" : dt_model.score(X_test, y_test),
    "Decision Tree (Pruned)"   : dt_best.score(X_test,  y_test),
    "Random Forest"            : rf_model.score(X_test, y_test),
    "AdaBoost"                 : ada_model.score(X_test, y_test),
}
for name, score in models.items():
    print(f"  {name:<28} : {score * 100:.2f}%")

# ── REGRESSION (Petal Width prediction) ───────────────────────
print("\n── Regression (Predicting Petal Width) ──")
y_reg = iris.data[:, 3]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.3, random_state=42)

# Decision Tree Regressor
dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg.fit(X_train_r, y_train_r)
y_pred_dt = dt_reg.predict(X_test_r)

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_r, y_train_r)
y_pred_rf = rf_reg.predict(X_test_r)

print(f"  DT  MSE : {mean_squared_error(y_test_r, y_pred_dt):.3f} | R²: {r2_score(y_test_r, y_pred_dt):.3f}")
print(f"  RF  MSE : {mean_squared_error(y_test_r, y_pred_rf):.3f} | R²: {r2_score(y_test_r, y_pred_rf):.3f}")