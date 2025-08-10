import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------
# Decision Tree Classifier
# -----------------------------
dtree = DecisionTreeClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_dt = {
    'max_depth': [2, 4, 6, None],
    'min_samples_split': [2, 5, 10]
}
grid_dt = GridSearchCV(dtree, param_dt, cv=5)
grid_dt.fit(X_train, y_train)

# Best Decision Tree
best_dtree = grid_dt.best_estimator_
dtree_pred = best_dtree.predict(X_test)
dtree_acc = accuracy_score(y_test, dtree_pred)

# -----------------------------
# Random Forest Classifier
# -----------------------------
rforest = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_rf = {
    'n_estimators': [10, 50, 100],
    'max_depth': [2, 4, 6, None],
    'min_samples_split': [2, 5, 10]
}
grid_rf = GridSearchCV(rforest, param_rf, cv=5)
grid_rf.fit(X_train, y_train)

# Best Random Forest
best_rforest = grid_rf.best_estimator_
rf_pred = best_rforest.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# -----------------------------
# Results
# -----------------------------
print("Decision Tree Accuracy:", dtree_acc)
print("Random Forest Accuracy:", rf_acc)

print("\nClassification Report - Decision Tree:\n", classification_report(y_test, dtree_pred))
print("\nClassification Report - Random Forest:\n", classification_report(y_test, rf_pred))

# Optional: Show best hyperparameters
print("Best Decision Tree Parameters:", grid_dt.best_params_)
print("Best Random Forest Parameters:", grid_rf.best_params_)
