import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

baseline = LogisticRegression(max_iter=5000)
baseline.fit(X_train, y_train)

preds = baseline.predict(X_test)
print("Baseline Model")
print(classification_report(y_test, preds))

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print("Random Forest")
print(classification_report(y_test, rf_preds))

gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)
print("Gradient Boosting")
print(classification_report(y_test, gb_preds))

estimators = [
    ("rf", RandomForestClassifier()),
    ("svm", SVC(probability=True))
]
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)
stack_model.fit(X_train, y_train)
stack_preds = stack_model.predict(X_test)
print("Stacking Model")
print(classification_report(y_test, stack_preds))

models = {
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "Stacking": stack_model
}

for name, model in models.items():
    prob = model.predict_proba(X_test)[:,1]
    score = roc_auc_score(y_test, prob)
    print(name, "ROC AUC:", score)

best_model = stack_model
preds = best_model.predict(X_test)

cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()

results = []

models = {
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "Stacking": stack_model
}

for name, model in models.items():
    preds = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]

    accuracy = (preds == y_test).mean()
    roc = roc_auc_score(y_test, prob)

    results.append([name, accuracy, roc])

comparison_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "ROC AUC"]
)

print(comparison_df)

from sklearn.metrics import RocCurveDisplay

plt.figure()

for name, model in models.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name)

plt.title("ROC Curve Comparison")
plt.show()

importances = rf_model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print(importance_df.head())

sns.barplot(data=importance_df.head(10), x="Importance", y="Feature")
plt.title("Top Feature Importance")
plt.show()

"""Advantages & Disadvantages

#Ensemble Method Comparison
* Random Forest (Bagging)

Advantages:

Reduces overfitting

Handles high-dimensional data well

Robust and stable

Disadvantages:

Can be slow with many trees

Less interpretable than simple models

* Gradient Boosting (Boosting)

Advantages:

High predictive accuracy

Learns complex patterns

Focuses on difficult samples

Disadvantages:

Sensitive to noise

Can overfit if not tuned properly

Training can be slow

* Stacking

Advantages:

Combines strengths of multiple models

Often provides best performance

Flexible architecture

Disadvantages:

Computationally expensive

Complex to implement

Harder to interpret
"""
