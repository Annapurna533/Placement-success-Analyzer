import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Make results reproducible
np.random.seed(42)

# ==============================
# CREATE DATASET
# ==============================

n = 200

CGPA = np.random.uniform(6, 9.8, n)
DSA_Hours = np.random.randint(10, 250, n)
Aptitude = np.random.randint(40, 100, n)
Internship = np.random.randint(0, 2, n)
Communication = np.random.randint(4, 10, n)
Projects = np.random.randint(1, 7, n)

Placed = []

for i in range(n):
    score = (
        (CGPA[i] * 2) +
        (DSA_Hours[i] * 0.02) +
        (Aptitude[i] * 0.02) +
        (Internship[i] * 2) +
        (Communication[i] * 0.5) +
        (Projects[i] * 0.3)
    )

    score += np.random.normal(0, 2)

    if score > 30:
        Placed.append(1)
    else:
        Placed.append(0)

df = pd.DataFrame({
    "CGPA": CGPA,
    "DSA_Hours": DSA_Hours,
    "Aptitude": Aptitude,
    "Internship": Internship,
    "Communication": Communication,
    "Projects": Projects,
    "Placed": Placed
})

# ==============================
# TRAIN TEST SPLIT
# ==============================

X = df[["CGPA", "DSA_Hours", "Aptitude",
        "Internship", "Communication", "Projects"]]

y = df["Placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# LOGISTIC REGRESSION
# ==============================

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

log_predictions = log_model.predict(X_test)
log_accuracy = accuracy_score(y_test, log_predictions)

print("\nLogistic Regression Accuracy:", round(log_accuracy, 3))

# ==============================
# DECISION TREE
# ==============================

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

print("Decision Tree Accuracy:", round(dt_accuracy, 3))

# ==============================
# SAMPLE PROBABILITIES
# ==============================

probabilities = log_model.predict_proba(X_test)

print("\nSample Prediction Results:\n")

for i in range(5):
    print("Student", i+1)
    print("Placement Probability:", round(probabilities[i][1]*100, 2), "%")
    print("Not Placed Probability:", round(probabilities[i][0]*100, 2), "%")
    print("------")

# ==============================
# FEATURE IMPORTANCE
# ==============================

print("\nFeature Importance (Logistic Regression):")

importance = log_model.coef_[0]

for i, col in enumerate(X.columns):
    print(col, ":", round(importance[i], 3))

# ==============================
# FEATURE IMPORTANCE GRAPH
# ==============================

plt.figure()
plt.bar(X.columns, importance)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==============================
# MODEL ACCURACY COMPARISON GRAPH
# ==============================

models = ["Logistic Regression", "Decision Tree"]
accuracies = [log_accuracy, dt_accuracy]

plt.figure()
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# ==============================
# SAFE USER INPUT SECTION
# ==============================

print("\nEnter student details:")

try:
    cgpa = float(input("CGPA: "))
    dsa = int(input("DSA Hours: "))
    apt = int(input("Aptitude Score: "))
    intern = int(input("Internship (1 = Yes, 0 = No): "))
    comm = int(input("Communication Skill (1-10): "))
    proj = int(input("Number of Projects: "))

    new_data = pd.DataFrame([[cgpa, dsa, apt, intern, comm, proj]],
                            columns=["CGPA", "DSA_Hours", "Aptitude",
                                     "Internship", "Communication", "Projects"])

    prediction = log_model.predict(new_data)
    prob = log_model.predict_proba(new_data)

    print("\nPrediction:", "Placed" if prediction[0] == 1 else "Not Placed")
    print("Placement Probability:", round(prob[0][1]*100, 2), "%")

except ValueError:
    print("\nInvalid input! Please enter numbers only.")