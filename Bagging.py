from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Example data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])

# Model
model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10)
model.fit(X, y)

# Predictions
predictions = model.predict(X)
print("Predictions:", predictions)
