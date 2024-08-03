from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])

model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10)
model.fit(X, y)

predictions = model.predict(X)
print("Predictions:", predictions)
