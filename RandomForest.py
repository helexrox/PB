from sklearn.ensemble import RandomForestClassifier
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])

model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

predictions = model.predict(X)
print("Predictions:", predictions)
