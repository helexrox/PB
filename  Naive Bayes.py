from sklearn.naive_bayes import GaussianNB
import numpy as np

# Example data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])

# Model
model = GaussianNB()
model.fit(X, y)

# Predictions
predictions = model.predict(X)
print("Predictions:", predictions)
