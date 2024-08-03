from sklearn.linear_model import LinearRegression
import numpy as np

# Example data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 3, 3, 2, 5])

# Model
model = LinearRegression()
model.fit(X, y)

# Predictions
predictions = model.predict(X)
print("Predictions:", predictions)
