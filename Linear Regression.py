from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 3, 3, 2, 5])

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)
print("Predictions:", predictions)
