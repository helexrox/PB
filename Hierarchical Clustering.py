from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Example data
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

# Model
model = AgglomerativeClustering(n_clusters=2)
labels = model.fit_predict(X)

print("Labels:", labels)
