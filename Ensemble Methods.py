from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])

model1 = LogisticRegression()
model2 = DecisionTreeClassifier()
model3 = SVC(probability=True)

ensemble = VotingClassifier(estimators=[('lr', model1), ('dt', model2), ('svc', model3)], voting='soft')
ensemble.fit(X, y)

predictions = ensemble.predict(X)
print("Predictions:", predictions)
