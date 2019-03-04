import numpy as np
import pandas as pd
from sklearn import model_selection, neighbors

df = pd.read_csv('breast-cancer.csv')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Prediction is {}%".format(round(100 * accuracy, 2)))

example = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])
example = example.reshape(1, -1)
prediction = clf.predict(example)
print("The Prediction is {}".format(prediction[0]))
