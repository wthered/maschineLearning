import pandas as pd
import quandl
import math
import numpy as np

from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

# quandl.ApiConfig.api_key = '_L6ktq9Fq7jNDqPFtXsG'

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume', ]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']
df['CH_PCT'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']
df = df[['Adj. Close', 'HL_PCT', 'CH_PCT', 'Adj. Volume']]

forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

print(df.tail())

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

X = preprocessing.scale(X)

print("*******************************")
# print(len(X), len(y))
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, train_size=0.2)

# Classifier with as many jobs as my processor can handle
clit_line = LinearRegression(n_jobs=-1)
clit_line.fit(X_train, y_train)
accuracy_line = clit_line.score(X_test, y_test)

clit_sup = svm.SVR()
clit_sup.fit(X_train, y_train)
accuracy_clit = clit_sup.score(X_test, y_test)
print(accuracy_line, accuracy_clit)
