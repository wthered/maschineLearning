import datetime
import math
# Video #6
import pickle

import matplotlib.pyplot as plt
import numpy as np
import quandl
from matplotlib import style
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume', ]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']
df['CH_PCT'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']
df = df[['Adj. Close', 'HL_PCT', 'CH_PCT', 'Adj. Volume']]

forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True)

forecast_out = math.ceil(0.01 * len(df))

df['label'] = df[forecast_col].shift(-forecast_out)
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, train_size=0.2)

# Classifier with as many jobs as my processor can handle
clit_line = LinearRegression(n_jobs=-1)
clit_line.fit(X_train, y_train)
with open('linearregression.picle', 'wb') as f:
	pickle.dump(clit_line, f)

picle_in = open('linearregression.picle', 'rb')
clit_line = pickle.load(picle_in)

accuracy_line = clit_line.score(X_test, y_test)

forecast_set = clit_line.predict(X_lately)
print(forecast_set, accuracy_line, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
