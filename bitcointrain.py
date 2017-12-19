import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style

df = quandl.get("BCHARTS/bitstampUSD")
df = df[['Open',  'High',  'Low',  'Close', 'Volume (Currency)']]
df['HL_PCT'] = (df['High'] - df['Low']) / df['Low'] * 100.0
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume (Currency)']]
print(df.head())

forecast_col='Close'
df.fillna(value=-99999, inplace=True)
forecast_out=10 #forecast Bitcoin Close price 10 dads into the future

df['label']=df[forecast_col].shift(-forecast_out) #Bitcoin Close price 10 days into future

X=np.array(df.drop(['label'],1))
X =preprocessing.scale(X)
X_lately=X[-forecast_out:]
X=X[:-forecast_out]

df.dropna(inplace=True)

y=np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2) #train here
clf = LinearRegression(n_jobs=-1) #Linear Regression using multithreading
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)
forecast_set = clf.predict(X_lately)
print(forecast_set, confidence, forecast_out)

style.use('ggplot')
df['Forecast']=np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()