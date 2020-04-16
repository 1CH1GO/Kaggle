"""
Created on Wed Apr 15 10:22:46 2020
@author: ichigo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = train.dropna()
test = test.dropna()

X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, 1:2].values

X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, 1:2].values

# bools = pd.isnull(y_test)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("X_train vs y_train")
plt.xlabel("X_train")
plt.ylabel("y_train")
plt.show()

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("X_test vs y_test")
plt.xlabel("X_test")
plt.ylabel("y_test")
plt.show()

accuracy = regressor.score(X_test, y_test)
print(accuracy)
