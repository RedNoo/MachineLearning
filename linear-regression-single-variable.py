import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("homeprice.csv", sep=';')
print(df.head())

plt.scatter(df['area'], df['price'], color='red', marker='*')
plt.savefig('homeprice_plot.png')

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price) 
p = reg.predict([[3300]])
print("Predicted price for 3300 sq ft area:", p)

plt.scatter(df['area'], df['price'], color='red', marker='*')
plt.plot(df['area'], reg.predict(df[['area']]), color='blue')
plt.savefig('homepredictedprice_plot.png')


d = pd.read_csv("areas.csv", sep=';')
p = reg.predict(d[['area']])
print("Predicted prices for the given areas:", p)
d['price'] = p
d.to_csv("predicted_prices.csv", index=False)