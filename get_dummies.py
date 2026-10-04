import pandas as pd
import numpy as np

df = pd.read_csv('town_prices.csv')

r = pd.get_dummies(df['town'])

merged = pd.concat([df, r], axis='columns')
final = merged.drop(['town'], axis='columns')


from sklearn import linear_model
model = linear_model.LinearRegression()
r = model.fit(final[['area',  'Ankara', 'İstanbul', 'İzmir']], final.price)
p = r.predict([[3000,  0, 1, 0]])
print("Predicted price for 3000 sq ft area, 3 bedrooms, 40 years old in İstanbul:", p)