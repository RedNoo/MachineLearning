import pandas as pd
import numpy as np
from sklearn import linear_model
import math

df = pd.read_csv("homemultyprice.csv", sep=';')

print(df.head)

df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")

print("Median number of bedrooms:", df.bedrooms.median())

m = math.floor(df.bedrooms.median())

print("Median number of bedrooms (rounded down):", m)

df.bedrooms = df.bedrooms.fillna(m)

print(df.head)

reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df.price) #trarining the model
print(reg.predict([[3000, 3, 40]])) #predicting the price of a home with 3000 sq ft area, 3 bedrooms, and 40 years old

import pickle 
with open('model_pickle', 'wb') as f:
    pickle.dump(reg, f)

model = pickle.load(open('model_pickle', 'rb'))
r = model.predict([[3000, 3, 40]])
print(r)

import joblib   
joblib.dump(reg, 'model_joblib')
model = joblib.load('model_joblib')
r = model.predict([[3000, 3, 40]])
print(r)
