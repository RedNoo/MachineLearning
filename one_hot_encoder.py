import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer


from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_csv('town_prices.csv')


le = LabelEncoder()
dfle = df.copy()
dfle['town'] = le.fit_transform(dfle['town'])
X = dfle[['town', 'area']].values
y = dfle['price'].values



ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0])],
    remainder="passthrough",
)
X = np.array(ct.fit_transform(X))

X = X[:, 1:]  # Avoiding the dummy variable trap
print(X)
model = linear_model.LinearRegression()
model.fit(X, y)
r = model.predict([[1, 0, 3000]])  # Predicting for İstanbul with 3000 sq ft area


