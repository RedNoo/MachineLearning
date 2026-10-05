import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("insurance_age.csv", sep=',')
print(df.head())
plt.scatter(df['age'], df['have_insurance'], color='red', marker='*')
plt.savefig('insurance_age_plot.png')

from sklearn.model_selection import train_test_split
x = df[['age']]
y = df['have_insurance']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print("Training data size:", x_train.shape)
print("Testing data size:", x_test.shape)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(x_test)
print("Predicted insurance status:", y_pred)    
print(model.score(x_test, y_test))
print(model.predict_proba(x_test))