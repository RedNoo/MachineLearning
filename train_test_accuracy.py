import pandas as pd
df = pd.read_csv("car_price.csv", sep=',')
print(df.head())

import matplotlib.pyplot as plt
# plt.scatter(df['Mileage'], df['Price'], color='red', marker='*')
# plt.savefig('carprice_plot.png')

plt.scatter(df['Age'], df['Price'], color='blue', marker='*')
plt.savefig('carprice_plot_age.png')

x = df[['Mileage', 'Age']]
y = df['Price']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
print("Training data size:", len(x_train))
print("Testing data size:", len(x_test))

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Predicted prices:", y_pred)

model.accuracy = model.score(x_test, y_test)
print("Model accuracy:", model.accuracy)
