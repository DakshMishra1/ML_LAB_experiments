import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Experiment1/housing.csv")

df = df.dropna()

df = df.drop('ocean_proximity', axis=1)

X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

predictions = model.predict(X_test)

print("Predicted Prices:")
print(predictions[:10])

comparison = pd.DataFrame({'Actual Price': y_test,'Predicted Price': predictions})

print(comparison.head())

print("MSE:", mse)
print("R2 Score:", r2)