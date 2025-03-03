import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset from Kaggle
df = pd.read_csv("train.csv")

# Select relevant features
X = df[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = df["SalePrice"]

# Handle missing values
X = X.fillna(X.median())

# User input for house features
sqft = float(input("Enter square footage: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
user_input = pd.DataFrame([[sqft, bedrooms, bathrooms]], columns=X.columns)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
user_price_pred = model.predict(user_input)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print(f"Predicted price for input house: ${user_price_pred[0]:,.2f}")

# Visualizing actual vs predicted prices
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")

# Distribution of residuals
plt.subplot(1, 2, 2)
sns.histplot(y_test - y_pred, bins=20, kde=True)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")

plt.tight_layout()
plt.show()

# Feature importance visualization
plt.figure(figsize=(8, 5))
sns.barplot(x=X.columns, y=model.coef_)
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Feature Importance in House Price Prediction")
plt.show()
