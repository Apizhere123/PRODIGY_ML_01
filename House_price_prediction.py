import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ── Step 1: Load the dataset ──────────────────────────
df = pd.read_csv("train.csv")

# ── Step 2: Select the features we care about ─────────
# GrLivArea   = above-ground living area (square footage)
# BedroomAbvGr = number of bedrooms
# FullBath    = number of full bathrooms
# SalePrice   = the target we want to predict
features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
target = "SalePrice"

data = df[features + [target]]

X = data[features]
y = data[target]

# ── Step 3: Split into training and testing sets ──────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# ── Step 4: Train the linear regression model ─────────
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained!")

# ── Step 5: Evaluate on the test set ───────────────────
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² score: {r2:.3f}")

# ── Step 6: Visualize predictions vs actual prices ─────
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="royalblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title(f"House Price Prediction (R² = {r2:.3f})")
plt.tight_layout()
plt.savefig("prediction_plot.png")
plt.show()

# ── Step 7: Predict the price of a brand new house ─────
new_house = pd.DataFrame({
    "GrLivArea": [1800],
    "BedroomAbvGr": [3],
    "FullBath": [2]
})
predicted_price = model.predict(new_house)
print(f"Predicted price for new house (1800 sqft, 3 bed, 2 bath): ${predicted_price[0]:,.2f}")