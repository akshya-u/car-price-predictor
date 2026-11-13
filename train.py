import pandas as pd
from sklearn.model_selection import train_test_split

# Step 3: load the data
df = pd.read_csv("data/car data.csv")

# Step 4: clean
df = df.drop("Car_Name", axis=1)
df["Car_Age"] = 2026 - df["Year"]
df = df.drop("Year", axis=1)
df = df[df["Kms_Driven"] < 200000]
df = pd.get_dummies(df, columns=["Fuel_Type", "Seller_Type", "Transmission"], drop_first=True, dtype=int)

# Step 5: split
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

# Step 6:Train your first model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("R2:", r2_score(y_test, predictions))
print("MAE:", mean_absolute_error(y_test, predictions))

# Step 7: Train Random Forest and compare
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)
print("RF R2:", r2_score(y_test, rf_predictions))
print("RF MAE:", mean_absolute_error(y_test, rf_predictions))

# Step 8: Save the trained model
import joblib
joblib.dump(rf, "model.pkl")
joblib.dump(list(X.columns), "columns.pkl")
print("Saved!")

# Step 9b: Cross-validation
from sklearn.model_selection import cross_val_score,KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lr_scores = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2")
rf_scores = cross_val_score(RandomForestRegressor(random_state=42), X, y, cv=kf, scoring="r2")
print("LR R2 per fold:", lr_scores.round(2))
print("LR average:", lr_scores.mean().round(3))
print("RF R2 per fold:", rf_scores.round(2))
print("RF average:", rf_scores.mean().round(3))