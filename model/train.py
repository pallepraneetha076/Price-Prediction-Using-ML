import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# 1. LOAD DATA
# =========================
data = pd.read_csv("dataset/train.csv")

# =========================
# 2. CLEAN DATA
# =========================
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].fillna(data[col].mode()[0])
    else:
        data[col] = data[col].fillna(data[col].median())

# =========================
# 3. SPLIT FEATURES / TARGET
# =========================
y = data["SalePrice"]
X = data.drop(["SalePrice", "Id"], axis=1)

X = pd.get_dummies(X, drop_first=True)

# =========================
# 4. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. MODEL TRAINING
# =========================
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# =========================
# 6. EVALUATION (ONLY PRINT)
# =========================
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# =========================
# 7. SAVE MODEL + FEATURES
# =========================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("Model and features saved successfully")