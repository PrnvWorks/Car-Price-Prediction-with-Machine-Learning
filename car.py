import pandas as pd
pd.set_option('display.max_columns', None)

fname = "task2\car data.csv"   # replace with your filename
df = pd.read_csv(fname)

print("=== DF HEAD ===")
print(df.head().to_string(index=False))
print("\n=== COLUMNS ===")
print(list(df.columns))
print("\n=== INFO ===")
print(df.info())


#=========step2===========#

# step2_summary_and_featureprep.py
import pandas as pd
pd.set_option('display.max_columns', None)

# use forward-slash path to avoid escape warnings
fname = "task2/car data.csv"   # adjust only if your CSV is elsewhere

df = pd.read_csv(fname)

print("=== SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")
print(list(df.columns))

# likely price column
candidates = [c for c in df.columns if any(k in c.lower() for k in ['selling','sell_price','price','resale','target'])]
print("\nLikely price/target columns:", candidates)

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== DTYPE SUMMARY ===")
print(df.dtypes)

print("\n=== NUMERIC SUMMARY ===")
print(df.describe().transpose())

# show unique examples for object columns
print("\n=== SAMPLE UNIQUE VALUES (object columns) ===")
for c in df.select_dtypes(include=['object']).columns:
    print(f"\n-- {c} (unique sample up to 12) --")
    print(df[c].dropna().unique()[:12])

# Create Age feature if Year exists
if 'Year' in df.columns:
    current_year = pd.Timestamp.now().year
    df['Age'] = current_year - df['Year']
    print("\nCreated Age column from Year. Example head():")
    print(df[['Car_Name','Year','Age','Selling_Price','Present_Price']].head().to_string(index=False))

# Save a quick cleaned preview (not dropping anything yet)
df.to_csv("task2/car_data_preview.csv", index=False)
print("\nSaved preview as task2/car_data_preview.csv")


#==================step 3=====================#

# STEP 3 — Clean, Encode, Split

import pandas as pd
from sklearn.model_selection import train_test_split

# Load
df = pd.read_csv("task2/car data.csv")
df.columns = [c.strip() for c in df.columns]  # strip spaces

# Create Age feature
current_year = pd.Timestamp.now().year
df["Age"] = current_year - df["Year"]

# Drop columns not useful for ML
df = df.drop(["Car_Name", "Year"], axis=1)

# Target variable
y = df["Selling_Price"]

# Feature matrix
X = df.drop(["Selling_Price"], axis=1)

print("\n=== FEATURES BEFORE ENCODING ===")
print(X.head().to_string(index=False))

# Encode categorical features (Fuel_Type, Selling_type, Transmission)
X = pd.get_dummies(X, drop_first=True)

print("\n=== FEATURES AFTER ENCODING ===")
print(X.head().to_string(index=False))
print("\nNumber of features:", X.shape[1])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)


#====================step 4===============#

# STEP 4 — Linear Regression Model

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Train model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on test set
y_pred_lr = lr.predict(X_test)

# Evaluate
print("\n=== LINEAR REGRESSION RESULTS ===")
print("R2 Score:", r2_score(y_test, y_pred_lr))
print("MAE:", mean_absolute_error(y_test, y_pred_lr))


#=================step 5=================#
# STEP 5 — Random Forest Regressor (Better Model)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Train Random Forest
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Predict
y_pred_rf = rf.predict(X_test)

# Evaluate
print("\n=== RANDOM FOREST RESULTS ===")
print("R2 Score:", r2_score(y_test, y_pred_rf))
print("MAE:", mean_absolute_error(y_test, y_pred_rf))


# =================== Step 6 ============ #
print("\n=== MODEL COMPARISON ===")
print("Linear Regression R2 :", r2_score(y_test, y_pred_lr))
print("Random Forest   R2 :", r2_score(y_test, y_pred_rf))

print("\nLinear Regression MAE:", mean_absolute_error(y_test, y_pred_lr))
print("Random Forest   MAE:", mean_absolute_error(y_test, y_pred_rf))


# ==================== Step 7 =====================#
# STEP 7 — Predict price of a new car

import numpy as np

def predict_price(present_price, driven_kms, fuel_type, selling_type, transmission, owner, age):
    
    # One-hot encoding manually
    ft_diesel = 1 if fuel_type.lower() == "diesel" else 0
    ft_petrol = 1 if fuel_type.lower() == "petrol" else 0
    st_individual = 1 if selling_type.lower() == "individual" else 0
    trans_manual = 1 if transmission.lower() == "manual" else 0
    
    data = np.array([[present_price, driven_kms, owner, age,
                      ft_diesel, ft_petrol, st_individual, trans_manual]])
    
    pred = rf.predict(data)
    return pred[0]

# Example prediction:
predicted = predict_price(
    present_price=5.59,
    driven_kms=27000,
    fuel_type="Petrol",
    selling_type="Dealer",
    transmission="Manual",
    owner=0,
    age=11
)

print("\nPredicted Selling Price:", predicted, "lakhs")


# +++++++++++++++++ Step 8 ++++++++++++++++ #
# STEP 8 — Save model
import joblib
joblib.dump(rf, "car_price_model.pkl")
joblib.dump(X.columns, "model_features.pkl")

print("\nModel saved as car_price_model.pkl")
