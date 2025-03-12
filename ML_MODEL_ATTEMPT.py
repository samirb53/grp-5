import pandas as pd
import numpy as np
import logging
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# -------------------------------------------------------------
# ✅ SETUP LOGGING
# -------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------------------------------------
# ✅ LOAD DATA
# -------------------------------------------------------------
logging.info("📂 Loading preprocessed training and testing datasets...")
train_df = pd.read_csv("cleaned_stock_data_train.csv", low_memory=False)
test_df = pd.read_csv("cleaned_stock_data_test.csv", low_memory=False)

# Debug: Print columns at the start
print("🔍 [START] Train Columns:", train_df.columns.tolist())
print("🔍 [START] Test Columns:", test_df.columns.tolist())

# Ensure "close" is present
if "close" not in train_df.columns or "close" not in test_df.columns:
    logging.error("❌ 'close' column is missing from train or test DataFrame!")
    raise KeyError("'close' column is missing!")

# -------------------------------------------------------------
# ✅ PRESERVE DATE, TICKER, CLOSE (FOR LATER MERGE)
# -------------------------------------------------------------
train_dates = train_df[["date", "ticker", "close"]].copy()
test_dates  = test_df[["date", "ticker", "close"]].copy()

# -------------------------------------------------------------
# ✅ DEFINE TARGET & FEATURES
# -------------------------------------------------------------
target = "price_movement"
remove_cols = ["ticker", "date", "company_name", "isin", "market", "main_currency"]

# Make sure "close" is NOT removed
if "close" in remove_cols:
    remove_cols.remove("close")

features = [c for c in train_df.columns if c not in remove_cols + [target]]

logging.info(f"🔎 Feature selection completed! Using {len(features)} features.")
print("🔍 Feature List:", features)

# -------------------------------------------------------------
# ✅ CONVERT CATEGORICAL -> NUMERIC (LABEL ENCODING)
# -------------------------------------------------------------
logging.info("🔄 Converting categorical columns to numeric format...")
categorical_cols = train_df.select_dtypes(include=["object"]).columns

label_encoders = {}
for col in categorical_cols:
    if col != "close":   # Don't encode "close" (it's numeric)
        le = LabelEncoder()
        train_df[col] = train_df[col].astype(str)
        le.fit(train_df[col])

        # Safely transform train & test
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = test_df[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        label_encoders[col] = le

logging.info("✅ Categorical encoding completed!")

# -------------------------------------------------------------
# ✅ SHIFT TARGET FOR NEXT-DAY PREDICTION
# -------------------------------------------------------------
logging.info("🔄 Adjusting target variable for next-day predictions...")
train_df[target] = train_df[target].shift(-1)
test_df[target]  = test_df[target].shift(-1)

# Drop NaNs so features & target align
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# -------------------------------------------------------------
# ✅ FILTER OUT ROWS WITH CLOSE = 0 (TO AVOID OVERFITTING)
# -------------------------------------------------------------
train_df = train_df[train_df["close"] > 0]
test_df  = test_df[test_df["close"] > 0]

print("🔍 [BEFORE MERGE] Train Columns:", train_df.columns.tolist())
print("🔍 [BEFORE MERGE] Test Columns:",  test_df.columns.tolist())

# -------------------------------------------------------------
# ✅ REATTACH date, ticker, close (This causes close_x, close_y)
# -------------------------------------------------------------
train_df = train_df.merge(train_dates, left_index=True, right_index=True, how="left", suffixes=("", "_DROPPED"))
test_df  = test_df.merge(test_dates,   left_index=True, right_index=True, how="left", suffixes=("", "_DROPPED"))

print("🔍 [AFTER MERGE] Train Columns:", train_df.columns.tolist())
print("🔍 [AFTER MERGE] Test Columns:",  test_df.columns.tolist())

# -------------------------------------------------------------
# ✅ RENAME 'close_x' => 'close', 'date_x' => 'date', 'ticker_x' => 'ticker'
#    & drop the duplicated columns if they appear (like 'close_y')
# -------------------------------------------------------------
# For train_df
if "close_x" in train_df.columns:
    train_df.drop(columns=["close_y"], errors="ignore", inplace=True)
    train_df.rename(columns={
        "close_x": "close"
    }, inplace=True)

if "ticker_x" in train_df.columns:
    train_df.drop(columns=["ticker_y"], errors="ignore", inplace=True)
    train_df.rename(columns={
        "ticker_x": "ticker"
    }, inplace=True)

if "date_x" in train_df.columns:
    train_df.drop(columns=["date_y"], errors="ignore", inplace=True)
    train_df.rename(columns={
        "date_x": "date"
    }, inplace=True)

# For test_df
if "close_x" in test_df.columns:
    test_df.drop(columns=["close_y"], errors="ignore", inplace=True)
    test_df.rename(columns={
        "close_x": "close"
    }, inplace=True)

if "ticker_x" in test_df.columns:
    test_df.drop(columns=["ticker_y"], errors="ignore", inplace=True)
    test_df.rename(columns={
        "ticker_x": "ticker"
    }, inplace=True)

if "date_x" in test_df.columns:
    test_df.drop(columns=["date_y"], errors="ignore", inplace=True)
    test_df.rename(columns={
        "date_x": "date"
    }, inplace=True)

print("🔍 [FINAL] Train Columns:", train_df.columns.tolist())
print("🔍 [FINAL] Test  Columns:", test_df.columns.tolist())

# -------------------------------------------------------------
# ✅ PREPARE FINAL X, Y
# -------------------------------------------------------------
X_train = train_df[features]
y_train = train_df[target]
X_test  = test_df[features]
y_test  = test_df[target]

logging.info("🚀 Training XGBoost model...")

# -------------------------------------------------------------
# ✅ TRAIN THE MODEL
# -------------------------------------------------------------
xgb_model = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=0.7,
    objective='binary:logistic',
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

# -------------------------------------------------------------
# ✅ EVALUATE
# -------------------------------------------------------------
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"📉 XGBoost Accuracy: {accuracy:.4f}")

# -------------------------------------------------------------
# ✅ SAVE MODEL
# -------------------------------------------------------------
joblib.dump((xgb_model, features, label_encoders), "best_trading_model.pkl")
logging.info("✅ Model saved successfully!")

# -------------------------------------------------------------
# ✅ PREDICTION FUNCTION
# -------------------------------------------------------------
def predict_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict trading signals using the trained XGBoost model,
    and add a 'Buy'/'Sell' action column.
    """
    import logging
    import joblib
    import numpy as np
    import pandas as pd

    if data.empty:
        logging.warning("⚠️ No data provided for prediction.")
        return pd.DataFrame({"Error": ["No data available for prediction."]})

    # ✅ Load the model
    try:
        model, feature_list, lbl_encoders = joblib.load("best_trading_model.pkl")
    except FileNotFoundError:
        logging.error("❌ Model file not found! Make sure you have trained the model first.")
        return pd.DataFrame({"Error": ["Model file not found!"]})

    # ✅ Preserve essential columns before feature transformation
    prediction_data = data[["date", "ticker", "close"]].copy()

    # ✅ Check for missing features
    missing_features = [f for f in feature_list if f not in data.columns]
    if missing_features:
        logging.error(f"❌ Missing features in provided data: {missing_features}")
        return pd.DataFrame({"Error": [f"Missing features: {missing_features}"]})

    # ✅ Encode categorical columns, except 'close'
    for col, le in lbl_encoders.items():
        if col in data.columns:
            data.loc[:, col] = data[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    # ✅ Ensure correct numeric types
    data = data[feature_list].apply(pd.to_numeric, errors='coerce').astype(np.float32)

    # ✅ Predict
    predictions = model.predict(data)

    # ✅ Scale predictions for visualization
    volatility = data["close"].rolling(window=10, min_periods=1).std()
    # Use `.loc` to avoid SettingWithCopyWarning
    prediction_data.loc[:, "Scaled Signal"] = predictions * volatility.mean() * 2

    # ✅ Add predicted signal
    prediction_data.loc[:, "Predicted Signal"] = predictions

    # ✅ ADD the "Action" column (Buy if 1, Sell if 0)
    prediction_data.loc[:, "Action"] = prediction_data["Predicted Signal"].map({1: "Buy", 0: "Sell"})

    return prediction_data
