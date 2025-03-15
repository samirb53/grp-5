import pandas as pd
import numpy as np
import logging
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --------------------------------------------------
# 1) JSON COLUMN MAP (from your mapping file)
# --------------------------------------------------
MAPPING_DICT = {
    # Basic price columns
    "Date": "date",
    "Dividend": "dividend",
    "Shares Outstanding": "shares_outstanding",
    "Close": "close",
    "Adj. Close": "adj._close",
    "High": "high",
    "Low": "low",
    "Open": "open",
    "Volume": "volume",
    # Example for statements
    "Fiscal Period": "Fiscal Period",
    "Fiscal Year": "Fiscal Year",
    "Report Date": "Report Date",
    "Publish Date": "Publish Date",
    "Restated Date": "Restated",
    # Additional columns from CSV mapping
    "SimFinId": "simfinid",
    "Company Name": "company_name",
    "Ticker": "ticker",
    "IndustryId": "industryid",
    "End of financial year (month)": "end_of_financial_year_(month)",
    "Market": "market",
    "ISIN": "isin",
}


# --------------------------------------------------
# 2) PLACEHOLDER DEFAULTS
# --------------------------------------------------
PLACEHOLDER_DEFAULTS = {
    "simfinid": 0,
    "adj._close": 0.0,
    "industryid": "Unknown Industry",
    "end_of_financial_year_(month)": 12,
    # add more if needed
}

def rename_and_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns from API data to match training data,
    then adds placeholder columns for any required columns missing.
    """
    rename_dict = {
        "Date": "date",
        "Last Closing Price": "close",
        "Highest Price": "high",
        "Lowest Price": "low",
        "Opening Price": "open",
        "Trading Volume": "volume",
        "Dividend Paid": "dividend",
        "Common Shares Outstanding": "shares_outstanding",
        "Adjusted Closing Price": "adj._close"  # if available
    }
    df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})
    
    placeholder_defaults = {
        "simfinid_x": 0,
        "simfinid_y": 0,
        "company_name": "Unknown",
        "isin": "Unknown",
        "market": "Unknown",
        "main_currency": "Unknown",
        "industryid": "Unknown Industry",
        "end_of_financial_year_(month)": 12
    }
    for col, default_val in placeholder_defaults.items():
        if col not in df.columns:
            df[col] = default_val
    return df


def transform_api_data(api_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw API data:
      - Renames columns and adds placeholders.
      - Converts the date column.
      - Ensures key numeric columns are numeric.
      - Computes rolling features on a normalized close price.
      - Computes a new feature: daily_return.
      - Replaces infinite values with NaN and fills them with 0.
    """
    df = api_df.copy()
    if df.empty:
        return df

    # 1) Rename columns and fill missing columns
    df = rename_and_fill(df)

    # 2) Convert 'date' to datetime and sort
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.sort_values("date", inplace=True)

    # 3) Ensure numeric conversion for key columns
    numeric_cols = ["open", "high", "low", "close", "adj._close", "volume", "dividend", "shares_outstanding"]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan  # Create placeholder if missing
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4) Normalize close price relative to the first day
    if "close" in df.columns and df["close"].iloc[0] != 0:
        df["norm_close"] = df["close"] / df["close"].iloc[0]
    else:
        df["norm_close"] = df["close"]

    # 4.5) Compute daily return as a new feature
    if "close" in df.columns:
        df["daily_return"] = df["close"].pct_change().fillna(0)
    else:
        df["daily_return"] = 0

    # 5) Compute rolling features on the normalized close price
    if "norm_close" in df.columns and df["norm_close"].notna().any():
        df["ma_5"] = df["norm_close"].rolling(window=5, min_periods=1).mean()
        df["ma_20"] = df["norm_close"].rolling(window=20, min_periods=1).mean()
        df["volatility_10"] = df["norm_close"].rolling(window=10, min_periods=1).std()
    else:
        df["ma_5"] = np.nan
        df["ma_20"] = np.nan
        df["volatility_10"] = np.nan

    # 6) Replace inf values with NaN and then fill NaN with 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df

def train_model():
    """
    Loads train/test CSVs, transforms them, trains an XGBoost classifier for next-day price movement,
    and saves the model along with features and label encoders.
    """
    logging.info("Loading preprocessed training and testing datasets...")
    train_df = pd.read_csv("cleaned_stock_data_train.csv", low_memory=False)
    test_df = pd.read_csv("cleaned_stock_data_test.csv", low_memory=False)

    # Rename and fill missing columns
    train_df = rename_and_fill(train_df)
    test_df = rename_and_fill(test_df)

    # Compute daily_return for training data
    if "close" in train_df.columns:
        train_df["daily_return"] = train_df["close"].pct_change().fillna(0)
    if "close" in test_df.columns:
        test_df["daily_return"] = test_df["close"].pct_change().fillna(0)

    # Basic check for required columns
    required_cols = {"date", "ticker", "close"}
    if not required_cols.issubset(train_df.columns) or not required_cols.issubset(test_df.columns):
        raise KeyError(f"Missing required columns {required_cols} in train/test data.")

    # Shift target for next-day price movement (>= 1%)
    train_df["price_movement"] = (
        (train_df["close"].shift(-1) - train_df["close"]) / train_df["close"] >= 0.01
    ).astype(int)
    test_df["price_movement"] = (
        (test_df["close"].shift(-1) - test_df["close"]) / test_df["close"] >= 0.01
    ).astype(int)

    train_df.dropna(subset=["price_movement"], inplace=True)
    test_df.dropna(subset=["price_movement"], inplace=True)

    # FEATURE SELECTION: remove only 'date', 'ticker', and target column
    target_col = "price_movement"
    remove_cols = ["date", "ticker", target_col]
    features = [c for c in train_df.columns if c not in remove_cols]
    logging.info(f"Using features: {features}")

    # LABEL ENCODING for categorical features
    label_encoders = {}
    for col in features:
        if train_df[col].dtype == "object":
            le = LabelEncoder()
            train_df[col] = train_df[col].astype(str)
            test_df[col] = test_df[col].astype(str)

            le.fit(train_df[col])
            train_df[col] = le.transform(train_df[col])
            test_df[col] = test_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            label_encoders[col] = le

    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]

    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Debugging: Check for invalid values
    logging.info("Checking dataset statistics before training...")
    print(X_train.describe())  # Check for extreme values
    print(y_train.value_counts())  # Ensure classes are balanced

    # Handle Inf and NaN values
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    y_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    y_train.fillna(0, inplace=True)
    y_test.fillna(0, inplace=True)

    # Convert data to float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Compute scale_pos_weight to handle class imbalance
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1
    logging.info(f"Scale_pos_weight: {scale_pos_weight}")

    # HYPERPARAMETER SEARCH + TRAIN
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=-1,
        seed=42,
        scale_pos_weight=scale_pos_weight
    )

    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    rand_search = RandomizedSearchCV(
        xgb_clf,
        param_distributions=param_dist,
        n_iter=10,
        scoring='accuracy',
        cv=3,
        verbose=1,
        random_state=42
    )
    
    logging.info("Starting XGBoost training...")
    rand_search.fit(X_train, y_train)

    best_model = rand_search.best_estimator_
    logging.info(f"Best hyperparams: {rand_search.best_params_}")

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"XGBoost Accuracy: {acc:.4f}")

    joblib.dump((best_model, features, label_encoders), "best_trading_model.pkl")
    logging.info("Model saved as best_trading_model.pkl")


def predict_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts signals (BUY, SELL, HOLD) using the saved XGBoost model.
    Steps:
      1) Transform data using transform_api_data()
      2) Check for missing features
      3) Label encode features
      4) Predict probabilities and classes
      5) Apply thresholds: if probability > 0.52 then BUY, if < 0.48 then SELL, else HOLD.
    """
    import logging
    if data.empty:
        logging.warning("No data provided.")
        return pd.DataFrame({"Error": ["No data available"]})

    try:
        model, feature_list, lbl_encoders = joblib.load("best_trading_model.pkl")
    except FileNotFoundError:
        logging.error("Model file not found!")
        return pd.DataFrame({"Error": ["Model file not found"]})

    df = transform_api_data(data)

    missing_feats = [f for f in feature_list if f not in df.columns]
    if missing_feats:
        msg = f"Missing features: {missing_feats}"
        logging.error(msg)
        raise KeyError(msg)

    import numpy as np
    for col in feature_list:
        if col in lbl_encoders:
            encoder = lbl_encoders[col]
            df[col] = df[col].astype(str).apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

    X_live = df[feature_list].apply(pd.to_numeric, errors='coerce').astype(np.float32)

    probs = model.predict_proba(X_live)[:, 1]
    preds = model.predict(X_live)

    threshold_buy = 0.52
    threshold_sell = 0.48
    actions = []
    quantities = []
    for p in probs:
        if p > threshold_buy:
            actions.append("BUY")
            quantities.append(10)
        elif p < threshold_sell:
            actions.append("SELL")
            quantities.append(-10)
        else:
            actions.append("HOLD")
            quantities.append(0)

    out_df = df[["date", "ticker", "close"]].copy()
    out_df["Predicted Signal"] = preds
    out_df["Buy Probability"] = probs
    out_df["Action"] = actions
    out_df["Quantity"] = quantities

    return out_df


if __name__ == "__main__":
    train_model()
