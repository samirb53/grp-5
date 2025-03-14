import pandas as pd
import numpy as np
import logging
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_model():
    """
    Loads train/test CSVs, trains an XGBoost classifier for next-day (daily) price movement,
    does hyperparameter tuning, evaluates, and saves the model + label encoders.
    """
    logging.info("Loading preprocessed training and testing datasets...")
    train_df = pd.read_csv("cleaned_stock_data_train.csv", low_memory=False)
    test_df  = pd.read_csv("cleaned_stock_data_test.csv", low_memory=False)

    # Basic sanity
    required_cols = {"date", "ticker", "close"}
    if not required_cols.issubset(train_df.columns) or not required_cols.issubset(test_df.columns):
        raise KeyError(f"Missing required columns {required_cols} in train/test data.")

    # -------------------------------------------------------------
    # 1) SHIFT FOR NEXT-DAY PRICE MOVEMENT (>=1%)
    # -------------------------------------------------------------
    train_df["price_movement"] = ((train_df["close"].shift(-1) - train_df["close"]) 
                                  / train_df["close"] >= 0.01).astype(int)
    test_df["price_movement"]  = ((test_df["close"].shift(-1) - test_df["close"]) 
                                  / test_df["close"] >= 0.01).astype(int)

    # Drop NaN rows (from shift)
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # -------------------------------------------------------------
    # 2) FEATURE SELECTION
    # -------------------------------------------------------------
    remove_cols = ["ticker", "date", "company_name", "isin", "market", 
                   "main_currency", "price_movement"]
    features = [c for c in train_df.columns if c not in remove_cols]
    logging.info(f"Using features: {features}")

    # -------------------------------------------------------------
    # 3) LABEL ENCODING
    # -------------------------------------------------------------
    label_encoders = {}
    cat_cols = train_df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if col not in remove_cols:
            le = LabelEncoder()
            train_df[col] = train_df[col].astype(str)
            test_df[col]  = test_df[col].astype(str)
            le.fit(train_df[col])
            train_df[col] = le.transform(train_df[col])
            # Map test set 
            mapping = {cat: idx for idx, cat in enumerate(le.classes_)}
            test_df[col] = test_df[col].apply(lambda x: mapping[x] if x in mapping else -1)
            label_encoders[col] = le

    X_train = train_df[features]
    y_train = train_df["price_movement"]
    X_test  = test_df[features]
    y_test  = test_df["price_movement"]

    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # -------------------------------------------------------------
    # 4) TRAIN XGBoost with RandomizedSearchCV
    # -------------------------------------------------------------
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_jobs=-1, seed=42)

    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    from sklearn.model_selection import RandomizedSearchCV
    rand_search = RandomizedSearchCV(
        xgb_clf,
        param_distributions=param_dist,
        n_iter=10,
        scoring='accuracy',
        cv=3,
        verbose=1,
        random_state=42
    )
    rand_search.fit(X_train, y_train)

    best_model = rand_search.best_estimator_
    logging.info(f"Best hyperparams: {rand_search.best_params_}")

    # -------------------------------------------------------------
    # 5) EVALUATE
    # -------------------------------------------------------------
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"XGBoost Accuracy: {acc:.4f}")

    # -------------------------------------------------------------
    # 6) SAVE MODEL
    # -------------------------------------------------------------
    joblib.dump((best_model, features, label_encoders), "best_trading_model.pkl")
    logging.info("Model saved as best_trading_model.pkl")

def predict_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict signals (BUY, SELL, HOLD) using saved XGBoost model.
    Next-day classification logic.
    """
    import logging
    import pandas as pd
    import numpy as np
    import joblib

    if data.empty:
        logging.warning("No data provided.")
        return pd.DataFrame({"Error": ["No data available"]})

    try:
        model, feature_list, lbl_encoders = joblib.load("best_trading_model.pkl")
    except FileNotFoundError:
        logging.error("Model file not found!")
        return pd.DataFrame({"Error": ["Model file not found"]})

    # Transform
    df = transform_api_data(data)

    # Check features
    missing_feats = [f for f in feature_list if f not in df.columns]
    if missing_feats:
        msg = f"Missing features: {missing_feats}"
        logging.error(msg)
        raise KeyError(msg)

    # Label encoding 
    for col, le in lbl_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    X_live = df[feature_list].apply(pd.to_numeric, errors='coerce').astype(np.float32)
    # Probabilities
    probs = model.predict_proba(X_live)[:, 1]
    preds = model.predict(X_live)

    # Threshold logic 
    threshold_buy = 0.52
    threshold_sell = 0.48
    actions = []

    for p in probs:
        if p > threshold_buy:
            actions.append("BUY")
        elif p < threshold_sell:
            actions.append("SELL")
        else:
            actions.append("HOLD")

    out_df = df[["date", "ticker", "close"]].copy()
    out_df["Predicted Signal"] = preds
    out_df["Buy Probability"] = probs
    out_df["Action"] = actions

    # Now define quantities AFTER out_df exists
    out_df["Quantity"] = [1] * len(out_df)  # Default to 1 (or another logical value)

    return out_df



def transform_api_data(api_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw data to match training format.
    """
    df = api_df.copy()
    if df.empty:
        return df

    rename_dict = {
        "Date": "date",
        "Dividend Paid": "dividend",
        "Common Shares Outstanding": "shares_outstanding",
        "Adjusted Closing Price": "adj._close",
        "Highest Price": "high",
        "Lowest Price": "low",
        "Opening Price": "open",
        "Trading Volume": "volume"
    }
    df.rename(columns=rename_dict, inplace=True)

    # Ensure close
    if "Last Closing Price" in df.columns:
        df["close"] = df["Last Closing Price"]
    elif "adj._close" in df.columns:
        df["close"] = df["adj._close"]

    numeric_cols = ["open","high","low","close","adj._close","volume","dividend","shares_outstanding"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.sort_values("date", inplace=True)

    # Default columns
    default_vals = {
        "simfinid_x": 0,
        "simfinid_y": 0,
        "industryid": "Unknown Industry",
        "end_of_financial_year_(month)": 12
    }
    for c, d in default_vals.items():
        if c not in df.columns:
            df[c] = d

    # Rolling features 
    if "adj._close" in df.columns:
        df["ma_5"] = df["adj._close"].rolling(window=5, min_periods=1).mean()
        df["ma_20"] = df["adj._close"].rolling(window=20, min_periods=1).mean()
        df["volatility_10"] = df["adj._close"].rolling(window=10, min_periods=1).std()
    else:
        df["ma_5"] = np.nan
        df["ma_20"] = np.nan
        df["volatility_10"] = np.nan

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

if __name__ == "__main__":
    train_model()
