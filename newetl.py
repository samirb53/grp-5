import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

# ✅ Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_data():
    try:
        logging.info("📂 Loading datasets...")

        # ✅ Load CSVs
        shareprices_df = pd.read_csv("us-shareprices-daily.csv", sep=";", parse_dates=["Date"], dayfirst=False)
        companies_df = pd.read_csv("us-companies.csv", sep=";")
        income_df = pd.read_csv("us-income-quarterly.csv", sep=";")

        # ✅ Standardize column names
        shareprices_df.columns = shareprices_df.columns.str.lower().str.replace(" ", "_")
        companies_df.columns = companies_df.columns.str.lower().str.replace(" ", "_")
        income_df.columns = income_df.columns.str.lower().str.replace(" ", "_")

        logging.info("✅ Data loaded successfully!")

        # ✅ Ensure required columns exist
        required_columns = {"ticker", "date", "close", "adj._close", "volume"}
        missing_columns = required_columns - set(shareprices_df.columns)
        if missing_columns:
            raise KeyError(f"❌ Missing columns in share prices dataset: {missing_columns}")

        # ✅ Convert Date Column to Datetime Format
        shareprices_df["date"] = pd.to_datetime(shareprices_df["date"], errors="coerce")

        # ✅ Convert numeric columns
        numeric_cols = ["open", "high", "low", "close", "adj._close", "volume"]
        for col in numeric_cols:
            if col in shareprices_df.columns:
                shareprices_df[col] = pd.to_numeric(shareprices_df[col], errors="coerce")

        # ✅ Handle Missing Values
        shareprices_df.dropna(subset=["ticker", "close", "date"], inplace=True)
        shareprices_df["dividend"] = shareprices_df["dividend"].fillna(0)

        # ✅ Fix missing "shares_outstanding"
        if "shares_outstanding" in shareprices_df.columns:
            shareprices_df["shares_outstanding"] = shareprices_df["shares_outstanding"].fillna(
                shareprices_df["shares_outstanding"].median()
            )

        # ✅ Handle missing values in companies dataset
        companies_df["industryid"] = companies_df["industryid"].fillna("Unknown Industry")
        companies_df["isin"] = companies_df["isin"].fillna("Unknown")

        # ✅ Drop unnecessary columns
        companies_df.drop(columns=["business_summary", "number_employees", "cik"], inplace=True, errors="ignore")

        logging.info("✅ Missing values handled successfully!")

        # ✅ Find Top 10 Most Traded Stocks
        top_tickers = shareprices_df.groupby("ticker")["volume"].sum().nlargest(10).index
        shareprices_df = shareprices_df[shareprices_df["ticker"].isin(top_tickers)]

        # ✅ Merge Data with Company Info
        merged_df = shareprices_df.merge(companies_df, on="ticker", how="left")

        # ✅ Ensure `date` is still present after merging
        if "date" not in merged_df.columns:
            raise KeyError("'date' column is missing after merging!")

        logging.info(f"✅ Data merged successfully! New dataset shape: {merged_df.shape}")

        # ✅ Feature Engineering
        logging.info("⚙️ Applying feature engineering...")

        # ✅ Fix `fillna(method=...)` Future Warning
        merged_df["adj._close"] = merged_df["adj._close"].ffill().bfill()

        # ✅ Apply moving averages and volatility
        merged_df["ma_5"] = merged_df["adj._close"].rolling(window=5, min_periods=1).mean()
        merged_df["ma_20"] = merged_df["adj._close"].rolling(window=20, min_periods=1).mean()
        merged_df["volatility_10"] = merged_df["adj._close"].rolling(window=10, min_periods=1).std()

        # ✅ Fix Numeric Column Issues (Only Apply Median to Numeric Columns)
        numeric_features = ["ma_5", "ma_20", "volatility_10"]
        merged_df[numeric_features] = merged_df[numeric_features].ffill().fillna(merged_df[numeric_features].median())

        logging.info("✅ Feature engineering completed!")

        # ✅ Fix Target Variable (Price Movement)
        merged_df["price_movement"] = (merged_df["close"].shift(-1) > merged_df["close"]).astype(int)

        logging.info("✅ Target variable created!")

        # ✅ Reorder Columns & Drop NaNs
        column_order = ["date"] + [col for col in merged_df.columns if col != "date"]
        merged_df = merged_df[column_order].dropna()

        # ✅ Split into Train and Test Sets
        train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42, shuffle=False)

        # ✅ Save Train and Test Data
        train_df.to_csv("cleaned_stock_data_train.csv", index=False)
        test_df.to_csv("cleaned_stock_data_test.csv", index=False)

        logging.info("✅ Cleaned train and test datasets saved successfully!")

        return merged_df

    except Exception as e:
        logging.error(f"❌ Error in processing data: {e}")
        return None

if __name__ == "__main__":
    process_data()
