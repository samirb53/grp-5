import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date
from pysimfin import PySimFin
from ML_MODEL import predict_signals, transform_api_data

#####################################
# Streamlit UI
#####################################
st.title("Backtesting Simulator (Debug Version)")
st.markdown("""
This backtesting tool:
1. Fetches **historical price data** from PySimFin for your chosen ticker and date range.
2. Uses the **ML model** to generate signals (BUY, SELL, HOLD).
3. **Simulates trades** day by day, logging shares, cash, and portfolio value.
4. Displays a **debug table** and a **chart** of the daily portfolio value.
""")

API_KEY = "e1c75cc5-3bca-4b0c-b847-6447bd4ed901"
simfin = PySimFin(API_KEY)

# User Inputs
ticker = st.selectbox(
    "Select Company for Backtesting",
    ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'META', 'TSLA', 'NUVB', 'V', 'JNJ', 'WMT']
)
start_date = st.date_input("Start Date", date(2022, 1, 1))
end_date = st.date_input("End Date", date.today())
initial_capital = st.number_input("Initial Capital ($)", value=10000)

run_button = st.button("Run Backtesting")

#####################################
# When the user clicks "Run Backtesting"
#####################################
if run_button:
    st.write(f"**Backtesting {ticker}** from {start_date} to {end_date}")
    st.write(f"Initial Capital: **${initial_capital:,.2f}**")

    # --------------------------------
    # 1) Get Historical Data from PySimFin
    # --------------------------------
    try:
        raw_data = simfin.get_share_prices(
            ticker=ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d")
        )
    except Exception as e:
        st.error(f"Error fetching data from PySimFin: {e}")
        st.stop()

    if raw_data.empty:
        st.error("No data returned from API for the selected ticker and date range.")
        st.stop()

    st.write(f"Fetched {len(raw_data)} rows of raw data from PySimFin.")

    st.write("Raw Data Columns:", raw_data.columns.tolist())
    st.write("Raw Data Sample:", raw_data.head(10))

    # --------------------------------
    # 2) Transform Full Dataset
    # --------------------------------
    df = transform_api_data(raw_data)
    if df.empty:
        st.error("Transformed DataFrame is empty. Possibly no valid price data.")
        st.stop()

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Debug: Print transformed data columns and a sample
    st.write("Transformed Data Columns:", df.columns.tolist())
    st.write("Transformed Data Sample:", df.head(10))

    # Basic sanity check
    if "close" not in df.columns:
        st.error("No 'close' price found after transformation. Cannot proceed.")
        st.stop()

    st.write(f"Transformed data has {len(df)} rows after sorting/dropping duplicates.")
    st.dataframe(df.head(5))  # Show a small sample for debugging

    # --------------------------------
    # 3) Load the Trained Model
    # --------------------------------
    try:
        model, feature_list, lbl_encoders = joblib.load("best_trading_model.pkl")
    except Exception as e:
        st.error(f"Failed to load ML model (best_trading_model.pkl): {e}")
        st.stop()

    # --------------------------------
    # 4) Generate Signals for ALL Days
    # --------------------------------
    signals_df = predict_signals(df)
    if signals_df.empty:
        st.error("No signals returned. Possibly the model or data is incompatible.")
        st.stop()

    st.write("Signals DataFrame Columns:", signals_df.columns.tolist())
    st.write("Signals Sample:", signals_df.head(10))

    

    # Debug: Check columns in signals_df before merging
    st.write("Signals DataFrame Columns:", signals_df.columns)
    st.dataframe(signals_df.head())



    # Debugging: Print shapes before merging
    st.write(f"Shape of signals_df: {signals_df.shape}")
    st.write(f"Shape of df before merge: {df.shape}")

    # Merge signals onto the main DF by [date, ticker, close]
    combined_df = pd.merge(
        df,
        signals_df[["date", "ticker", "Action", "Quantity", "Buy Probability"]],
        on=["date", "ticker"],
        how="left"
    )

    # Fill any missing actions with HOLD
    combined_df["Action"].fillna("HOLD", inplace=True)
    combined_df["Quantity"].fillna(0, inplace=True)

    st.subheader("Signal Distribution")
    st.write(combined_df["Action"].value_counts().to_dict())

    st.subheader("Combined Data + Signals (Sample)")
    st.dataframe(combined_df.head(5))

    # --------------------------------
    # 5) Simulate Trades (Day-by-Day)
    # --------------------------------
    cash = initial_capital
    shares = 0

    portfolio_values = []
    dates = []
    trade_log = []  # for debugging each day

    for idx, row in combined_df.iterrows():
        current_date = row["date"]
        close_price = row["close"]
        action = row["Action"]

        # Skip if close_price is invalid
        if pd.isna(close_price) or close_price <= 0:
            portfolio_value = cash + shares * 0
            portfolio_values.append(portfolio_value)
            dates.append(current_date)
            trade_log.append({
                "date": current_date,
                "action": action,
                "close_price": close_price,
                "shares_held": shares,
                "cash_remaining": cash,
                "portfolio_value": portfolio_value,
                "note": "Skipping because close_price <= 0"
            })
            continue

        # Accumulate shares if action=BUY
        if action == "BUY":
            # Let's invest 10% of current cash
            invest_amount = 0.10 * cash
            shares_to_buy = int(invest_amount // close_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * close_price
                cash -= cost
                shares += shares_to_buy

        # Liquidate all if action=SELL
        elif action == "SELL":
            if shares > 0:
                cash += shares * close_price
                shares = 0

        # If HOLD, do nothing

        # Calculate daily portfolio value
        portfolio_value = cash + shares * close_price
        portfolio_values.append(portfolio_value)
        dates.append(current_date)

        # Log the day
        trade_log.append({
            "date": current_date,
            "action": action,
            "close_price": close_price,
            "shares_held": shares,
            "cash_remaining": cash,
            "portfolio_value": portfolio_value,
            "note": ""
        })

    # --------------------------------
    # 6) Debug Table of Each Day
    # --------------------------------
    debug_df = pd.DataFrame(trade_log)
    st.subheader("Daily Trade & Portfolio Debug Log")
    st.dataframe(debug_df.head(30))  # show first 30 days for brevity

    # --------------------------------
    # 7) Plot the Portfolio Value
    # --------------------------------
    portfolio_df = pd.DataFrame({"date": dates, "portfolio_value": portfolio_values})
    portfolio_df.set_index("date", inplace=True)

    st.subheader("Backtesting Results")
    if len(portfolio_values) > 0:
        final_val = portfolio_values[-1]
        st.write(f"Final Portfolio Value: **${final_val:,.2f}**")
        st.line_chart(portfolio_df)
    else:
        st.write("No portfolio values computed.")
