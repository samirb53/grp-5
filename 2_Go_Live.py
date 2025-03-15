import streamlit as st
import pandas as pd
from datetime import date, timedelta
from pysimfin import PySimFin
from ML_MODEL import predict_signals, transform_api_data

# Define the top 10 companies (plus any others you want to include)
top_companies = [
    'AAPL', 'AMZN', 'GOOG', 'MSFT', 'META', 
    'TSLA', 'NUVB', 'V', 'JNJ', 'WMT'
]

st.set_page_config(
    page_title="Go Live: Stock Data & Signals",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("Go Live: Stock Data, Financial Statements, and Trading Signals")

# -----------------------------------------------------
# 1) Global Ticker Selection (Sidebar)
# -----------------------------------------------------
st.sidebar.header("Global Settings")
common_ticker = st.sidebar.selectbox(
    "Select a Company",
    ["Select a ticker"] + top_companies
)
if common_ticker == "Select a ticker":
    st.sidebar.warning("Please select a company to proceed.")
    st.stop()

# Initialize the API wrapper with your API key
API_KEY = "e1c75cc5-3bca-4b0c-b847-6447bd4ed901"
simfin = PySimFin(API_KEY)

# -----------------------------------------------------
# 2) PRICE DATA
# -----------------------------------------------------
st.header("1. Price Data")

st.markdown("""
This section displays the share prices for the selected company 
over a user‐defined date range. You can adjust the date range in the sidebar.
""")

price_start_date = st.sidebar.date_input(
    "Price Data Start Date",
    date.today() - timedelta(days=30),
    key="price_start"
)
price_end_date = st.sidebar.date_input(
    "Price Data End Date",
    date.today(),
    key="price_end"
)

st.write(
    f"Fetching share prices for **{common_ticker}** "
    f"from **{price_start_date.strftime('%Y-%m-%d')}** to **{price_end_date.strftime('%Y-%m-%d')}**."
)


try:
    price_df = simfin.get_share_prices(
        ticker=common_ticker,
        start=price_start_date.strftime("%Y-%m-%d"),
        end=price_end_date.strftime("%Y-%m-%d")
    )
    if price_df.empty:
        st.error("No price data found for the selected ticker and date range.")
    else:
        st.subheader(f"Share Prices for {common_ticker}")
        st.dataframe(price_df.head())
        close_col = "Last Closing Price" if "Last Closing Price" in price_df.columns else "close"
        if "Date" in price_df.columns:
            price_df["Date"] = pd.to_datetime(price_df["Date"])
            price_df.sort_values("Date", inplace=True)
            st.line_chart(price_df.set_index("Date")[close_col])
except Exception as e:
    st.error(f"Error fetching price data: {e}")

# -----------------------------------------------------
# 3) FINANCIAL STATEMENTS
# -----------------------------------------------------
st.header("2. Financial Statements")
st.markdown("""
View the most recent financial statements for the selected company. 
By default, 2024-Q4 is shown; you can modify the year and period in the sidebar.
""")

fin_year = st.sidebar.number_input(
    "Fiscal Year", value=2024, step=1, key="fin_year"
)
fin_period = st.sidebar.selectbox(
    "Fiscal Period", ["Q1", "Q2", "Q3", "Q4", "FY"],
    index=3, key="fin_period"
)
fin_stmt_options = st.sidebar.multiselect(
    "Select Statement Types",
    options=["PL", "BS", "CF", "DERIVED"],
    default=["PL", "BS", "CF", "DERIVED"],
    key="fin_stmt"
)

st.write(
    f"Fetching financial statements for **{common_ticker}** "
    f"for fiscal year **{fin_year}** ({fin_period})."
)

try:
    statements_str = ",".join(fin_stmt_options)
    fin_data = simfin.get_financial_statement(
        ticker=common_ticker,
        fyear=fin_year,
        period=fin_period,
        statements=statements_str
    )
    if fin_data:
        for stmt_type, df in fin_data.items():
            st.subheader(f"{stmt_type} Statement")
            st.dataframe(df.head(10))
    else:
        st.error("No financial statement data found for the selected parameters.")
except Exception as e:
    st.error(f"Error fetching financial statements: {e}")

# -----------------------------------------------------
# 4) TRADING SIGNALS
# -----------------------------------------------------
st.header("3. Trading Signals")
st.markdown("""
Based on the **selected strategy**, we either:
- **Strategy 1 (Buy-and-Hold)**: 
  - If price is predicted to rise, BUY 1 share.
  - If price is predicted to fall, HOLD.
- **Strategy 2 (Buy-and-Sell)**:
  - If price is predicted to rise, BUY 1 share.
  - If price is predicted to fall, SELL 1 share.
""")

strategy_choice = st.selectbox(
    "Select Strategy for Live Signals",
    ["Strategy 1: Buy-and-Hold", "Strategy 2: Buy-and-Sell"]
)

prediction_date = st.sidebar.date_input(
    "Prediction Date",
    date.today() - timedelta(days=1),
    key="prediction_date"
)

st.write(f"Fetching trading data for **{common_ticker}** on **{prediction_date.strftime('%Y-%m-%d')}** for prediction.")

try:
    trade_df = simfin.get_share_prices(
        ticker=common_ticker,
        start=prediction_date.strftime("%Y-%m-%d"),
        end=prediction_date.strftime("%Y-%m-%d")
    )


    
    st.write("Raw API data:", trade_df.head())
    if trade_df.empty:
        st.error("No trading data available for the selected ticker on the specified date.")
    else:
        transformed_trade_df = transform_api_data(trade_df)
        st.subheader("Transformed Data for Prediction")
        st.dataframe(transformed_trade_df.head())

                
        # ADD THIS FOR DEBUGGING:
        st.write("Transformed trade data (debugging):")
        st.dataframe(transformed_trade_df.head())

        signals = predict_signals(transformed_trade_df)

        st.subheader("Debugging Model Predictions")
        st.write("### Model Predictions for Debugging")
        st.dataframe(signals.head())  # existing line

        # -----------------------------------------------------
        signals = predict_signals(transformed_trade_df)

        # -----------------------------------------------------
        # 🔍 Debugging Section - Prints Model Predictions
        # -----------------------------------------------------
        st.subheader("Debugging Model Predictions")

        if not signals.empty:
            st.write("### Model Predictions for Debugging")
            st.dataframe(signals.head())  # Show raw prediction output

            predicted_label = signals.iloc[0]["Predicted Signal"]
            probability = signals.iloc[0]["Buy Probability"]

            st.write(f"**Predicted Signal:** {predicted_label}")
            st.write(f"**Buy Probability:** {probability:.4f}")

        # -----------------------------------------------------

        st.subheader("Predicted Trading Signal (Aligned with Backtesting)")

        if not signals.empty:
            predicted_label = signals.iloc[0]["Predicted Signal"]
            probability = signals.iloc[0]["Buy Probability"]
            
            threshold_buy = 0.52
            threshold_sell = 0.48

            if strategy_choice == "Strategy 1: Buy-and-Hold":
                if probability > threshold_buy:
                    action_text = "BUY 1 share"
                    color = "#2ecc71"
                else:
                    action_text = "HOLD"
                    color = "#3498db"

            elif strategy_choice == "Strategy 2: Buy-and-Sell":
                if probability > threshold_buy:
                    action_text = "BUY 1 share"
                    color = "#2ecc71"
                elif probability < threshold_sell:
                    action_text = "SELL 1 share"
                    color = "#e74c3c"
                else:
                    action_text = "HOLD"
                    color = "#3498db"

            st.markdown(
                f"""<h2 style="color: {color};">ACTION: {action_text.upper()} NOW!</h2>""",
                unsafe_allow_html=True
            )
        else:
            st.error("No trading signals generated.")


except Exception as e:
    st.error(f"Error generating trading signals: {e}")
