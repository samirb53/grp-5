import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from ML_MODEL_ATTEMPT import predict_signals  # Your model function
from ML_MODEL_ATTEMPT import xgb_model, X_test, y_test  # Already-trained model & data
from newetl import process_data  # Your ETL function
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from ML_MODEL_ATTEMPT import xgb_model, X_test, y_test, predict_signals, features


# -----------------------------------------------------------------------------
# 🔧 SETUP LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -----------------------------------------------------------------------------
# 🔧 STREAMLIT PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Trading System",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 🔧 SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "ML Model", "Go Live", "Backtesting"]
)

# -----------------------------------------------------------------------------
# 🔧 PAGE 1: HOME
# -----------------------------------------------------------------------------
if page == "Home":
    st.title("🏠 AI Trading System – Home")
    st.markdown("""
    ## Welcome to the AI-Powered Trading System!
    
    ### Key Features
    - **Real-Time & Historical Data**: We fetch the latest market data from your ETL pipeline.
    - **Machine Learning Predictions**: Our XGBoost model predicts price movements.
    - **Buy/Sell Recommendations**: Based on predicted signals, the system advises whether to buy or sell.
    - **Backtesting**: Evaluate hypothetical investments over historical periods.
    
    ### Team Members
    - **Samir Barakat** (Lead Developer)
    - [Add your teammates here]
    
    ### Purpose
    Our goal is to help traders make **informed decisions** by leveraging **data-driven insights** and **machine learning**.
    """)

# -----------------------------------------------------------------------------
# 🔧 PAGE 2: ML MODEL
# -----------------------------------------------------------------------------
elif page == "ML Model":
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        confusion_matrix, classification_report,
        accuracy_score, roc_curve, auc
    )

    st.title("🤖 ML Model Overview")
    st.markdown("### Model Details & Performance")

    # 1) Predictions on test set
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 2) Confusion Matrix & Classification Report
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    # 3) Feature Importances
    # Use either xgb_model.feature_importances_ or get_booster().get_score().
    # Here we assume xgb_model.feature_importances_ is available:
    importances = xgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    # 4) ROC Curve
    # We need predicted probabilities for class=1
    y_probs = xgb_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    auc_val = auc(fpr, tpr)

    # ------------------------------
    # Create Tabs
    # ------------------------------
    tab_metrics, tab_confmat, tab_fi, tab_roc = st.tabs([
        "Metrics",
        "Confusion Matrix",
        "Feature Importances",
        "ROC Curve"
    ])

    # =============== TAB 1: METRICS ===============
    with tab_metrics:
        st.subheader("Overall Metrics")
        st.write(f"**Accuracy**: {accuracy:.2f}")

        st.subheader("Classification Report")
        st.dataframe(df_report)

        st.markdown("""
        ### Model Summary
        - Using **XGBoost** with ~50 estimators, `max_depth=3`.
        - Typical stock-prediction accuracy ~50%.
        - **Features**: Rolling averages, volatility, fundamentals, etc.
        """)

    # =============== TAB 2: CONFUSION MATRIX ===============
    with tab_confmat:
        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        cax = ax_cm.imshow(cm, interpolation='nearest', cmap='Blues')
        fig_cm.colorbar(cax)

        ax_cm.set_title("Confusion Matrix")
        tick_marks = np.arange(2)
        ax_cm.set_xticks(tick_marks)
        ax_cm.set_xticklabels(["Fall (0)", "Rise (1)"])
        ax_cm.set_yticks(tick_marks)
        ax_cm.set_yticklabels(["Fall (0)", "Rise (1)"])
        plt.setp(ax_cm.get_xticklabels(), rotation=45, ha="right")

        # Write values inside matrix cells
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")

        ax_cm.set_ylabel("Actual")
        ax_cm.set_xlabel("Predicted")
        st.pyplot(fig_cm)

    # =============== TAB 3: FEATURE IMPORTANCES ===============
    with tab_fi:
        st.subheader("Feature Importances")
        fig_fi, ax_fi = plt.subplots(figsize=(6, 4))
        ax_fi.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color="navy")
        ax_fi.invert_yaxis()  # So the most important feature is at the top
        ax_fi.set_xlabel("Importance")
        ax_fi.set_title("XGBoost Feature Importance")
        st.pyplot(fig_fi)

        st.markdown("**Feature Importance Data**:")
        st.dataframe(feature_importance_df.reset_index(drop=True))

    # =============== TAB 4: ROC CURVE ===============
    with tab_roc:
        st.subheader("ROC Curve")
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {auc_val:.2f})")
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Receiver Operating Characteristic")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

# -----------------------------------------------------------------------------
# 🔧 PAGE 3: GO LIVE
# -----------------------------------------------------------------------------
elif page == "Go Live":
    st.title("📡 Live Trading Dashboard")

    st.markdown("""
    Select a **stock ticker** from your top 10 traded list. The app will:
    1. Run your ETL pipeline (optional).
    2. Fetch historical & real-time data for that ticker.
    3. Apply the ML model for next-day signals.
    4. Show graphs & a final **buy/sell** recommendation.
    """)

    # Dropdown list for top tickers
    top_tickers = ["AAPL", "TSLA", "NVDA", "AMZN", "GOOGL", "MSFT", "META", "NFLX", "TGGI", "HCMC"]
    selected_ticker = st.selectbox("Select Ticker", top_tickers)

    st.markdown("---")
    if st.button("Fetch & Predict"):
        with st.spinner("🔄 Running ETL & Predicting..."):
            # Optional re-run ETL
            df = process_data()
            if df is None or df.empty:
                st.error("❌ ETL failed or returned no data.")
                st.stop()
            
            # Filter by selected ticker
            df_filtered = df[df["ticker"] == selected_ticker]
            if df_filtered.empty:
                st.warning(f"No data for {selected_ticker}.")
                st.stop()
            
            # Predict signals
            predictions_df = predict_signals(df_filtered)
            if "Error" in predictions_df.columns:
                st.error(predictions_df["Error"][0])
                st.stop()

            st.success("✅ Predictions complete!")
            
            # Show table with date, close, predicted signal, Action
            st.subheader(f"Predictions for {selected_ticker}")
            st.dataframe(predictions_df[["date", "close", "Predicted Signal", "Action"]].tail(20))
            
            # 📈 Plot Price
            st.subheader("Price Chart")
            chart_data_price = predictions_df[["date", "close"]].copy()
            chart_data_price.set_index("date", inplace=True)
            st.line_chart(chart_data_price)
            
            # 📉 Plot Signals
            st.subheader("Signal Chart")
            chart_data_signal = predictions_df[["date", "Scaled Signal"]].copy()
            chart_data_signal.set_index("date", inplace=True)
            st.line_chart(chart_data_signal)
            
            # ✅ Final Recommendation: Look at last row's predicted signal
            last_signal = predictions_df["Predicted Signal"].iloc[-1]
            if last_signal == 1:
                st.markdown(f"### **Recommendation**: BUY {selected_ticker} (model expects price to rise)")
            else:
                st.markdown(f"### **Recommendation**: SELL {selected_ticker} (model expects price to fall)")

# -----------------------------------------------------------------------------
# 🔧 PAGE 4: BACKTESTING
# -----------------------------------------------------------------------------
elif page == "Backtesting":
    st.title("📈 Backtesting Simulator")
    st.markdown("""
    Evaluate how different trading strategies would have performed historically using our model’s signals.
    """)

    # 🔸 Select Ticker
    ticker_bt = st.selectbox("Select Ticker for Backtest", ["AAPL", "TSLA", "NVDA", "AMZN", "GOOGL"])
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    initial_capital = st.number_input("Initial Capital ($)", min_value=100, step=100, value=1000)

    # 🔸 Choose Strategy
    st.markdown("### Choose Trading Strategy")
    strategy = st.radio(
        "Select a trading strategy:",
        ("Buy & Hold", "Buy & Sell")
    )

    if st.button("Run Backtest"):
        with st.spinner("Backtesting..."):
            # 🔸 Run ETL
            df = process_data()
            if df is None or df.empty:
                st.error("❌ ETL failed or returned no data.")
                st.stop()
            
            # 🔸 Filter data for chosen ticker & date
            df_bt = df[(df["ticker"] == ticker_bt) & (df["date"] >= pd.to_datetime(start_date))]
            if df_bt.empty:
                st.warning(f"No data for {ticker_bt} after {start_date}")
                st.stop()
            
            # 🔸 Predict signals
            pred_bt = predict_signals(df_bt)
            if "Error" in pred_bt.columns:
                st.error(pred_bt["Error"][0])
                st.stop()
            
            # 🔸 Initialize Portfolio
            balance = initial_capital
            shares_held = 0.0

            # ─────────────────────────────────────────────────────────
            # STRATEGY 1: BUY & HOLD
            #   - If signal=1 -> Buy as many shares as possible (only once)
            #   - If signal=0 -> Do nothing
            #
            # STRATEGY 2: BUY & SELL
            #   - If signal=1 -> Buy (spend all capital)
            #   - If signal=0 -> Sell (unload any shares)
            # ─────────────────────────────────────────────────────────
            bought_once = False  # For Buy & Hold, track if we've already bought

            for i, row in pred_bt.iterrows():
                signal = row["Predicted Signal"]
                price = row["close"]

                if strategy == "Buy & Hold":
                    # Only buy if signal=1 AND haven't bought before
                    if signal == 1 and not bought_once and balance > 0:
                        shares_held = balance / price
                        balance = 0
                        bought_once = True
                    # No selling in Buy & Hold. We keep holding.
                
                elif strategy == "Buy & Sell":
                    if signal == 1 and balance > 0:
                        # BUY
                        shares_held = balance / price
                        balance = 0
                    elif signal == 0 and shares_held > 0:
                        # SELL
                        balance = shares_held * price
                        shares_held = 0
            
            # 🔸 End of Backtest: If we still hold shares, convert to balance
            if shares_held > 0:
                balance = shares_held * pred_bt["close"].iloc[-1]
                shares_held = 0

            # 🔸 Final Results
            final_value = balance
            profit = final_value - initial_capital
            profit_pct = (profit / initial_capital) * 100.0

            st.success(f"**Final Portfolio Value:** ${final_value:,.2f}")
            st.write(f"**Profit:** ${profit:,.2f} ({profit_pct:.2f}%)")

            # 🔸 Display Historical Price Chart
            st.subheader("Historical Price Chart")
            chart_bt = pred_bt[["date", "close"]].copy()
            chart_bt.set_index("date", inplace=True)
            st.line_chart(chart_bt)

    st.markdown("""
    ### Disclaimer
    - This simulation ignores transaction fees, taxes, and slippage.
    - Real-world performance may differ.
    """)



