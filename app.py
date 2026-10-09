import streamlit as st
import yfinance as yf;
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
#ta is technical analysis library provides implementations of common financial technical indicators.
st.title("Market Intel")
st.write("AI-Powered Indian Stock Intelligence")

stock = st.text_input("Enter stock symbol", "RELIANCE.NS")

if st.button("Analyze"):
    data = yf.download(stock, period="3mo")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    #1mo means 1 month of data
    if data.empty:
        #if the stock entered is invalid
        st.error("No data found for this stock")
    else:
        data["Daily_Return (%)"]=data["Close"].pct_change()*100
        
        st.subheader("Historical Data")
        st.write(data)
        missing = data.isnull().sum()
        #checking values in data set that are missing
        st.write("Missing Values")
        st.write(missing)

        st.divider()
        st.subheader("Analysis for " + stock)

        st.caption("Key Statistics")
        
        average_return = data["Daily_Return (%)"].mean()
        volatility=data["Daily_Return (%)"].std()
        highest_price=data["Close"].max().item()
        lowest_price=data["Close"].min().item()
        latest_price=data["Close"].iloc[-1].item()
        price_range = highest_price - lowest_price
        total_volume=data["Volume"].sum().item()
        trading_days = len(data)
        price_change = latest_price - data["Close"].iloc[0].item()
        price_change_percent = (price_change / data["Close"].iloc[0].item()) * 100

        st.write("Trading Days:", trading_days)

        st.write("Latest Closing Price: ",round(latest_price,2))
        st.write("Price Change:", round(price_change, 2))
        st.write("Highest Closing Price: ",round(highest_price,2))
        st.write("Lowest Closing Price: ",round(lowest_price,2))
        st.write("Closing Price Range:", round(price_range, 2))
        st.write("Price Change (%):", round(price_change_percent, 2), "%")
        
        st.write("Average Daily Return: ",round(average_return,2),"%")
        st.write("Daily Volatility: ",round(volatility,2),"%")
        
        
        st.write("Total Trading Volume:", total_volume)
        st.divider()

        data["SMA_10"]=data["Close"].rolling(window=10).mean()
        #SIMPLE MOVING AVERAGE
        data["SMA_20"]=data["Close"].rolling(window=20).mean()
        data["SMA_Spread"]=data["SMA_10"]-data["SMA_20"]

        #Exponential moving average
        #similar to SMA but gives more weight to recent prices, so repond faster to recent changes
        data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()

        st.line_chart(data[["Close", "SMA_10", "SMA_20", "EMA_20"]])


        #RELATIVE STRENGTH INDEX measures how strongly the price is going upwards or downwards
        #ranges from 0 to 100
        rsi_indicator= RSIIndicator(close=data["Close"], window=14)
        data["RSI"]=rsi_indicator.rsi()
        st.subheader("RSI (Relative Strength Index)")
        st.line_chart(data["RSI"])

        
        #Moving average convergence divergence
        #gives us meaure of trend/momentum
        #MACD=12 day EMA - 26 day EMA
        #+ve means upward momentum and vice versa similarly macd signal tells 9 day EMA a smoother EMA
        #if macd crosses upward macd signal means increasing upward momentuma and downward means increasing downward momentum
        macd_indicator=MACD(close=data["Close"])
        data["MACD"]=macd_indicator.macd()
        data["MACD_Signal"]=macd_indicator.macd_signal();
        st.subheader("MACD (Moving Average Converge Divergence)")
        st.line_chart(data[["MACD","MACD_Signal"]])

        
        #Bollinger Bands 
        #- Middle Band → usually 20-day SMA
        #- Upper Band → middle band + 2 standard deviations
        #- Lower Band → middle band − 2 standard deviations
        bb_indicator = BollingerBands(close=data["Close"], window=20, window_dev=2)

        data["BB_Upper"] = bb_indicator.bollinger_hband()
        data["BB_Middle"] = bb_indicator.bollinger_mavg()
        data["BB_Lower"] = bb_indicator.bollinger_lband()
        st.subheader("Bollinger Bands")
        st.line_chart(data[["Close", "BB_Upper", "BB_Middle", "BB_Lower"]])


        #combined tech analysis section
        st.subheader("Latest Technical Indicators")

        latest = data.iloc[-1]

        st.write("SMA 10:", round(latest["SMA_10"], 2))
        st.write("SMA 20:", round(latest["SMA_20"], 2))
        st.write("EMA 20:", round(latest["EMA_20"], 2))
        st.write("RSI:", round(latest["RSI"], 2))
        st.write("MACD:", round(latest["MACD"], 2))
        st.write("MACD Signal:", round(latest["MACD_Signal"], 2))
        st.write("Bollinger Upper:", round(latest["BB_Upper"], 2))
        st.write("Bollinger Middle:", round(latest["BB_Middle"], 2))
        st.write("Bollinger Lower:", round(latest["BB_Lower"], 2))

        #what these values indicate
        #trend= sma/ema
        st.subheader("Technical Signal Summary")
        if latest["SMA_10"]> latest["SMA_20"] and latest["EMA_20"]> latest["SMA_20"]:
            trend_signal = "Positive"
        elif latest["SMA_10"] < latest["SMA_20"] and latest["EMA_20"] < latest["SMA_20"]:
            trend_signal = "Negative"
        else:
            trend_signal = "Mixed"
            
        #momentum= rsi/macd
        if latest["RSI"] > 70:
            momentum_signal = "Strong / Overbought"
        elif latest["RSI"] < 30:
            momentum_signal = "Weak / Oversold"
        elif latest["MACD"] > latest["MACD_Signal"]:
            momentum_signal = "Positive"  
        else:
            momentum_signal = "Negative" 

        #volatility= bollinger bands
        if latest["Close"] > latest["BB_Upper"]:
            volatility_signal = "High / Above Upper Band"
        elif latest["Close"] < latest["BB_Lower"]:
            volatility_signal = "High / Below Lower Band"
        else:
            volatility_signal = "Normal"

        st.write("Trend:", trend_signal)
        st.write("Momentum:", momentum_signal)
        st.write("Volatility:", volatility_signal)

        #ml dataset
        st.subheader("ML Dataset")
        data["Next_Day_Return (%)"] = ( data["Close"].shift(-1) / data["Close"] - 1 ) * 100
        ml_data = data[
            [
                "Close",
                "Volume",
                "Daily_Return (%)",
                "SMA_10",
                "SMA_20",
                "SMA_Spread",
                "EMA_20",
                "RSI",
                "MACD",
                "MACD_Signal",
                "BB_Upper",
                "BB_Middle",
                "BB_Lower"
                "Next_Day_Return (%)"
            ]
        ].dropna()
        
        st.write(ml_data)

