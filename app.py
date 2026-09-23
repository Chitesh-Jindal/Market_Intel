import streamlit as st
import yfinance as yf;
st.title("Market Intel")
st.write("AI-Powered Indian Stock Intelligence")

stock = st.text_input("Enter stock symbol", "RELIANCE.NS")

if st.button("Analyze"):
    data = yf.download(stock, period="1mo")
    #1mo means 1 month of data
    if data.empty:
        #if the stock entered is invalid
        st.error("No data found for this stock")
    else:
        data["Daily_Return (%)"]=data["Close"].pct_change()*100
        data["SMA_10"]=data["Close"].rolling(window=10).mean()
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
        st.line_chart(data[["Close", "SMA_10"]])

