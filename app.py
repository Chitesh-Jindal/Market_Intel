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
        st.write(data)
        missing = data.isnull().sum()
        #checking values in data set that are missing
        st.write("Missing Values")
        st.write(missing)
        data["Daily_Return"]=data["Close"].pct_change()*100
        average_return = data["Daily_Return"].mean()

        st.write("Average Daily Return: ",average_return,"%")
        st.line_chart(data["Close"])

