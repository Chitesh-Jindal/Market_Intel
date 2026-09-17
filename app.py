import streamlit as st
import yfinance as yf;
st.title("Market Intel")
st.write("AI-Powered Indian Stock Intelligence")

stock = st.text_input("Enter stock symbol", "RELIANCE.NS")

if st.button("Analyze"):
    data = yf.download(stock, period="1mo")
    #1mo means 1 month of data
    if data.empty:
        st.error("No data found for this stock")
    else:
        st.write(data)
        st.line_chart(data["Close"])

