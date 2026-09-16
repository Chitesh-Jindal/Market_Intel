import streamlit as st

st.title("EquiLens")
st.write("AI-Powered Indian Stock Intelligence")

stock = st.text_input("Enter stock symbol", "RELIANCE.NS")

if st.button("Analyze"):
    st.write("Selected stock:", stock)
