import streamlit as st

st.title("Using the sidebar")

# Set up the sidebar with model parameters
st.sidebar.header("Model Parameters")
with st.sidebar:
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    top_k = st.number_input("Top-K", min_value=1, max_value=50, value=20)
    top_p = st.slider("Top-P", min_value=0.0, max_value=1.0, value=0.7)

# you can also use "st.sidebar." instead of the with block as used above.
max_new_tokens = st.sidebar.number_input("Max New Tokens", min_value=25, max_value=4096, value=1024)

st.write(f"temp: {temperature}")
st.write(f"top_k = {top_k}")
st.write(f"top_p = {top_p}")
st.write(f"max_new_tokens = {max_new_tokens}")
