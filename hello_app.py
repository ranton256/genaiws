import streamlit as st

# Set up the page configuration
st.set_page_config(page_title="Hello World App", layout="wide")

# Display a title and some text
st.title("Hello World App")
st.write("This is a simple app to say hello to someone.")

# Define the input fields
name = st.text_input("Enter your name:", "World")
greeting = st.selectbox("Select a greeting:", ["Hello", "Hi", "Hey"])

# Create a button to trigger the greeting
if st.button("Say Hello"):
    st.write(f"{greeting}, {name}!")
    st.balloons()  # Show balloons when the button is clicked!

# you can run this app with this command:
# streamlit run hello_app.py

# Press Ctrl-C in the terminal to stop the app when you are done.