"""
Simple test to check if Streamlit works
"""
import streamlit as st

st.title("Simple Test")
st.write("If you can see this, Streamlit is working!")

# Test basic functionality
if st.button("Click me"):
    st.success("Button works!")
    
# Test session state
if 'test_counter' not in st.session_state:
    st.session_state.test_counter = 0

if st.button("Increment counter"):
    st.session_state.test_counter += 1

st.write(f"Counter: {st.session_state.test_counter}") 