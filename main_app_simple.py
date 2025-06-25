"""
Simplified PersonaFit App for Testing
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Configure Streamlit page
st.set_page_config(
    page_title="PersonaFit - Your AI Fitness Coach",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function"""
    st.title("PersonaFit - Simplified Test")
    st.write("Testing basic functionality...")
    
    # Test basic imports
    try:
        import pandas as pd
        st.success("✓ pandas imported successfully")
    except Exception as e:
        st.error(f"✗ pandas import failed: {e}")
    
    try:
        import numpy as np
        st.success("✓ numpy imported successfully")
    except Exception as e:
        st.error(f"✗ numpy import failed: {e}")
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        st.success("✓ scikit-learn imported successfully")
    except Exception as e:
        st.error(f"✗ scikit-learn import failed: {e}")
    
    try:
        import plotly.express as px
        st.success("✓ plotly imported successfully")
    except Exception as e:
        st.error(f"✗ plotly import failed: {e}")
    
    try:
        from scipy import stats
        st.success("✓ scipy imported successfully")
    except Exception as e:
        st.error(f"✗ scipy import failed: {e}")
    
    # Test database
    try:
        from database import DatabaseManager, create_tables
        st.success("✓ database module imported successfully")
        
        # Try to create tables
        create_tables()
        st.success("✓ database tables created successfully")
        
        # Try to create database manager
        db = DatabaseManager()
        st.success("✓ database manager created successfully")
        db.session.close()
        
    except Exception as e:
        st.error(f"✗ database test failed: {e}")
        st.exception(e)
    
    # Test auth
    try:
        from auth import AuthManager
        st.success("✓ auth module imported successfully")
        
        auth = AuthManager()
        st.success("✓ auth manager created successfully")
        
    except Exception as e:
        st.error(f"✗ auth test failed: {e}")
        st.exception(e)
    
    # Test health prediction
    try:
        from health_prediction import HealthPredictor
        st.success("✓ health_prediction module imported successfully")
        
        predictor = HealthPredictor()
        st.success("✓ health predictor created successfully")
        
    except Exception as e:
        st.error(f"✗ health prediction test failed: {e}")
        st.exception(e)
    
    st.write("---")
    st.write("If you can see all the success messages above, the app should work correctly!")
    
    # Simple navigation test
    st.subheader("Navigation Test")
    page = st.selectbox("Choose a page", ["Home", "About", "Contact"])
    
    if page == "Home":
        st.write("Welcome to the home page!")
    elif page == "About":
        st.write("This is the about page!")
    else:
        st.write("Contact us at test@example.com")

if __name__ == "__main__":
    main() 