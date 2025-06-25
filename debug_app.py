"""
Debug script to identify issues with PersonaFit app
"""
import streamlit as st
import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Configure Streamlit page
st.set_page_config(
    page_title="PersonaFit Debug",
    page_icon="🔍",
    layout="wide"
)

def debug_imports():
    """Debug import issues"""
    st.header("🔍 Import Debug")
    
    imports_to_test = [
        ("streamlit", "st"),
        ("pandas", "pd"),
        ("numpy", "np"),
        ("sklearn.ensemble", "RandomForestRegressor"),
        ("plotly.express", "px"),
        ("scipy", "stats"),
        ("scipy.signal", "find_peaks"),
        ("groq", "Groq"),
        ("sqlalchemy", "create_engine"),
        ("cryptography", "hazmat"),
        ("joblib", "dump"),
        ("datetime", "datetime"),
        ("sqlite3", "connect"),
        ("os", "path"),
        ("warnings", "filterwarnings")
    ]
    
    for module, item in imports_to_test:
        try:
            if "." in module:
                # Handle submodule imports
                parts = module.split(".")
                exec(f"from {parts[0]} import {parts[1]} as {item}")
            else:
                exec(f"import {module} as {item}")
            st.success(f"✓ {module}")
        except Exception as e:
            st.error(f"✗ {module}: {str(e)}")

def debug_database():
    """Debug database issues"""
    st.header("🗄️ Database Debug")
    
    try:
        from database import DatabaseManager, create_tables
        st.success("✓ Database module imported")
        
        # Test table creation
        create_tables()
        st.success("✓ Tables created")
        
        # Test database manager
        db = DatabaseManager()
        st.success("✓ Database manager created")
        db.session.close()
        
    except Exception as e:
        st.error(f"✗ Database error: {str(e)}")
        st.exception(e)

def debug_auth():
    """Debug authentication issues"""
    st.header("🔐 Authentication Debug")
    
    try:
        from auth import AuthManager, show_login_page
        st.success("✓ Auth module imported")
        
        auth = AuthManager()
        st.success("✓ Auth manager created")
        
        # Test dev mode
        if auth.is_authenticated():
            st.success("✓ Authentication working (dev mode)")
        else:
            st.warning("⚠ Authentication not working")
            
    except Exception as e:
        st.error(f"✗ Auth error: {str(e)}")
        st.exception(e)

def debug_health_prediction():
    """Debug health prediction issues"""
    st.header("🏥 Health Prediction Debug")
    
    try:
        from health_prediction import HealthPredictor
        st.success("✓ Health prediction module imported")
        
        predictor = HealthPredictor()
        st.success("✓ Health predictor created")
        
        # Test basic functionality
        fatigue, recovery = predictor.predict_fatigue(45, 6, 140, 8, 4, 1, 5)
        if fatigue and recovery:
            st.success(f"✓ Fatigue prediction: {fatigue:.1f}, Recovery: {recovery:.1f}")
        else:
            st.warning("⚠ Fatigue prediction returned None")
            
    except Exception as e:
        st.error(f"✗ Health prediction error: {str(e)}")
        st.exception(e)

def debug_session_state():
    """Debug session state issues"""
    st.header("💾 Session State Debug")
    
    # Test basic session state
    if 'debug_counter' not in st.session_state:
        st.session_state.debug_counter = 0
    
    st.write(f"Current counter: {st.session_state.debug_counter}")
    
    if st.button("Increment"):
        st.session_state.debug_counter += 1
        st.rerun()
    
    if st.button("Reset"):
        st.session_state.debug_counter = 0
        st.rerun()

def main():
    """Main debug function"""
    st.title("🔍 PersonaFit Debug Tool")
    st.write("This tool helps identify issues with the PersonaFit application.")
    
    # Create tabs for different debug areas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📦 Imports", "🗄️ Database", "🔐 Auth", "🏥 Health", "💾 Session"
    ])
    
    with tab1:
        debug_imports()
    
    with tab2:
        debug_database()
    
    with tab3:
        debug_auth()
    
    with tab4:
        debug_health_prediction()
    
    with tab5:
        debug_session_state()
    
    # Summary
    st.header("📋 Summary")
    st.write("If you see any red error messages above, those are the issues causing blank output.")
    st.write("Green checkmarks indicate working components.")

if __name__ == "__main__":
    main() 