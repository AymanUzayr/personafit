"""
Minimal PersonaFit App - Basic Functionality Test
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Configure Streamlit page
st.set_page_config(
    page_title="PersonaFit - Minimal Test",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_minimal_dashboard():
    """Render minimal dashboard"""
    st.title("PersonaFit - Minimal Test")
    st.write("Testing basic app functionality...")
    
    # Test session state
    if 'test_counter' not in st.session_state:
        st.session_state.test_counter = 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Functionality")
        if st.button("Increment Counter"):
            st.session_state.test_counter += 1
        st.write(f"Counter: {st.session_state.test_counter}")
        
        if st.button("Reset Counter"):
            st.session_state.test_counter = 0
            st.rerun()
    
    with col2:
        st.subheader("Component Tests")
        
        # Test basic imports
        try:
            import pandas as pd
            st.success("✓ pandas")
        except:
            st.error("✗ pandas")
        
        try:
            import numpy as np
            st.success("✓ numpy")
        except:
            st.error("✗ numpy")
        
        try:
            import plotly.express as px
            st.success("✓ plotly")
        except:
            st.error("✗ plotly")

def render_minimal_health():
    """Render minimal health prediction"""
    st.title("Health Prediction - Minimal")
    
    try:
        from health_prediction import HealthPredictor
        st.success("✓ HealthPredictor imported")
        
        # Create predictor without wearable data
        predictor = HealthPredictor()
        st.success("✓ HealthPredictor created")
        
        # Test basic functionality
        st.subheader("Fatigue Prediction Test")
        
        col1, col2 = st.columns(2)
        with col1:
            duration = st.slider("Duration (min)", 0, 120, 45)
            intensity = st.slider("Intensity (1-10)", 1, 10, 6)
            hr_avg = st.slider("Heart Rate", 60, 200, 140)
        
        with col2:
            sleep_hrs = st.slider("Sleep (hours)", 4, 12, 8)
            stress = st.slider("Stress (1-10)", 1, 10, 4)
            rest_days = st.slider("Rest Days", 0, 7, 1)
            prev_fatigue = st.slider("Prev Fatigue (1-10)", 1, 10, 5)
        
        if st.button("Predict Fatigue"):
            try:
                fatigue, recovery = predictor.predict_fatigue(
                    duration, intensity, hr_avg, sleep_hrs, stress, rest_days, prev_fatigue
                )
                
                if fatigue and recovery:
                    st.success(f"Fatigue: {fatigue:.1f}/10")
                    st.success(f"Recovery: {recovery:.1f}/10")
                    
                    # Show advice
                    advice = predictor.get_recovery_advice(fatigue, recovery)
                    st.subheader("Recommendations")
                    for tip in advice:
                        st.write(f"• {tip}")
                else:
                    st.error("Prediction failed")
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.exception(e)
        
    except Exception as e:
        st.error(f"Health prediction import failed: {e}")
        st.exception(e)

def main():
    """Main application function"""
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'dashboard'
    
    # Simple sidebar
    with st.sidebar:
        st.title("PersonaFit")
        st.write("Minimal Test App")
        
        if st.button("Dashboard"):
            st.session_state.current_page = 'dashboard'
            st.rerun()
        
        if st.button("Health"):
            st.session_state.current_page = 'health'
            st.rerun()
        
        st.write("---")
        st.write("Current page:", st.session_state.current_page)
    
    # Render current page
    page = st.session_state.current_page
    
    if page == 'dashboard':
        render_minimal_dashboard()
    elif page == 'health':
        render_minimal_health()

if __name__ == "__main__":
    main() 