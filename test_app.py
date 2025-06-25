"""
Test script to check if PersonaFit components work correctly
"""
import sys
import traceback

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✓ streamlit imported successfully")
    except Exception as e:
        print(f"✗ streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except Exception as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except Exception as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        print("✓ scikit-learn imported successfully")
    except Exception as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False
    
    try:
        import plotly.express as px
        print("✓ plotly imported successfully")
    except Exception as e:
        print(f"✗ plotly import failed: {e}")
        return False
    
    try:
        from scipy import stats
        from scipy.signal import find_peaks
        print("✓ scipy imported successfully")
    except Exception as e:
        print(f"✗ scipy import failed: {e}")
        return False
    
    try:
        from groq import Groq
        print("✓ groq imported successfully")
    except Exception as e:
        print(f"✗ groq import failed: {e}")
        return False
    
    return True

def test_database():
    """Test database functionality"""
    print("\nTesting database...")
    
    try:
        from database import DatabaseManager, create_tables
        print("✓ database module imported successfully")
    except Exception as e:
        print(f"✗ database import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        create_tables()
        print("✓ database tables created successfully")
    except Exception as e:
        print(f"✗ database table creation failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        db = DatabaseManager()
        print("✓ database manager created successfully")
        db.session.close()
    except Exception as e:
        print(f"✗ database manager creation failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_auth():
    """Test authentication functionality"""
    print("\nTesting authentication...")
    
    try:
        from auth import AuthManager
        print("✓ auth module imported successfully")
    except Exception as e:
        print(f"✗ auth import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        auth = AuthManager()
        print("✓ auth manager created successfully")
    except Exception as e:
        print(f"✗ auth manager creation failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_health_prediction():
    """Test health prediction functionality"""
    print("\nTesting health prediction...")
    
    try:
        from health_prediction import HealthPredictor
        print("✓ health_prediction module imported successfully")
    except Exception as e:
        print(f"✗ health_prediction import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        predictor = HealthPredictor()
        print("✓ health predictor created successfully")
    except Exception as e:
        print(f"✗ health predictor creation failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def main():
    """Run all tests"""
    print("PersonaFit Component Test")
    print("=" * 40)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test database
    if not test_database():
        all_passed = False
    
    # Test auth
    if not test_auth():
        all_passed = False
    
    # Test health prediction
    if not test_health_prediction():
        all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! The app should work correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    main() 