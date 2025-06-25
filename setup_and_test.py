"""
Setup and test script for PersonaFit
"""
import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    
    dependencies = [
        "streamlit>=1.28.0",
        "pandas>=2.0.0", 
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.17.0",
        "requests>=2.31.0",
        "sqlalchemy>=2.0.0",
        "cryptography>=41.0.0",
        "openai>=0.28.0",
        "joblib>=1.3.0",
        "python-dateutil>=2.8.0",
        "scipy>=1.11.0",
        "groq>=0.4.0"
    ]
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✓ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {dep}: {e}")
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("Creating directories...")
    
    directories = ["data", "models", "static"]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        except Exception as e:
            print(f"✗ Failed to create {directory}: {e}")
            return False
    
    return True

def test_imports():
    """Test if all imports work"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✓ streamlit")
    except ImportError as e:
        print(f"✗ streamlit: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas")
    except ImportError as e:
        print(f"✗ pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        print("✓ scikit-learn")
    except ImportError as e:
        print(f"✗ scikit-learn: {e}")
        return False
    
    try:
        import plotly.express as px
        print("✓ plotly")
    except ImportError as e:
        print(f"✗ plotly: {e}")
        return False
    
    try:
        from scipy import stats
        print("✓ scipy")
    except ImportError as e:
        print(f"✗ scipy: {e}")
        return False
    
    try:
        from groq import Groq
        print("✓ groq")
    except ImportError as e:
        print(f"✗ groq: {e}")
        return False
    
    return True

def test_database():
    """Test database functionality"""
    print("Testing database...")
    
    try:
        from database import create_tables
        create_tables()
        print("✓ Database tables created")
        return True
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("PersonaFit Setup and Test")
    print("=" * 40)
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies")
        return False
    
    # Create directories
    if not create_directories():
        print("Failed to create directories")
        return False
    
    # Test imports
    if not test_imports():
        print("Import tests failed")
        return False
    
    # Test database
    if not test_database():
        print("Database test failed")
        return False
    
    print("\n" + "=" * 40)
    print("🎉 Setup completed successfully!")
    print("\nTo run the app:")
    print("1. streamlit run main_app_minimal.py")
    print("2. streamlit run main_app.py")
    print("\nTo test basic functionality:")
    print("streamlit run simple_test.py")
    
    return True

if __name__ == "__main__":
    main() 