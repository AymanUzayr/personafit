"""
Test script for wearable device dataset integration
"""
import os
import pandas as pd
import numpy as np
import sys

try:
    print("Attempting to read Excel file...")
    df = pd.read_excel("fndds_nutrient_values.xlsx")
    print(f"Successfully read Excel file!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns[:5])}")  # Show first 5 columns
    print("\nFirst few rows:")
    print(df.head(2))
except Exception as e:
    print(f"Error reading Excel file: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"File exists: {os.path.exists('fndds_nutrient_values.xlsx')}")