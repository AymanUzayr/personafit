"""
PersonaFit Configuration & Constants
"""
import os
from pathlib import Path

# Application Configuration
APP_NAME = "PersonaFit"
APP_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Database Configuration
BASE_DIR = Path(__file__).parent
DATABASE_URL = f"sqlite:///{BASE_DIR}/persona_fit.db"

# Security Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
PASSWORD_SALT_LENGTH = 32
SESSION_TIMEOUT_HOURS = 24

# API Configuration
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Data Sources
USDA_FOOD_DATA_URL = "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_foundation_food_csv_2023-04-20.zip"
USDA_BRANDED_DATA_URL = "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_branded_food_csv_2023-04-20.zip"
EXERCISE_DATA_URL = "https://raw.githubusercontent.com/pannaf/valkyrie/main/final_exercise_list.csv"
IMU_DATA_URL = "https://www.mdpi.com/2306-5729/8/1/9"  # Need to implement scraper

# File Paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, STATIC_DIR]:
    directory.mkdir(exist_ok=True)

# Nutrition Constants
CALORIES_PER_GRAM = {
    'protein': 4,
    'carbs': 4,
    'fat': 9,
    'alcohol': 7
}

# Activity Levels & Multipliers
ACTIVITY_LEVELS = {
    'sedentary': 1.2,
    'lightly_active': 1.375,
    'moderately_active': 1.55,
    'very_active': 1.725,
    'extremely_active': 1.9
}

# Fitness Goals
FITNESS_GOALS = {
    'weight_loss': {'calorie_deficit': 500, 'protein_ratio': 0.3, 'carb_ratio': 0.4, 'fat_ratio': 0.3},
    'muscle_gain': {'calorie_surplus': 300, 'protein_ratio': 0.25, 'carb_ratio': 0.45, 'fat_ratio': 0.3},
    'maintenance': {'calorie_change': 0, 'protein_ratio': 0.2, 'carb_ratio': 0.5, 'fat_ratio': 0.3},
    'endurance': {'calorie_surplus': 200, 'protein_ratio': 0.15, 'carb_ratio': 0.65, 'fat_ratio': 0.2}
}

# UI Configuration
THEME_CONFIG = {
    'primary_color': '#FF6B6B',
    'background_color': '#FFFFFF',
    'secondary_background_color': '#F0F2F6',
    'text_color': '#262730',
    'font': 'sans serif'
}

# Streamlit Page Config
PAGE_CONFIG = {
    'page_title': APP_NAME,
    'page_icon': '💪',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}