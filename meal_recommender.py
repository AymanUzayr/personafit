"""
PersonaFit Meal Recommender - USDA Food Database Integration
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
import requests
import zipfile
import os
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from database import DatabaseManager, FoodDatabase
from config import DATA_DIR, MODELS_DIR, USDA_FOOD_DATA_URL, USDA_BRANDED_DATA_URL, FITNESS_GOALS, CALORIES_PER_GRAM
import ast

class MealRecommender:
    def __init__(self):
        self.food_data = None
        self.tfidf_vectorizer = None
        self.food_features = None
        self.nutrition_model = None
        self.load_or_create_models()
        
            
    def process_usda_data(self):
        """Process and clean USDA food data (no download, use local files)"""

        # Load and process FNDDS data
        fndds_df = pd.read_excel("fndds_nutrient_values.xlsx")  # type: ignore
        
        # Add source identifier
        fndds_df['source'] = 'fndds'

        # Select relevant columns (nutrients are already per 100g)
        # Keep the existing nutrient columns from FNDDS dataset
        
        # Save processed data
        processed_file = DATA_DIR / "processed" / "fndds_foods.csv"
        processed_file.parent.mkdir(exist_ok=True)
        fndds_df.to_csv(processed_file, index=False)

        return fndds_df

    def load_food_database(self):
        """Load food database from processed file or create new one"""
        processed_file = DATA_DIR / "processed" / "fndds_foods.csv"
        
        if processed_file.exists():
            return pd.read_csv(processed_file)
        else:
            return self.process_usda_data()

    def create_food_features(self, food_df):
        """Create TF-IDF features from food descriptions"""
        # Use the actual description column from FNDDS
        food_df['combined_text'] = food_df['Main food description'].fillna('')
        
        # Create TF-IDF features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.food_features = self.tfidf_vectorizer.fit_transform(food_df['combined_text'])
        
        # Save vectorizer
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(self.tfidf_vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
        joblib.dump(self.food_features, MODELS_DIR / "food_features.pkl")

    def train_nutrition_model(self, food_df):
        """Train model to predict nutritional satisfaction"""
        # Create features using actual FNDDS nutrient columns
        features = []
        
        for _, row in food_df.iterrows():
            feature_vector = [
                row['Energy (kcal)'],
                row['Protein (g)'],
                row['Carbohydrate, by difference (g)'],
                row['Total lipid (fat) (g)'],
                row['Fiber, total dietary (g)'],
                row['Protein (g)'] / max(row['Energy (kcal)'], 1),  # Protein ratio
                row['Fiber, total dietary (g)'] / max(row['Carbohydrate, by difference (g)'], 1),  # Fiber ratio
                row['Sodium, Na (mg)']
            ]
            features.append(feature_vector)
        
        X = np.array(features)
        
        # Create nutritional quality score using actual nutrients
        y = (food_df['Protein (g)'] * 0.3 + 
            food_df['Fiber, total dietary (g)'] * 0.3 + 
            (100 - food_df['Sodium, Na (mg)'] / 20) * 0.2 +
            (500 - food_df['Energy (kcal)']) / 10 * 0.2)
        
        # Train Random Forest model
        self.nutrition_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.nutrition_model.fit(X, y)
        
        # Save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(self.nutrition_model, MODELS_DIR / "nutrition_model.pkl")
        
    def load_or_create_models(self):
        """Load existing models or create new ones"""
        try:
            import joblib
            self.tfidf_vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
            self.food_features = joblib.load(MODELS_DIR / "food_features.pkl")
            self.nutrition_model = joblib.load(MODELS_DIR / "nutrition_model.pkl")
            self.food_data = self.load_food_database()
        except:
            st.info("Building food recommendation models...")
            self.food_data = self.load_food_database()
            if self.food_data is not None:
                self.create_food_features(self.food_data)
                self.train_nutrition_model(self.food_data)
    
    def calculate_daily_calories(self, user):
        """Calculate daily calorie needs based on user profile"""
        if not all([user.age, user.height, user.weight, user.gender, user.activity_level]):
            return 2000  # Default calories
        
        # Mifflin-St Jeor Equation
        if str(user.gender).lower() == 'male':
            bmr = 10 * user.weight + 6.25 * user.height - 5 * user.age + 5
        else:
            bmr = 10 * user.weight + 6.25 * user.height - 5 * user.age - 161
        
        # Apply activity multiplier
        from config import ACTIVITY_LEVELS
        activity_multiplier = ACTIVITY_LEVELS.get(user.activity_level, 1.5)
        tdee = bmr * activity_multiplier
        
        # Apply goal adjustment
        goal_config = FITNESS_GOALS.get(user.fitness_goal, FITNESS_GOALS['maintenance'])
        calorie_adjustment = goal_config.get('calorie_deficit', 0) - goal_config.get('calorie_surplus', 0)
        
        return int(tdee - calorie_adjustment)
    
    def get_macro_targets(self, user, daily_calories):
        """Calculate macro targets based on user goals"""
        goal_config = FITNESS_GOALS.get(user.fitness_goal, FITNESS_GOALS['maintenance'])
        
        protein_calories = daily_calories * goal_config['protein_ratio']
        carb_calories = daily_calories * goal_config['carb_ratio']
        fat_calories = daily_calories * goal_config['fat_ratio']
        
        return {
            'protein_g': protein_calories / CALORIES_PER_GRAM['protein'],
            'carbs_g': carb_calories / CALORIES_PER_GRAM['carbs'],
            'fat_g': fat_calories / CALORIES_PER_GRAM['fat']
        }
    
    def search_foods(self, query, limit=20):
        """Search for foods using TF-IDF similarity"""
        if not self.tfidf_vectorizer or self.food_data is None:
            return []
        
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.food_features).flatten()
        
        # Get top matches
        top_indices = similarities.argsort()[-limit:][::-1]
        
        results = []
        for idx in top_indices:
            food = self.food_data.iloc[idx]
            results.append({
                'fdc_id': food['fdc_id'],
                'description': food['description'],
                'brand_name': food.get('brand_name', ''),
                'calories_per_100g': food['calories_per_100g'],
                'protein_per_100g': food['protein_per_100g'],
                'carbs_per_100g': food['carbs_per_100g'],
                'fat_per_100g': food['fat_per_100g'],
                'similarity': similarities[idx]
            })
        
        return results
    
    def recommend_meal_plan(self, user, target_calories, macro_targets):
        """Generate personalized meal plan"""
        # Robust user field handling for preferences and allergies
        try:
            dietary_prefs = ast.literal_eval(str(getattr(user, 'dietary_preferences', '[]')))
            if not isinstance(dietary_prefs, list):
                dietary_prefs = []
        except Exception:
            dietary_prefs = []
        try:
            allergies = ast.literal_eval(str(getattr(user, 'allergies', '[]')))
            if not isinstance(allergies, list):
                allergies = []
        except Exception:
            allergies = []
        
        # Define meal distribution
        meal_distribution = {
            'breakfast': 0.25,
            'lunch': 0.35,
            'dinner': 0.30,
            'snack': 0.10
        }
        
        meal_plan = {}
        
        for meal_type, calorie_ratio in meal_distribution.items():
            meal_calories = target_calories * calorie_ratio
            meal_protein = macro_targets['protein_g'] * calorie_ratio
            meal_carbs = macro_targets['carbs_g'] * calorie_ratio
            meal_fat = macro_targets['fat_g'] * calorie_ratio
            
            # Generate meal recommendations
            meal_foods = self.generate_meal_foods(
                meal_type, meal_calories, meal_protein, meal_carbs, meal_fat,
                dietary_prefs, allergies
            )
            
            meal_plan[meal_type] = {
                'target_calories': meal_calories,
                'target_protein': meal_protein,
                'target_carbs': meal_carbs,
                'target_fat': meal_fat,
                'foods': meal_foods
            }
        
        return meal_plan
    
    def generate_meal_foods(self, meal_type, calories, protein, carbs, fat, dietary_prefs, allergies):
        """Generate food recommendations for a specific meal"""
        # Define meal-specific food categories
        meal_categories = {
            'breakfast': ['cereal', 'egg', 'toast', 'fruit', 'yogurt', 'oatmeal'],
            'lunch': ['sandwich', 'salad', 'soup', 'chicken', 'vegetables'],
            'dinner': ['meat', 'fish', 'rice', 'pasta', 'vegetables', 'potato'],
            'snack': ['nuts', 'fruit', 'yogurt', 'crackers', 'cheese']
        }
        
        categories = meal_categories.get(meal_type, ['food'])
        
        foods = []
        remaining_calories = calories
        
        for category in categories[:3]:  # Limit to 3 foods per meal
            if remaining_calories <= 0:
                break
            
            # Search for foods in this category
            search_results = self.search_foods(category, limit=10)
            
            if search_results:
                # Filter based on dietary preferences and allergies
                filtered_foods = self.filter_foods(search_results, dietary_prefs, allergies)
                
                if filtered_foods:
                    food = filtered_foods[0]  # Take the best match
                    
                    # Calculate appropriate portion size
                    portion_size = min(200, remaining_calories / food['calories_per_100g'] * 100)
                    
                    food_info = {
                        'name': food['description'],
                        'portion_g': portion_size,
                        'calories': food['calories_per_100g'] * portion_size / 100,
                        'protein': food['protein_per_100g'] * portion_size / 100,
                        'carbs': food['carbs_per_100g'] * portion_size / 100,
                        'fat': food['fat_per_100g'] * portion_size / 100
                    }
                    
                    foods.append(food_info)
                    remaining_calories -= food_info['calories']
        
        return foods
    
    def filter_foods(self, foods, dietary_prefs, allergies):
        """Filter foods based on dietary preferences and allergies"""
        filtered = []
        
        for food in foods:
            description = food['description'].lower()
            
            # Check allergies
            allergy_conflict = False
            for allergy in allergies:
                if allergy.lower() in description:
                    allergy_conflict = True
                    break
            
            if allergy_conflict:
                continue
            
            # Check dietary preferences
            if 'vegetarian' in dietary_prefs:
                if any(meat in description for meat in ['chicken', 'beef', 'pork', 'fish', 'turkey']):
                    continue
            
            if 'vegan' in dietary_prefs:
                if any(animal in description for animal in ['chicken', 'beef', 'pork', 'fish', 'turkey', 'cheese', 'milk', 'egg']):
                    continue
            
            filtered.append(food)
        
        return filtered

    def generate_quick_meal(self, user, target_calories):
        """Generate a single balanced meal quickly"""
        categories = ['protein', 'carbohydrates', 'vegetables']
        meal_foods = []
        remaining_calories = target_calories
        for category in categories:
            if remaining_calories <= 0:
                break
            search_results = self.search_foods(category, limit=5)
            if search_results:
                food = search_results[0]
                portion_size = min(150, remaining_calories / food['calories_per_100g'] * 100)
                if portion_size > 20:
                    food_info = {
                        'name': food['description'],
                        'portion_g': portion_size,
                        'calories': food['calories_per_100g'] * portion_size / 100,
                        'protein': food['protein_per_100g'] * portion_size / 100,
                        'carbs': food['carbs_per_100g'] * portion_size / 100,
                        'fat': food['fat_per_100g'] * portion_size / 100
                    }
                    meal_foods.append(food_info)
                    remaining_calories -= food_info['calories']
        return meal_foods if meal_foods else None

def show_meal_recommender():
    """Display meal recommender interface"""
    st.title("🍽️ Personalized Meal Planner")
    
    # Initialize recommender
    if 'meal_recommender' not in st.session_state:
        with st.spinner("Initializing meal recommender..."):
            st.session_state.meal_recommender = MealRecommender()
    
    recommender = st.session_state.meal_recommender
    
    # Get current user
    from auth import AuthManager
    auth_manager = AuthManager()
    user = auth_manager.get_current_user()
    
    if not user:
        st.error("Please login to access meal recommendations")
        return
    
    # Check if user profile is complete
    if not all([user.age, user.height, user.weight, user.activity_level, user.fitness_goal]):
        st.warning("Please complete your profile to get personalized recommendations")
        if st.button("Go to Profile"):
            st.switch_page("Profile")
        return
    
    # Create tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Daily Targets", "🍽️ Meal Plan", "🔍 Food Search", "📝 Meal Log"])
    
    # Calculate daily targets
    daily_calories = recommender.calculate_daily_calories(user)
    macro_targets = recommender.get_macro_targets(user, daily_calories)
    
    with tab1:
        st.subheader("Your Daily Nutrition Targets")
        
        # Display targets in an attractive layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Daily Calories", f"{daily_calories:,}", help="Based on your BMR and activity level")
        with col2:
            st.metric("Protein", f"{macro_targets['protein_g']:.0f}g", 
                    help=f"{(macro_targets['protein_g']*4/daily_calories*100):.0f}% of total calories")
        with col3:
            st.metric("Carbohydrates", f"{macro_targets['carbs_g']:.0f}g", 
                    help=f"{(macro_targets['carbs_g']*4/daily_calories*100):.0f}% of total calories")
        with col4:
            st.metric("Fat", f"{macro_targets['fat_g']:.0f}g", 
                    help=f"{(macro_targets['fat_g']*9/daily_calories*100):.0f}% of total calories")
        
        # Macro distribution chart
        st.subheader("Macronutrient Distribution")
        macro_data = {
            'Nutrient': ['Protein', 'Carbohydrates', 'Fat'],
            'Grams': [macro_targets['protein_g'], macro_targets['carbs_g'], macro_targets['fat_g']],
            'Calories': [macro_targets['protein_g']*4, macro_targets['carbs_g']*4, macro_targets['fat_g']*9]
        }
        
        fig = px.pie(values=macro_data['Calories'], names=macro_data['Nutrient'], 
                    title="Calorie Distribution by Macronutrient",
                    color_discrete_map={'Protein': '#FF6B6B', 'Carbohydrates': '#4ECDC4', 'Fat': '#45B7D1'})
        st.plotly_chart(fig, use_container_width=True)
        
        # BMR and TDEE breakdown
        with st.expander("📈 Calorie Calculation Breakdown"):
            if str(user.gender).lower() == 'male':
                bmr = 10 * user.weight + 6.25 * user.height - 5 * user.age + 5
            else:
                bmr = 10 * user.weight + 6.25 * user.height - 5 * user.age - 161
            
            from config import ACTIVITY_LEVELS
            activity_multiplier = ACTIVITY_LEVELS.get(str(user.activity_level), 1.5)
            tdee = bmr * activity_multiplier
            
            st.write(f"**Basal Metabolic Rate (BMR):** {bmr:.0f} calories")
            st.write(f"**Activity Level:** {user.activity_level} (×{activity_multiplier})")
            st.write(f"**Total Daily Energy Expenditure (TDEE):** {tdee:.0f} calories")
            st.write(f"**Goal Adjustment:** {daily_calories - tdee:+.0f} calories")
            st.write(f"**Final Target:** {daily_calories} calories")
    
    with tab2:
        st.subheader("🍽️ Generate Your Meal Plan")
        
        # Meal plan customization
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dietary Preferences:**")
            dietary_prefs = st.multiselect(
                "Select your dietary preferences",
                ["vegetarian", "vegan", "keto", "paleo", "mediterranean", "low_carb", "high_protein"],
                default=ast.literal_eval(str(getattr(user, 'dietary_preferences', '[]'))) if getattr(user, 'dietary_preferences', None) else []
            )
            meals_per_day = st.selectbox("Meals per day", [3, 4, 5, 6], index=1)
            
        with col2:
            st.write("**Allergies & Restrictions:**")
            allergies = st.multiselect(
                "Select any allergies or foods to avoid",
                ["nuts", "dairy", "gluten", "eggs", "fish", "shellfish", "soy", "peanuts"],
                default=ast.literal_eval(str(getattr(user, 'allergies', '[]'))) if getattr(user, 'allergies', None) else []
            )
            
            include_snacks = st.checkbox("Include snacks", value=True)
        
        # Generate meal plan button
        if st.button("🎯 Generate Personalized Meal Plan", type="primary"):
            with st.spinner("Creating your personalized meal plan..."):
                # Update user preferences in DB
                try:
                    with DatabaseManager() as db:
                        db.update_user_profile(user.id, {
                            'dietary_preferences': str(dietary_prefs),
                            'allergies': str(allergies)
                        })
                except Exception as e:
                    st.warning(f"Could not update preferences: {e}")
                meal_plan = recommender.recommend_meal_plan(user, daily_calories, macro_targets)
                st.success("✅ Your meal plan is ready!")
                
                # Display meal plan
                total_plan_calories = 0
                total_plan_protein = 0
                total_plan_carbs = 0
                total_plan_fat = 0
                
                for meal_type, meal_info in meal_plan.items():
                    with st.expander(f"🍴 {meal_type.title()} - Target: {meal_info['target_calories']:.0f} calories", expanded=True):
                        
                        if meal_info['foods']:
                            # Create a nice table for foods
                            food_df = pd.DataFrame(meal_info['foods'])
                            food_df['portion_g'] = food_df['portion_g'].round(0).astype(int)
                            food_df['calories'] = food_df['calories'].round(0).astype(int)
                            food_df['protein'] = food_df['protein'].round(1)
                            food_df['carbs'] = food_df['carbs'].round(1)
                            food_df['fat'] = food_df['fat'].round(1)
                            
                            st.dataframe(
                                food_df[['name', 'portion_g', 'calories', 'protein', 'carbs', 'fat']],
                                column_config={
                                    "name": "Food Item",
                                    "portion_g": "Portion (g)",
                                    "calories": "Calories",
                                    "protein": "Protein (g)",
                                    "carbs": "Carbs (g)",
                                    "fat": "Fat (g)"
                                },
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Calculate and show totals
                            meal_calories = sum(food['calories'] for food in meal_info['foods'])
                            meal_protein = sum(food['protein'] for food in meal_info['foods'])
                            meal_carbs = sum(food['carbs'] for food in meal_info['foods'])
                            meal_fat = sum(food['fat'] for food in meal_info['foods'])
                            
                            total_plan_calories += meal_calories
                            total_plan_protein += meal_protein
                            total_plan_carbs += meal_carbs
                            total_plan_fat += meal_fat
                            
                            # Progress bars for targets
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                def safe_float(val):
                                    try:
                                        # SQLAlchemy columns have a .__class__.__name__ of 'InstrumentedAttribute'
                                        if hasattr(val, 'name') and hasattr(val, 'type'):
                                            return 0.0
                                        return float(val)
                                    except Exception:
                                        return 0.0
                                cal_val = safe_float(meal_calories)
                                daily_val = safe_float(meal_info['target_calories'])
                                try:
                                    cal_progress = cal_val / daily_val if daily_val else 0.0
                                except Exception:
                                    cal_progress = 0.0
                                cal_progress = min(max(cal_progress, 0.0), 1.0)
                                st.metric("Calories", f"{cal_val:.0f} / {daily_val:.0f}", 
                                        f"{cal_val - daily_val:+.0f}")
                                st.progress(cal_progress)
                            
                            with col2:
                                protein_val = float(meal_protein or 0)
                                protein_target = float(meal_info['target_protein'] or 1)
                                protein_progress = min(protein_val / protein_target, 1.0)
                                st.metric("Protein", f"{protein_val:.1f}g / {protein_target:.0f}g", 
                                        f"{protein_val - protein_target:+.1f}g")
                                st.progress(protein_progress)
                            
                            with col3:
                                carbs_val = float(meal_carbs or 0)
                                carbs_target = float(meal_info['target_carbs'] or 1)
                                carbs_progress = min(carbs_val / carbs_target, 1.0)
                                st.metric("Carbs", f"{carbs_val:.1f}g / {carbs_target:.0f}g", 
                                        f"{carbs_val - carbs_target:+.1f}g")
                                st.progress(carbs_progress)
                            
                            with col4:
                                fat_val = float(meal_fat or 0)
                                fat_target = float(meal_info['target_fat'] or 1)
                                fat_progress = min(fat_val / fat_target, 1.0)
                                st.metric("Fat", f"{fat_val:.1f}g / {fat_target:.0f}g", 
                                        f"{fat_val - fat_target:+.1f}g")
                                st.progress(fat_progress)
                        else:
                            st.warning("No suitable foods found for this meal. Try adjusting your preferences.")
                
                # Overall plan summary
                st.subheader("📊 Daily Plan Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Calories", f"{total_plan_calories:.0f}", 
                            f"{total_plan_calories - daily_calories:+.0f} vs target")
                with col2:
                    st.metric("Total Protein", f"{total_plan_protein:.1f}g", 
                            f"{total_plan_protein - macro_targets['protein_g']:+.1f}g vs target")
                with col3:
                    st.metric("Total Carbs", f"{total_plan_carbs:.1f}g", 
                            f"{total_plan_carbs - macro_targets['carbs_g']:+.1f}g vs target")
                with col4:
                    st.metric("Total Fat", f"{total_plan_fat:.1f}g", 
                            f"{total_plan_fat - macro_targets['fat_g']:+.1f}g vs target")
                
                # Save meal plan option
                if st.button("💾 Save This Meal Plan", type="secondary"):
                    try:
                        with DatabaseManager() as db:
                            meal_data = {
                                'meal_type': 'daily_plan',
                                'food_items': json.dumps(meal_plan),
                                'total_calories': total_plan_calories,
                                'protein': total_plan_protein,
                                'carbs': total_plan_carbs,
                                'fat': total_plan_fat
                            }
                            db.log_meal(user.id, meal_data)
                            st.success("✅ Meal plan saved to your history!")
                    except Exception as e:
                        st.error(f"Error saving meal plan: {e}")
    
    with tab3:
        st.subheader("🔍 Food Database Search")
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Search for foods (e.g., 'chicken breast', 'quinoa', 'apple')...", 
                                    placeholder="Enter food name or ingredient")
        with col2:
            search_limit = st.selectbox("Results", [10, 20, 50], index=0)
        
        if search_query:
            with st.spinner("Searching food database..."):
                results = recommender.search_foods(search_query, limit=search_limit)
            
            if results:
                st.write(f"Found {len(results)} results for '{search_query}':")
                
                # Create a more detailed results display
                for i, food in enumerate(results):
                    with st.expander(f"{food['description']} - {food['calories_per_100g']:.0f} cal/100g", 
                                expanded=(i < 3)):  # Expand first 3 results
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**Nutritional Info (per 100g):**")
                            st.write(f"🔥 Calories: {food['calories_per_100g']:.1f}")
                            st.write(f"🥩 Protein: {food['protein_per_100g']:.1f}g")
                            st.write(f"🍞 Carbs: {food['carbs_per_100g']:.1f}g")
                            st.write(f"🥑 Fat: {food['fat_per_100g']:.1f}g")
                        
                        with col2:
                            if food.get('brand_name'):
                                st.write(f"**Brand:** {food['brand_name']}")
                            st.write(f"**Match Score:** {food['similarity']:.2f}")
                            st.write(f"**FDC ID:** {food['fdc_id']}")
                            
                            # Macronutrient ratios
                            total_macros = food['protein_per_100g'] + food['carbs_per_100g'] + food['fat_per_100g']
                            if total_macros > 0:
                                protein_pct = (food['protein_per_100g'] / total_macros) * 100
                                carbs_pct = (food['carbs_per_100g'] / total_macros) * 100
                                fat_pct = (food['fat_per_100g'] / total_macros) * 100
                                st.write("**Macro Ratios:**")
                                st.write(f"P: {protein_pct:.0f}% | C: {carbs_pct:.0f}% | F: {fat_pct:.0f}%")
                        
                        with col3:
                            st.write("**Quick Add to Meal:**")
                            portion = st.number_input(f"Portion (g)", min_value=1, value=100, 
                                                    key=f"portion_{food['fdc_id']}")
                            
                            if st.button(f"Add to Meal Log", key=f"add_{food['fdc_id']}"):
                                # Calculate nutrition for portion
                                calories = food['calories_per_100g'] * portion / 100
                                protein = food['protein_per_100g'] * portion / 100
                                carbs = food['carbs_per_100g'] * portion / 100
                                fat = food['fat_per_100g'] * portion / 100
                                
                                # Store in session state for meal logging
                                if 'quick_add_foods' not in st.session_state:
                                    st.session_state.quick_add_foods = []
                                
                                st.session_state.quick_add_foods.append({
                                    'name': food['description'],
                                    'portion_g': portion,
                                    'calories': calories,
                                    'protein': protein,
                                    'carbs': carbs,
                                    'fat': fat
                                })
                                
                                st.success(f"Added {food['description']} to quick meal log!")
            else:
                st.warning("No foods found matching your search. Try different keywords.")
    
    with tab4:
        st.subheader("📝 Log Your Meals")
        
        # Quick add from search results
        if 'quick_add_foods' in st.session_state and st.session_state.quick_add_foods:
            st.write("**Quick Add Foods:**")
            for i, food in enumerate(st.session_state.quick_add_foods):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"{food['name']} - {food['portion_g']}g ({food['calories']:.0f} cal)")
                with col2:
                    meal_type = st.selectbox("Meal", ["breakfast", "lunch", "dinner", "snack"], 
                                        key=f"quick_meal_{i}")
                with col3:
                    if st.button("Log", key=f"quick_log_{i}"):
                        try:
                            with DatabaseManager() as db:
                                meal_data = {
                                    'meal_type': meal_type,
                                    'food_items': json.dumps([food]),
                                    'total_calories': food['calories'],
                                    'protein': food['protein'],
                                    'carbs': food['carbs'],
                                    'fat': food['fat']
                                }
                                db.log_meal(user.id, meal_data)
                                st.success(f"Logged to {meal_type}!")
                                
                                # Remove from quick add list
                                st.session_state.quick_add_foods.pop(i)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error logging meal: {e}")
            
            if st.button("Clear Quick Add List"):
                st.session_state.quick_add_foods = []
                st.rerun()
            
            st.divider()
        
        # Manual meal logging
        with st.form("meal_log_form"):
            st.write("**Manual Meal Entry:**")
            
            col1, col2 = st.columns(2)
            with col1:
                meal_type = st.selectbox("Meal Type", ["breakfast", "lunch", "dinner", "snack"])
                food_description = st.text_input("Food Description", 
                                            placeholder="e.g., 'Grilled chicken breast'")
            
            with col2:
                portion_size = st.number_input("Portion Size (g)", min_value=1, value=100)
                
                # Option for direct nutrition entry
                manual_entry = st.checkbox("Enter nutrition manually")
            
            if manual_entry:
                st.write("**Manual Nutrition Entry:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    manual_calories = st.number_input("Calories", min_value=0.0, value=0.0)
                with col2:
                    manual_protein = st.number_input("Protein (g)", min_value=0.0, value=0.0)
                with col3:
                    manual_carbs = st.number_input("Carbs (g)", min_value=0.0, value=0.0)
                with col4:
                    manual_fat = st.number_input("Fat (g)", min_value=0.0, value=0.0)
            
            submitted = st.form_submit_button("🍽️ Log Meal", type="primary")
            
            if submitted:
                if food_description:
                    if manual_entry:
                        # Use manual nutrition data
                        food_data = {
                            'name': food_description,
                            'portion_g': portion_size,
                            'calories': manual_calories,
                            'protein': manual_protein,
                            'carbs': manual_carbs,
                            'fat': manual_fat
                        }
                        calories = manual_calories
                        protein = manual_protein
                        carbs = manual_carbs
                        fat = manual_fat
                    else:
                        # Search for the food
                        search_results = recommender.search_foods(food_description, limit=1)
                        
                        if search_results:
                            food = search_results[0]
                            
                            # Calculate nutrition for portion
                            calories = food['calories_per_100g'] * portion_size / 100
                            protein = food['protein_per_100g'] * portion_size / 100
                            carbs = food['carbs_per_100g'] * portion_size / 100
                            fat = food['fat_per_100g'] * portion_size / 100
                            
                            food_data = {
                                'name': food['description'],
                                'portion_g': portion_size,
                                'calories': calories,
                                'protein': protein,
                                'carbs': carbs,
                                'fat': fat
                            }
                        else:
                            st.warning("Food not found in database. Please use manual entry or try a different search term.")
                            st.stop()
                    
                    # Log meal to database
                    try:
                        with DatabaseManager() as db:
                            meal_data = {
                                'meal_type': meal_type,
                                'food_items': json.dumps([food_data]),
                                'total_calories': calories,
                                'protein': protein,
                                'carbs': carbs,
                                'fat': fat
                            }
                            db.log_meal(user.id, meal_data)
                            st.success(f"✅ Logged {food_description} - {calories:.0f} calories to {meal_type}!")
                    except Exception as e:
                        st.error(f"Error logging meal: {e}")
                else:
                    st.warning("Please enter a food description.")
        
        # Show today's meals summary
        st.subheader("📊 Today's Meal Summary")
        
        try:
            with DatabaseManager() as db:
                today_meals = db.get_user_meal_history(user.id, days=1)
                
                if today_meals:
                    # Filter for today's meals only
                    today = datetime.now().date()
                    today_meals = [meal for meal in today_meals if hasattr(meal.date, 'date') and meal.date.date() == today]
                    
                    if today_meals:
                        # Calculate totals
                        total_calories = sum(meal.total_calories or 0 for meal in today_meals)
                        total_protein = sum(meal.protein or 0 for meal in today_meals)
                        total_carbs = sum(meal.carbs or 0 for meal in today_meals)
                        total_fat = sum(meal.fat or 0 for meal in today_meals)
                        
                        # Progress towards daily targets
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            def safe_float(val):
                                try:
                                    # SQLAlchemy columns have a .__class__.__name__ of 'InstrumentedAttribute'
                                    if hasattr(val, 'name') and hasattr(val, 'type'):
                                        return 0.0
                                    return float(val)
                                except Exception:
                                    return 0.0
                            cal_val = safe_float(total_calories)
                            daily_val = safe_float(daily_calories)
                            try:
                                cal_progress = cal_val / daily_val if daily_val else 0.0
                            except Exception:
                                cal_progress = 0.0
                            cal_progress = min(max(cal_progress, 0.0), 1.0)
                            st.metric("Calories", f"{cal_val:.0f} / {daily_val:.0f}", 
                                    f"{cal_val - daily_val:+.0f}")
                            st.progress(cal_progress)
                        with col2:
                            protein_progress = min(total_protein / macro_targets['protein_g'], 1.0)
                            st.metric("Protein", f"{total_protein:.1f}g / {macro_targets['protein_g']:.0f}g", 
                                    f"{total_protein - macro_targets['protein_g']:+.1f}g")
                            st.progress(protein_progress)
                        
                        with col3:
                            carbs_progress = min(total_carbs / macro_targets['carbs_g'], 1.0)
                            st.metric("Carbs", f"{total_carbs:.1f}g / {macro_targets['carbs_g']:.0f}g", 
                                    f"{total_carbs - macro_targets['carbs_g']:+.1f}g")
                            st.progress(carbs_progress)
                        
                        with col4:
                            fat_progress = min(total_fat / macro_targets['fat_g'], 1.0)
                            st.metric("Fat", f"{total_fat:.1f}g / {macro_targets['fat_g']:.0f}g", 
                                    f"{total_fat - macro_targets['fat_g']:+.1f}g")
                            st.progress(fat_progress)
                        
                        # Meal breakdown
                        st.write("**Today's Meals:**")
                        for meal in sorted(today_meals, key=lambda x: x.date if isinstance(x.date, datetime) else datetime.now()):
                            time_str = meal.date.strftime('%H:%M')
                            st.write(f"**{time_str} - {meal.meal_type.title()}:** {meal.total_calories:.0f} cal, "
                                f"{meal.protein:.1f}g protein, {meal.carbs:.1f}g carbs, {meal.fat:.1f}g fat")
                    else:
                        st.info("No meals logged today. Start tracking your nutrition!")
                else:
                    st.info("No meals logged today. Start tracking your nutrition!")
        except Exception as e:
            st.error(f"Error loading meal history: {e}")

    # Meal history and analytics (additional tab or section)
    st.divider()
    st.subheader("📈 Nutrition Analytics")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        days_back = st.selectbox("View data for:", [7, 14, 30, 90], index=0, format_func=lambda x: f"Last {x} days")
    with col2:
        if st.button("📊 Show Analytics"):
            try:
                with DatabaseManager() as db:
                    meal_history = db.get_user_meal_history(user.id, days=days_back)
                    
                    if meal_history and len(meal_history) > 1:
                        # Create daily summary for analytics
                        daily_data = {}
                        for meal in meal_history:
                            date_str = meal.date.strftime('%Y-%m-%d')
                            if date_str not in daily_data:
                                daily_data[date_str] = {
                                    'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0, 'meals': 0
                                }
                            
                            daily_data[date_str]['calories'] += meal.total_calories or 0
                            daily_data[date_str]['protein'] += meal.protein or 0
                            daily_data[date_str]['carbs'] += meal.carbs or 0
                            daily_data[date_str]['fat'] += meal.fat or 0
                            daily_data[date_str]['meals'] += 1
                        
                        # Convert to DataFrame for plotting
                        df = pd.DataFrame.from_dict(daily_data, orient='index')
                        df.index = pd.to_datetime(df.index)
                        df = df.sort_index()
                        
                        # Create charts
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_calories = px.line(df, y='calories', title="Daily Calorie Intake",
                                                color_discrete_sequence=['#FF6B6B'])
                            fig_calories.add_hline(y=daily_calories, line_dash="dash", 
                                                annotation_text=f"Target: {daily_calories} cal")
                            fig_calories.update_layout(xaxis_title="Date", yaxis_title="Calories")
                            st.plotly_chart(fig_calories, use_container_width=True)
                        
                        with col2:
                            fig_macros = go.Figure()
                            fig_macros.add_trace(go.Scatter(x=df.index, y=df['protein'], 
                                                        mode='lines', name='Protein (g)', line=dict(color='#FF6B6B')))
                            fig_macros.add_trace(go.Scatter(x=df.index, y=df['carbs'], 
                                                        mode='lines', name='Carbs (g)', line=dict(color='#4ECDC4')))
                            fig_macros.add_trace(go.Scatter(x=df.index, y=df['fat'], 
                                                        mode='lines', name='Fat (g)', line=dict(color='#45B7D1')))
                            
                            fig_macros.update_layout(title="Daily Macronutrient Intake", 
                                                xaxis_title="Date", yaxis_title="Grams")
                            st.plotly_chart(fig_macros, use_container_width=True)
                        
                        # Summary statistics
                        st.subheader("📊 Summary Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            avg_calories = df['calories'].mean()
                            st.metric("Avg Daily Calories", f"{avg_calories:.0f}", 
                                    f"{avg_calories - daily_calories:+.0f} vs target")
                        
                        with col2:
                            avg_protein = df['protein'].mean()
                            st.metric("Avg Daily Protein", f"{avg_protein:.1f}g", 
                                    f"{avg_protein - macro_targets['protein_g']:+.1f}g vs target")
                        
                        with col3:
                            consistency = (df['calories'] > 0).sum() / len(df) * 100
                            st.metric("Logging Consistency", f"{consistency:.0f}%")
                        
                        with col4:
                            avg_meals = df['meals'].mean()
                            st.metric("Avg Meals/Day", f"{avg_meals:.1f}")
                    else:
                        st.info("Not enough meal data for analytics. Keep logging your meals!")
            except Exception as e:
                st.error(f"Error generating analytics: {e}")

if __name__ == "__main__":
    show_meal_recommender()

# Streamlit UI enhancements and final integration
def show_nutrition_dashboard():
    """Enhanced nutrition dashboard with more interactive features"""
    st.header("🎯 Nutrition Dashboard")
    
    # User stats and goals overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Weekly Goal", "85%", "12%", help="Calorie target adherence")
    
    with col2:
        st.metric("Streak", "5 days", "2", help="Consecutive days of logging")
    
    with col3:
        st.metric("Health Score", "82/100", "5", help="Overall nutrition quality")

def add_custom_css():
    """Add custom CSS for better UI/UX"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .food-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f9fafb;
    }
    
    .success-message {
        background: #d1fae5;
        border: 1px solid #10b981;
        color: #065f46;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Enhanced error handling and data validation
def validate_user_input(user_data):
    """Validate user input data for meal recommendations"""
    errors = []
    
    if not user_data.get('age') or user_data['age'] < 16 or user_data['age'] > 100:
        errors.append("Age must be between 16 and 100")
    
    if not user_data.get('weight') or user_data['weight'] < 30 or user_data['weight'] > 300:
        errors.append("Weight must be between 30 and 300 kg")
    
    if not user_data.get('height') or user_data['height'] < 100 or user_data['height'] > 250:
        errors.append("Height must be between 100 and 250 cm")
    
    return errors

# Performance optimization utilities
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_cached_food_data():
    """Load and cache food data for better performance"""
    # This would load the processed USDA data
    processed_file = "data/processed/usda_foods.csv"
    if os.path.exists(processed_file):
        return pd.read_csv(processed_file)
    return None

@st.cache_resource
def initialize_ml_models():
    """Initialize and cache ML models"""
    # Load pre-trained models for food recommendations
    try:
        import joblib
        tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        nutrition_model = joblib.load("models/nutrition_model.pkl")
        return tfidf_vectorizer, nutrition_model
    except:
        return None, None

# Main execution with error handling
if __name__ == "__main__":
    try:
        add_custom_css()
        show_meal_recommender()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page or contact support if the problem persists.")