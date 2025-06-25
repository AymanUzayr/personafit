"""
PersonaFit Workout Recommender - Exercise Database & Personalization
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from database import DatabaseManager
from config import DATA_DIR, MODELS_DIR
import joblib
import os
import typing

class WorkoutRecommender:
    def __init__(self):
        self.exercise_data: typing.Optional[pd.DataFrame] = None
        self.workout_model = None
        self.label_encoders = {}
        self.db = DatabaseManager()
        self.load_or_create_models()
    
    def download_exercise_data(self):
        """Download exercise database from Valkyrie dataset"""
        exercise_file = DATA_DIR / "exercises" / "exercise_list.csv"
        exercise_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not exercise_file.exists():
            st.info("Downloading exercise database...")
            try:
                url = "https://raw.githubusercontent.com/pannaf/valkyrie/main/final_exercise_list.csv"
                response = requests.get(url)
                response.raise_for_status()
                
                with open(exercise_file, 'wb') as f:
                    f.write(response.content)
                
                st.success("Exercise database downloaded!")
            except Exception as e:
                st.error(f"Error downloading exercise data: {e}")
                return None
        
        return exercise_file
    
    def process_exercise_data(self):
        """Process and clean exercise data"""
        exercise_file = self.download_exercise_data()
        if not exercise_file:
            return None
        
        try:
            df = pd.read_csv(exercise_file)
            df = df.dropna(subset=['Exercise Name']).copy()

            
            # Standardize columns
            column_mapping = {
                    'Exercise Name': 'name',
                    'Exercise Type': 'category',
                    'Target Muscle Groups': 'muscle_group',
                    'Equipment Required': 'equipment',
                    'Exercise Difficulty/Intensity': 'difficulty',
                    'Exercise Form and Technique': 'instructions'
                    }

            
            for old, new in column_mapping.items():
                if old in df.columns:
                    df[new] = df[old]
            
            # Clean data
            required_cols = ['name', 'category', 'muscle_group', 'equipment', 'difficulty']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 'unknown'
                df[col] = df[col].astype(str).str.lower().fillna('unknown')
            
            # Add calorie estimates
            calorie_map = {'cardio': 8, 'strength': 6, 'flexibility': 3, 'plyometrics': 10}
            df['calories_per_minute'] = df['category'].map(calorie_map).fillna(6)  # type: ignore
            
            # Save processed data
            processed_file = DATA_DIR / "processed" / "exercises.csv"
            processed_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(processed_file, index=False)
            
            return df
            
        except Exception as e:
            st.error(f"Error processing exercise data: {e}")
            return None
    
    def load_exercise_database(self):
        """Load exercise database"""
        processed_file = DATA_DIR / "processed" / "exercises.csv"
        return pd.read_csv(processed_file) if processed_file.exists() else self.process_exercise_data()
    
    def train_workout_model(self, exercise_df):
        """Train workout recommendation model"""
        if exercise_df is None or exercise_df.empty:
            return
        
        # Encode categorical variables
        categorical_cols = ['category', 'muscle_group', 'equipment', 'difficulty']
        
        for col in categorical_cols:
            if col in exercise_df.columns:
                le = LabelEncoder()
                exercise_df[f'{col}_encoded'] = le.fit_transform(exercise_df[col])
                self.label_encoders[col] = le
        
        # Create features
        feature_cols = [f'{col}_encoded' for col in categorical_cols if f'{col}_encoded' in exercise_df.columns]
        
        if feature_cols:
            X = exercise_df[feature_cols].values
            y = np.random.randint(0, 2, len(exercise_df))  # Synthetic target
            
            self.workout_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.workout_model.fit(X, y)
            
            # Save model
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.workout_model, MODELS_DIR / "workout_model.pkl")
            joblib.dump(self.label_encoders, MODELS_DIR / "workout_encoders.pkl")
    
    def load_or_create_models(self):
        """Load existing models or create new ones"""
        try:
            self.workout_model = joblib.load(MODELS_DIR / "workout_model.pkl")
            self.label_encoders = joblib.load(MODELS_DIR / "workout_encoders.pkl")
            self.exercise_data = self.load_exercise_database()
        except:
            self.exercise_data = self.load_exercise_database()
            if self.exercise_data is not None:
                self.train_workout_model(self.exercise_data)
    
    def get_exercises_by_criteria(self, muscle_group=None, equipment=None, difficulty=None, category=None, limit=10):
        """
        Get exercises filtered by criteria
        
        Args:
            muscle_group: Target muscle group(s) - can be string or list of strings
            equipment: Available equipment
            difficulty: Difficulty level
            category: Exercise category
            limit: Maximum number of exercises to return
            
        Returns:
            List of exercise dictionaries
        """
        if self.exercise_data is None:
            return []
        assert isinstance(self.exercise_data, pd.DataFrame), "exercise_data must be a DataFrame at this point"
        df = typing.cast(pd.DataFrame, self.exercise_data.copy())
        # Flexible muscle group matching
        if muscle_group:
            if isinstance(muscle_group, list):
                mask = df['muscle_group'].apply(lambda x: any(mg.lower() in str(x).lower() for mg in muscle_group))
                df = df[mask]
            else:
                df = df[df['muscle_group'].astype(str).str.contains(muscle_group.lower(), na=False)]
        # Flexible equipment matching
        if equipment:
            df['equipment'] = df['equipment'].astype(str)
            if equipment.lower() == 'none':
                df = df[
                    df['equipment'].isnull() |  # type: ignore
                    df['equipment'].str.lower().isin(['', 'nan', 'none', 'bodyweight'])  # type: ignore
                ]
            else:
                # Robust matching: check if any equipment in the list matches the user's selection
                df = df[df['equipment'].apply(lambda eq: any(e.strip() == equipment.lower() for e in str(eq).lower().split(',')))]  # type: ignore
        # Difficulty matching
        if difficulty:
            df = df[df['difficulty'].astype(str).str.contains(difficulty.lower(), na=False)]  # type: ignore
        # Category matching
        if category:
            df = df[df['category'].astype(str).str.contains(category.lower(), na=False)]  # type: ignore
        return df.head(limit).to_dict('records')  # type: ignore
    
    def calculate_workout_calories(self, exercises, duration_minutes):
        """Calculate estimated calories burned"""
        total_calories = 0
        for exercise in exercises:
            calories_per_min = exercise.get('calories_per_minute', 6)
            total_calories += calories_per_min * (duration_minutes / len(exercises))
        return int(total_calories)
    
    def generate_workout_plan(self, user_id, target_muscle_groups, equipment_available, difficulty, duration_minutes=30):
        """Generate personalized workout plan"""
        exercises = []
        
        for muscle_group in target_muscle_groups:
            muscle_exercises = self.get_exercises_by_criteria(
                muscle_group=muscle_group,
                equipment=equipment_available,
                difficulty=difficulty,
                limit=3
            )
            exercises.extend(muscle_exercises)
        
        # Add cardio if duration > 20 minutes
        if duration_minutes > 20:
            cardio_exercises = self.get_exercises_by_criteria(
                category='cardio',
                equipment=equipment_available,
                limit=1
            )
            exercises.extend(cardio_exercises)

        # Calculate workout metrics
        estimated_calories = self.calculate_workout_calories(exercises, duration_minutes)
        
        workout_plan = {
            'user_id': user_id,
            'exercises': exercises,
            'duration_minutes': duration_minutes,
            'estimated_calories': estimated_calories,
            'target_muscle_groups': target_muscle_groups,
            'created_at': datetime.now()
        }
        
        return workout_plan
    
    def log_workout_completion(self, user_id, workout_plan, exercise_logs, difficulty_rating, notes=""):
        """Log completed workout"""
        workout_logs = {
            'exercises': json.dumps(exercise_logs),
            'difficulty_rating': difficulty_rating,
            'calories_burned': workout_plan['estimated_calories'],
            'notes': notes,
            'date': datetime.now()
        }
        self.db.log_workout(user_id, workout_logs)
        return True
    
    def get_workout_history(self, user_id, days=30):
        """Get user's workout history"""
        return self.db.get_user_workout_history(user_id, days)
    
    def get_workout_analytics(self, user_id):
        """Generate workout analytics"""
        workouts = self.get_workout_history(user_id, days=90)
        
        if not workouts:
            return None
        
        df = pd.DataFrame([w.__dict__ for w in workouts])
        if '_sa_instance_state' in df.columns:
            df = df.drop(columns=['_sa_instance_state'])
        
        analytics = {
            'total_workouts': len(df),
            'total_duration': df['duration_minutes'].sum(),
            'avg_duration': df['duration_minutes'].mean(),
            'total_calories': df['calories_burned'].sum(),
            'avg_difficulty': df['difficulty_rating'].mean(),
            'workout_frequency': len(df) / 13 if len(df) > 0 else 0,  # per week
            'recent_trend': 'improving' if df['difficulty_rating'].tail(5).mean() > df['difficulty_rating'].head(5).mean() else 'stable'
        }
        
        return analytics

def render_workout_interface():
    """Render Streamlit workout interface"""
    st.header("🏋️ Workout Planner")
    
    # Set default user_id if not present (remove authentication requirement)
    if 'user_id' not in st.session_state:
        st.session_state.user_id = 1  # Default user ID
    
    recommender = WorkoutRecommender()
    user_id = st.session_state.user_id
    
    # Workout planner
    st.subheader("Generate Workout Plan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        muscle_groups = st.multiselect(
            "Target Muscle Groups",
            ['chest', 'back', 'shoulders', 'arms', 'legs', 'core', 'full body'],
            default=['chest', 'arms']
        )
        
        equipment = st.selectbox(
            "Available Equipment",
            ['none', 'dumbbells', 'barbell', 'resistance bands', 'gym']
        )
    
    with col2:
        difficulty = st.selectbox(
            "Difficulty Level",
            ['beginner', 'intermediate', 'advanced']
        )
        
        duration = st.slider("Workout Duration (minutes)", 15, 90, 30)
    
    if st.button("Generate Workout Plan", type="primary"):
        with st.spinner("Creating your personalized workout..."):
            workout_plan = recommender.generate_workout_plan(
                user_id, muscle_groups, equipment, difficulty, duration
            )
            
            st.session_state.current_workout = workout_plan
            
            st.success(f"Workout plan generated! Estimated calories: {workout_plan['estimated_calories']}")
            
            # Display workout plan
            st.subheader("Your Workout Plan")
            
            for i, exercise in enumerate(workout_plan['exercises'], 1):
                with st.expander(f"{i}. {exercise['name'].title()}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Category:** {exercise['category'].title()}")
                        st.write(f"**Muscle Group:** {exercise['muscle_group'].title()}")
                    
                    with col2:
                        st.write(f"**Equipment:** {exercise['equipment'].title()}")
                        st.write(f"**Difficulty:** {exercise['difficulty'].title()}")
                    
                    with col3:
                        st.write(f"**Calories/min:** {exercise['calories_per_minute']}")
                    
                    if 'instructions' in exercise and exercise['instructions']:
                        st.write(f"**Instructions:** {exercise['instructions']}")
    
    # Workout completion
    if 'current_workout' in st.session_state:
        st.subheader("Complete Workout")
        exercise_logs = []
        for i, exercise in enumerate(st.session_state.current_workout['exercises']):
            st.markdown(f"**{exercise['name'].title()}** ({exercise['category'].title()})")
            if exercise['category'].lower() == 'cardio':
                duration = st.number_input(f"Duration (min) for {exercise['name']}", min_value=1, max_value=180, value=10, key=f"duration_{i}")
                exercise_logs.append({
                    "name": exercise['name'],
                    "category": exercise['category'],
                    "duration_minutes": duration
                })
            else:
                col_sets, col_reps, col_weight = st.columns([1, 1, 2])
                with col_sets:
                    sets = st.number_input("Sets", min_value=1, max_value=20, value=3, step=1, key=f"sets_{i}", format="%d")
                with col_reps:
                    reps = st.number_input("Reps", min_value=1, max_value=100, value=10, step=1, key=f"reps_{i}", format="%d")
                with col_weight:
                    weight = st.number_input("Weight (kg)", min_value=0, max_value=500, value=0, step=1, key=f"weight_{i}", format="%d")
                exercise_logs.append({
                    "name": exercise['name'],
                    "category": exercise['category'],
                    "sets": sets,
                    "reps": reps,
                    "weight": weight
                })
        difficulty_rating = st.slider("How challenging was it? (1-10)", 1, 10, 5)
        notes = st.text_area("Workout Notes (optional)")
        if st.button("Log Workout Completion"):
            recommender.log_workout_completion(
                user_id, 
                st.session_state.current_workout,
                exercise_logs,
                difficulty_rating,
                notes
            )
            st.success("Workout logged successfully!")
            del st.session_state.current_workout
            st.rerun()
    
    # Workout analytics
    st.subheader("Workout Analytics")
    
    analytics = recommender.get_workout_analytics(user_id)
    
    if analytics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Workouts", analytics['total_workouts'])
            st.metric("Total Calories Burned", f"{analytics['total_calories']:,}")
        
        with col2:
            st.metric("Avg Duration", f"{analytics['avg_duration']:.0f} min")
            st.metric("image.png", f"{analytics['workout_frequency']:.1f}/week")
        
        with col3:
            st.metric("Avg Difficulty", f"{analytics['avg_difficulty']:.1f}/10")
            st.metric("Trend", analytics['recent_trend'].title())
        
        # Workout history chart
        history = recommender.get_workout_history(user_id, days=30)
        if history:
            df = pd.DataFrame([w.__dict__ for w in history])
            if '_sa_instance_state' in df.columns:
                df = df.drop(columns=['_sa_instance_state'])
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                fig = px.line(df, x='date', y='duration_minutes', 
                             title="Workout Duration Trend",
                             labels={'date': 'Date', 'duration_minutes': 'Duration (min)'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No 'date' field found in workout history. Please check your data.")
                st.write("Available columns:", df.columns.tolist())
                st.dataframe(df)
    
    else:
        st.info("Complete your first workout to see analytics!")

if __name__ == "__main__":
    render_workout_interface()
