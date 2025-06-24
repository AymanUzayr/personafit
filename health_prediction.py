"""
PersonaFit Health Prediction - Progress Forecast & Recovery/Fatigue Estimation
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import sqlite3
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class HealthPredictor:
    def __init__(self):
        self.progress_model = None
        self.fatigue_model = None
        self.scaler = StandardScaler()
        self.db_path = "data/personafit.db"
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        self.load_or_create_models()
    
    def get_user_workouts(self, user_id, days=30):
        """Get user workout history from database"""
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT duration_minutes, difficulty_rating, completed_at, workout_type
        FROM workouts 
        WHERE user_id = ? AND completed_at >= date('now', '-{} days')
        ORDER BY completed_at DESC
        """.format(days)
        
        df = pd.read_sql_query(query, conn, params=[user_id])
        conn.close()
        return df.to_dict('records') if not df.empty else []
    
    def download_fatigue_data(self):
        """Download and prepare fatigue dataset"""
        data_dir = "data/fatigue"
        os.makedirs(data_dir, exist_ok=True)
        fatigue_file = f"{data_dir}/fatigue_data.csv"
        
        if not os.path.exists(fatigue_file):
            st.info("Preparing fatigue dataset...")
            # Generate synthetic fatigue data based on physiological patterns
            synthetic_data = self.generate_fatigue_data()
            synthetic_data.to_csv(fatigue_file, index=False)
            st.success("Fatigue dataset ready!")
        
        return fatigue_file
    
    def generate_fatigue_data(self):
        """Generate synthetic fatigue data based on training load principles"""
        np.random.seed(42)
        n = 2000
        
        data = {
            'duration': np.random.normal(45, 20, n),
            'intensity': np.random.uniform(2, 9, n),
            'hr_avg': np.random.normal(140, 25, n),
            'sleep_hrs': np.random.normal(7.5, 1.5, n),
            'stress': np.random.uniform(1, 8, n),
            'rest_days': np.random.poisson(1.5, n),
            'prev_fatigue': np.random.uniform(1, 10, n)
        }
        
        df = pd.DataFrame(data)
        
        # Calculate fatigue based on training load theory
        training_load = (df['duration'] * df['intensity']) / 100
        recovery_factor = df['sleep_hrs'] / 8 * (1 - df['stress']/10)
        
        df['fatigue'] = (
            0.4 * training_load +
            0.2 * df['prev_fatigue'] +
            0.15 * df['rest_days'] +
            0.15 * (1 - recovery_factor) * 10 +
            np.random.normal(0, 0.8, n)
        ).clip(1, 10)
        
        df['recovery'] = (10 - df['fatigue'] + np.random.normal(0, 0.5, n)).clip(1, 10)
        
        return df
    
    def train_progress_model(self, user_data):
        """Train progress forecasting model"""
        if len(user_data) < 15:
            return None
        
        df = pd.DataFrame(user_data)
        df['completed_at'] = pd.to_datetime(df['completed_at'])
        df = df.sort_values('completed_at')
        
        # Feature engineering
        df['days_elapsed'] = (df['completed_at'] - df['completed_at'].min()).dt.days
        df['week_day'] = df['completed_at'].dt.dayofweek
        df['difficulty_ma'] = df['difficulty_rating'].rolling(5, min_periods=1).mean()
        df['duration_ma'] = df['duration_minutes'].rolling(5, min_periods=1).mean()
        
        # Target: next workout difficulty
        df['target'] = df['difficulty_rating'].shift(-1)
        df = df.dropna(subset=['target'])
        
        if len(df) < 8:
            return None
        
        features = ['duration_minutes', 'difficulty_rating', 'days_elapsed', 
                   'week_day', 'difficulty_ma', 'duration_ma']
        
        X = df[features].values
        y = df['target'].values
        
        self.progress_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.progress_model.fit(X, y)
        
        joblib.dump(self.progress_model, f"{self.models_dir}/progress_model.pkl")
        return self.progress_model
    
    def train_fatigue_model(self):
        """Train fatigue/recovery model"""
        fatigue_file = self.download_fatigue_data()
        
        try:
            df = pd.read_csv(fatigue_file)
            
            features = ['duration', 'intensity', 'hr_avg', 'sleep_hrs', 
                       'stress', 'rest_days', 'prev_fatigue']
            
            X = df[features].values
            y = df['fatigue'].values
            
            X_scaled = self.scaler.fit_transform(X)
            
            self.fatigue_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.fatigue_model.fit(X_scaled, y)
            
            joblib.dump(self.fatigue_model, f"{self.models_dir}/fatigue_model.pkl")
            joblib.dump(self.scaler, f"{self.models_dir}/scaler.pkl")
            
            return self.fatigue_model
            
        except Exception as e:
            st.error(f"Error training fatigue model: {e}")
            return None
    
    def load_or_create_models(self):
        """Load existing models or create new ones"""
        try:
            self.progress_model = joblib.load(f"{self.models_dir}/progress_model.pkl")
            self.fatigue_model = joblib.load(f"{self.models_dir}/fatigue_model.pkl")
            self.scaler = joblib.load(f"{self.models_dir}/scaler.pkl")
        except:
            self.train_fatigue_model()
    
    def predict_next_difficulty(self, user_id):
        """Predict next workout difficulty"""
        workouts = self.get_user_workouts(user_id, 60)
        
        if len(workouts) >= 15:
            self.train_progress_model(workouts)
        
        if not self.progress_model or len(workouts) < 5:
            return None
        
        df = pd.DataFrame(workouts[-10:])
        df['completed_at'] = pd.to_datetime(df['completed_at'])
        df = df.sort_values('completed_at')
        
        latest = df.iloc[-1]
        
        features = np.array([[
            latest['duration_minutes'],
            latest['difficulty_rating'],
            7,  # days ahead
            datetime.now().weekday(),
            df['difficulty_rating'].tail(5).mean(),
            df['duration_minutes'].tail(5).mean()
        ]])
        
        pred = self.progress_model.predict(features)[0]
        return max(1, min(10, pred))
    
    def predict_fatigue(self, duration, intensity, hr_avg, sleep_hrs, stress, rest_days, prev_fatigue):
        """Predict fatigue and recovery scores"""
        if not self.fatigue_model:
            return None, None
        
        features = np.array([[duration, intensity, hr_avg, sleep_hrs, stress, rest_days, prev_fatigue]])
        features_scaled = self.scaler.transform(features)
        
        fatigue = self.fatigue_model.predict(features_scaled)[0]
        recovery = max(1, min(10, 11 - fatigue))
        
        return max(1, min(10, fatigue)), recovery
    
    def get_recovery_advice(self, fatigue, recovery):
        """Generate recovery recommendations"""
        advice = []
        
        if fatigue > 7:
            advice.extend([
                "🛌 Take a rest day or light activity only",
                "💤 Prioritize 8+ hours sleep",
                "🧘 Try meditation/stretching"
            ])
        elif fatigue > 5:
            advice.extend([
                "⚖️ Moderate workout recommended",
                "💧 Stay hydrated",
                "🚶 Include active recovery"
            ])
        else:
            advice.extend([
                "💪 Ready for challenging workout!",
                "🎯 Good time to push limits",
                "⚡ Energy levels optimal"
            ])
        
        if recovery < 4:
            advice.extend([
                "🔋 Focus on recovery nutrition",
                "❄️ Consider cold therapy",
                "💆 Light massage recommended"
            ])
        
        return advice
    
    def get_progress_forecast(self, user_id):
        """Generate progress forecast"""
        workouts = self.get_user_workouts(user_id, 30)
        if len(workouts) < 5:
            return None
        
        df = pd.DataFrame(workouts)
        df['difficulty_rating'] = pd.to_numeric(df['difficulty_rating'])
        
        recent_avg = df['difficulty_rating'].head(7).mean()
        overall_avg = df['difficulty_rating'].mean()
        
        if recent_avg > overall_avg + 0.3:
            trend = 'improving'
        elif recent_avg < overall_avg - 0.3:
            trend = 'declining'
        else:
            trend = 'stable'
        
        next_difficulty = self.predict_next_difficulty(user_id)
        
        return {
            'trend': trend,
            'recent_avg': recent_avg,
            'predicted_next': next_difficulty,
            'recommendation': self.get_progress_advice(trend, recent_avg, next_difficulty)
        }
    
    def get_progress_advice(self, trend, recent_avg, next_difficulty):
        """Get progress recommendation"""
        if trend == 'improving' and recent_avg > 6:
            return "Excellent progress! Consider adding exercise variety."
        elif trend == 'declining':
            return "Focus on recovery. Quality over quantity."
        elif next_difficulty and next_difficulty > recent_avg + 0.5:
            return "Ready to level up! Push yourself next session."
        else:
            return "Maintain consistency. Steady progress is key."

def render_health_prediction():
    """Main Streamlit interface"""
    st.header("📊 Health Analytics & Predictions")
    
    if 'user_id' not in st.session_state:
        st.warning("Please log in to access health predictions.")
        return
    
    predictor = HealthPredictor()
    user_id = st.session_state.user_id
    
    tab1, tab2 = st.tabs(["🎯 Progress Forecast", "🔋 Recovery & Fatigue"])
    
    with tab1:
        st.subheader("Progress Forecast")
        
        forecast = predictor.get_progress_forecast(user_id)
        
        if forecast:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trend_color = {'improving': '🟢', 'stable': '🟡', 'declining': '🔴'}
                st.metric("Trend", f"{trend_color[forecast['trend']]} {forecast['trend'].title()}")
            
            with col2:
                st.metric("Recent Avg", f"{forecast['recent_avg']:.1f}/10")
            
            with col3:
                if forecast['predicted_next']:
                    st.metric("Next Predicted", f"{forecast['predicted_next']:.1f}/10")
                else:
                    st.metric("Next Predicted", "N/A")
            
            st.info(f"💡 **Recommendation:** {forecast['recommendation']}")
            
            # Progress visualization
            workouts = predictor.get_user_workouts(user_id, 30)
            if workouts:
                df = pd.DataFrame(workouts)
                df['completed_at'] = pd.to_datetime(df['completed_at'])
                df['difficulty_rating'] = pd.to_numeric(df['difficulty_rating'])
                
                fig = px.line(df.sort_values('completed_at'), 
                             x='completed_at', y='difficulty_rating',
                             title="30-Day Progress Trend",
                             markers=True)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Complete at least 5 workouts to see progress forecasting.")
    
    with tab2:
        st.subheader("Recovery & Fatigue Assessment")
        
        st.write("Enter your current status for personalized recovery recommendations:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            duration = st.slider("Recent Workout Duration (min)", 0, 120, 45)
            intensity = st.slider("Workout Intensity (1-10)", 1, 10, 6)
            hr_avg = st.slider("Average Heart Rate", 60, 200, 140)
            sleep_hrs = st.slider("Sleep Last Night (hours)", 4, 12, 8)
        
        with col2:
            stress = st.slider("Current Stress Level (1-10)", 1, 10, 4)
            rest_days = st.slider("Days Since Last Rest", 0, 7, 1)
            prev_fatigue = st.slider("Previous Fatigue Level (1-10)", 1, 10, 5)
        
        if st.button("Assess Recovery Status", type="primary"):
            fatigue, recovery = predictor.predict_fatigue(
                duration, intensity, hr_avg, sleep_hrs, stress, rest_days, prev_fatigue
            )
            
            if fatigue and recovery:
                st.subheader("Assessment Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Fatigue Score", f"{fatigue:.1f}/10", 
                             delta=f"{'High' if fatigue > 6 else 'Moderate' if fatigue > 4 else 'Low'}")
                
                with col2:
                    st.metric("Recovery Score", f"{recovery:.1f}/10",
                             delta=f"{'Good' if recovery > 6 else 'Fair' if recovery > 4 else 'Poor'}")
                
                # Recommendations
                advice = predictor.get_recovery_advice(fatigue, recovery)
                st.subheader("Recommendations")
                for tip in advice:
                    st.write(f"• {tip}")
                
                # Gauge charts
                fig = go.Figure()
                
                fig.add_trace(go.Indicator(
                    mode = "gauge+number",
                    value = fatigue,
                    domain = {'x': [0, 0.5], 'y': [0, 1]},
                    title = {'text': "Fatigue Level"},
                    gauge = {
                        'axis': {'range': [None, 10]},
                        'bar': {'color': "red"},
                        'steps': [
                            {'range': [0, 3], 'color': "lightgray"},
                            {'range': [3, 7], 'color': "yellow"},
                            {'range': [7, 10], 'color': "orange"}],
                        'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 8}}))
                
                fig.add_trace(go.Indicator(
                    mode = "gauge+number",
                    value = recovery,
                    domain = {'x': [0.5, 1], 'y': [0, 1]},
                    title = {'text': "Recovery Score"},
                    gauge = {
                        'axis': {'range': [None, 10]},
                        'bar': {'color': "green"},
                        'steps': [
                            {'range': [0, 4], 'color': "lightgray"},
                            {'range': [4, 7], 'color': "yellow"},
                            {'range': [7, 10], 'color': "lightgreen"}],
                        'threshold': {'line': {'color': "green", 'width': 4},
                                    'thickness': 0.75, 'value': 8}}))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    render_health_prediction()