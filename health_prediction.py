"""
PersonaFit Health Prediction - Progress Forecast & Recovery/Fatigue Estimation
Enhanced with Wearable Device Dataset Integration
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.model_selection import train_test_split
import sqlite3
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import warnings
import glob
from scipy import stats
from scipy.signal import find_peaks
warnings.filterwarnings('ignore')

class WearableDataProcessor:
    """Process and extract features from wearable device sensor data"""
    
    def __init__(self, dataset_path="wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.0"):
        self.dataset_path = dataset_path
        self.sensor_data = {}
        self.features_df = None
        
    def load_sensor_data(self, activity_type="STRESS", subjects=None):
        """Load sensor data for specified activity type and subjects"""
        if subjects is None:
            # Get all available subjects
            subjects = self._get_available_subjects(activity_type)
        
        sensor_data = {}
        
        for subject in subjects:
            subject_path = os.path.join(self.dataset_path, "Wearable_Dataset", activity_type, subject)
            if not os.path.exists(subject_path):
                continue
                
            subject_data = {}
            
            # Load different sensor files
            sensor_files = {
                'HR': 'HR.csv',
                'EDA': 'EDA.csv', 
                'TEMP': 'TEMP.csv',
                'ACC': 'ACC.csv',
                'IBI': 'IBI.csv'
            }
            
            for sensor_name, filename in sensor_files.items():
                file_path = os.path.join(subject_path, filename)
                if os.path.exists(file_path):
                    try:
                        data = self._load_sensor_file(file_path, sensor_name)
                        subject_data[sensor_name] = data
                    except Exception as e:
                        st.warning(f"Error loading {sensor_name} for {subject}: {e}")
            
            if subject_data:
                sensor_data[subject] = subject_data
        
        self.sensor_data = sensor_data
        return sensor_data
    
    def _load_sensor_file(self, file_path, sensor_type):
        """Load individual sensor file with proper parsing"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # First line is timestamp, second is sampling rate
            start_time = pd.to_datetime(lines[0].strip())
            sampling_rate = float(lines[1].strip())
            
            # Data starts from line 3
            data_lines = lines[2:]
            
            if sensor_type == 'ACC':
                # ACC has 3 columns (x, y, z)
                data = []
                for line in data_lines:
                    if line.strip():
                        values = line.strip().split(',')
                        if len(values) >= 3:
                            data.append([float(v) for v in values[:3]])
                
                df = pd.DataFrame(data, columns=pd.Index(['acc_x', 'acc_y', 'acc_z']))
            else:
                # Other sensors have single column
                data = [float(line.strip()) for line in data_lines if line.strip()]
                df = pd.DataFrame(data, columns=pd.Index([sensor_type.lower()]))
            
            # Add timestamp index
            time_delta = pd.Timedelta(seconds=1/sampling_rate)
            df.index = pd.date_range(start=start_time, periods=len(df), freq=time_delta)
            
            return df
            
        except Exception as e:
            st.error(f"Error parsing {sensor_type} file: {e}")
            return pd.DataFrame()
    
    def _get_available_subjects(self, activity_type):
        """Get list of available subjects for given activity type"""
        activity_path = os.path.join(self.dataset_path, "Wearable_Dataset", activity_type)
        if not os.path.exists(activity_path):
            return []
        
        subjects = [d for d in os.listdir(activity_path) 
                   if os.path.isdir(os.path.join(activity_path, d))]
        return subjects
    
    def extract_features(self, window_size=60, overlap=0.5):
        """Extract features from sensor data using sliding windows"""
        features_list = []
        
        for subject, sensors in self.sensor_data.items():
            if not sensors:
                continue
            
            # Get common time range across all sensors
            common_start = max([sensor.index[0] for sensor in sensors.values() if not sensor.empty])
            common_end = min([sensor.index[-1] for sensor in sensors.values() if not sensor.empty])
            
            # Create time windows
            window_seconds = window_size
            step_seconds = int(window_seconds * (1 - overlap))
            
            current_time = common_start
            while current_time + pd.Timedelta(seconds=window_seconds) <= common_end:
                window_end = current_time + pd.Timedelta(seconds=window_seconds)
                
                window_features = {'subject': subject, 'timestamp': current_time}
                
                # Extract features for each sensor
                for sensor_name, sensor_data in sensors.items():
                    if sensor_data.empty:
                        continue
                    
                    # Get data for current window
                    window_data = sensor_data[
                        (sensor_data.index >= current_time) & 
                        (sensor_data.index < window_end)
                    ]
                    
                    if len(window_data) < 10:  # Minimum data points
                        continue
                    
                    # Extract statistical features
                    sensor_features = self._extract_sensor_features(window_data, sensor_name)
                    window_features.update(sensor_features)
                
                if len(window_features) > 2:  # Has at least subject and timestamp
                    features_list.append(window_features)
                
                current_time += pd.Timedelta(seconds=step_seconds)
        
        if features_list:
            self.features_df = pd.DataFrame(features_list)
            return self.features_df
        else:
            return pd.DataFrame()
    
    def _extract_sensor_features(self, data, sensor_type):
        """Extract statistical features from sensor data"""
        features = {}
        prefix = sensor_type.lower()
        
        if sensor_type == 'HR':
            features.update({
                f'{prefix}_mean': data['hr'].mean(),
                f'{prefix}_std': data['hr'].std(),
                f'{prefix}_min': data['hr'].min(),
                f'{prefix}_max': data['hr'].max(),
                f'{prefix}_range': data['hr'].max() - data['hr'].min(),
                f'{prefix}_median': data['hr'].median(),
                f'{prefix}_skew': data['hr'].skew(),
                f'{prefix}_kurtosis': data['hr'].kurtosis(),
                f'{prefix}_rmssd': self._calculate_rmssd(data['hr']),
                f'{prefix}_hrv': self._calculate_hrv_features(data['hr'])
            })
        
        elif sensor_type == 'EDA':
            features.update({
                f'{prefix}_mean': data['eda'].mean(),
                f'{prefix}_std': data['eda'].std(),
                f'{prefix}_min': data['eda'].min(),
                f'{prefix}_max': data['eda'].max(),
                f'{prefix}_range': data['eda'].max() - data['eda'].min(),
                f'{prefix}_median': data['eda'].median(),
                f'{prefix}_skew': data['eda'].skew(),
                f'{prefix}_kurtosis': data['eda'].kurtosis(),
                f'{prefix}_peaks': len(find_peaks(data['eda'].values, height=data['eda'].mean())[0]),
                f'{prefix}_slope': np.polyfit(range(len(data)), data['eda'].values, 1)[0]
            })
        
        elif sensor_type == 'TEMP':
            features.update({
                f'{prefix}_mean': data['temp'].mean(),
                f'{prefix}_std': data['temp'].std(),
                f'{prefix}_min': data['temp'].min(),
                f'{prefix}_max': data['temp'].max(),
                f'{prefix}_range': data['temp'].max() - data['temp'].min(),
                f'{prefix}_median': data['temp'].median(),
                f'{prefix}_slope': np.polyfit(range(len(data)), data['temp'].values, 1)[0]
            })
        
        elif sensor_type == 'ACC':
            # Calculate magnitude and features for each axis
            acc_magnitude = np.sqrt(data['acc_x']**2 + data['acc_y']**2 + data['acc_z']**2)
            
            features.update({
                f'{prefix}_magnitude_mean': acc_magnitude.mean(),
                f'{prefix}_magnitude_std': acc_magnitude.std(),
                f'{prefix}_magnitude_max': acc_magnitude.max(),
                f'{prefix}_x_mean': data['acc_x'].mean(),
                f'{prefix}_x_std': data['acc_x'].std(),
                f'{prefix}_y_mean': data['acc_y'].mean(),
                f'{prefix}_y_std': data['acc_y'].std(),
                f'{prefix}_z_mean': data['acc_z'].mean(),
                f'{prefix}_z_std': data['acc_z'].std(),
                f'{prefix}_activity_level': self._calculate_activity_level(acc_magnitude)
            })
        
        return features
    
    def _calculate_rmssd(self, hr_data):
        """Calculate RMSSD (Root Mean Square of Successive Differences)"""
        if len(hr_data) < 2:
            return 0
        
        differences = np.diff(hr_data.values)
        return np.sqrt(np.mean(differences**2))
    
    def _calculate_hrv_features(self, hr_data):
        """Calculate basic HRV features"""
        if len(hr_data) < 10:
            return 0
        
        # Simple HRV as coefficient of variation
        return hr_data.std() / hr_data.mean() if hr_data.mean() > 0 else 0
    
    def _calculate_activity_level(self, acc_magnitude):
        """Calculate activity level from accelerometer data"""
        # Normalize and calculate activity level
        normalized = (acc_magnitude - acc_magnitude.mean()) / acc_magnitude.std()
        return np.sum(np.abs(normalized)) / len(normalized)
    
    def load_stress_labels(self):
        """Load stress level labels from CSV files"""
        stress_v1_path = os.path.join(self.dataset_path, "Stress_Level_v1.csv")
        stress_v2_path = os.path.join(self.dataset_path, "Stress_Level_v2.csv")
        
        stress_data = {}
        
        for file_path in [stress_v1_path, stress_v2_path]:
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, index_col=0)
                    # Convert to long format
                    for subject in df.index:
                        for stage in df.columns:
                            if pd.notna(df.loc[subject, stage]):
                                stress_data[f"{subject}_{stage}"] = df.loc[subject, stage]
                except Exception as e:
                    st.warning(f"Error loading stress data from {file_path}: {e}")
        
        return stress_data
    
    def load_subject_info(self):
        """Load subject demographic information"""
        subject_info_path = os.path.join(self.dataset_path, "subject-info.csv")
        
        if os.path.exists(subject_info_path):
            try:
                df = pd.read_csv(subject_info_path, index_col=0)
                return df
            except Exception as e:
                st.warning(f"Error loading subject info: {e}")
        
        return pd.DataFrame()

class HealthPredictor:
    def __init__(self):
        self.progress_model = None
        self.fatigue_model = None
        self.stress_model = None
        self.scaler = StandardScaler()
        self.db_path = "data/personafit.db"
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize wearable data processor
        self.wearable_processor = WearableDataProcessor()
        
        self.load_or_create_models()
    
    def get_user_workouts(self, user_id, days=30):
        """Get user workout history from database"""
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT duration_minutes, difficulty_rating, date as completed_at, workout_type
        FROM workout_logs 
        WHERE user_id = ? AND date >= date('now', '-{} days')
        ORDER BY date DESC
        """.format(days)
        
        df = pd.read_sql_query(query, conn, params=[user_id])
        conn.close()
        return df.to_dict('records') if not df.empty else []
    
    def prepare_wearable_dataset(self, activity_types=["STRESS", "AEROBIC", "ANAEROBIC"]):
        """Prepare comprehensive dataset from wearable device data"""
        st.info("Processing wearable device dataset...")
        
        all_features = []
        stress_labels = self.wearable_processor.load_stress_labels()
        subject_info = self.wearable_processor.load_subject_info()
        
        for activity_type in activity_types:
            st.write(f"Processing {activity_type} data...")
            
            # Load sensor data
            sensor_data = self.wearable_processor.load_sensor_data(activity_type)
            
            if not sensor_data:
                st.warning(f"No data found for {activity_type}")
                continue
            
            # Extract features
            features_df = self.wearable_processor.extract_features()
            
            if not features_df.empty:
                # Add activity type and labels
                features_df['activity_type'] = activity_type
                
                # Add stress labels for stress sessions
                if activity_type == "STRESS":
                    features_df['stress_level'] = features_df['subject'].map(
                        lambda x: stress_labels.get(x, np.nan)
                    )
                
                all_features.append(features_df)
        
        if all_features:
            combined_df = pd.concat(all_features, ignore_index=True)
            
            # Add subject demographics
            if not subject_info.empty:
                combined_df = combined_df.merge(
                    subject_info[['Gender', 'Age', 'Height (cm)', 'Weight (kg)']], 
                    left_on='subject', 
                    right_index=True, 
                    how='left'
                )
            
            # Save processed dataset
            combined_df.to_csv("data/wearable_features.csv", index=False)
            st.success(f"Processed dataset saved with {len(combined_df)} samples and {len(combined_df.columns)} features")
            
            return combined_df
        else:
            st.error("No features extracted from wearable data")
            return pd.DataFrame()
    
    def train_stress_model(self, features_df):
        """Train stress prediction model using wearable data"""
        if features_df.empty or 'stress_level' not in features_df.columns:
            st.warning("No stress data available for training")
            return None
        
        # Prepare features for stress prediction
        stress_data = features_df.dropna(subset=['stress_level'])
        
        if len(stress_data) < 50:
            st.warning("Insufficient stress data for training")
            return None
        
        # Select relevant features
        feature_columns = [col for col in stress_data.columns 
                          if col not in ['subject', 'timestamp', 'activity_type', 'stress_level']]
        
        # Remove columns with too many missing values
        feature_columns = [col for col in feature_columns 
                          if stress_data[col].notna().sum() > len(stress_data) * 0.5]
        
        if len(feature_columns) < 5:
            st.warning("Insufficient features for stress prediction")
            return None
        
        X = stress_data[feature_columns].fillna(0)
        y = stress_data['stress_level']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.stress_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.stress_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.stress_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        st.success(f"Stress model trained! MAE: {mae:.2f}")
        
        # Save model
        joblib.dump(self.stress_model, f"{self.models_dir}/stress_model.pkl")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.stress_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def predict_stress_from_sensors(self, hr_data, eda_data, temp_data, acc_data):
        """Predict stress level from sensor data"""
        if not self.stress_model:
            return None
        
        # Extract features from input sensor data
        features = {}
        
        if hr_data is not None:
            hr_features = self.wearable_processor._extract_sensor_features(hr_data, 'HR')
            features.update(hr_features)
        
        if eda_data is not None:
            eda_features = self.wearable_processor._extract_sensor_features(eda_data, 'EDA')
            features.update(eda_features)
        
        if temp_data is not None:
            temp_features = self.wearable_processor._extract_sensor_features(temp_data, 'TEMP')
            features.update(temp_features)
        
        if acc_data is not None:
            acc_features = self.wearable_processor._extract_sensor_features(acc_data, 'ACC')
            features.update(acc_features)
        
        if not features:
            return None
        
        # Convert to DataFrame and predict
        features_df = pd.DataFrame([features])
        
        # Ensure all expected features are present
        try:
            expected_features = self.stress_model.feature_names_in_  # type: ignore
        except AttributeError:
            # Fallback for older scikit-learn versions
            expected_features = list(features_df.columns)
        
        for feature in expected_features:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        features_df = features_df[expected_features]
        
        stress_prediction = self.stress_model.predict(features_df)[0]
        return max(0, min(10, stress_prediction))
    
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
            
            # Try to load stress model
            try:
                self.stress_model = joblib.load(f"{self.models_dir}/stress_model.pkl")
            except:
                self.stress_model = None
                
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
    
    # Set default user_id if not present (remove authentication requirement)
    if 'user_id' not in st.session_state:
        st.session_state.user_id = 1  # Default user ID
    
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