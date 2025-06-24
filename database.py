"""
PersonaFit Database Models & Operations
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
import json
from config import DATABASE_URL

# Database Setup
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    salt = Column(String, nullable=False)
    
    # Profile Information
    age = Column(Integer)
    height = Column(Float)  # in cm
    weight = Column(Float)  # in kg
    gender = Column(String)
    activity_level = Column(String)
    fitness_goal = Column(String)
    dietary_preferences = Column(Text)  # JSON string
    allergies = Column(Text)  # JSON string
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_login = Column(DateTime)
    
    # Relationships
    meal_logs = relationship("MealLog", back_populates="user")
    workout_logs = relationship("WorkoutLog", back_populates="user")
    health_metrics = relationship("HealthMetric", back_populates="user")
    chat_history = relationship("ChatHistory", back_populates="user")

class MealLog(Base):
    __tablename__ = "meal_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    date = Column(DateTime, default=func.now())
    meal_type = Column(String)  # breakfast, lunch, dinner, snack
    food_items = Column(Text)  # JSON string of food items
    total_calories = Column(Float)
    protein = Column(Float)
    carbs = Column(Float)
    fat = Column(Float)
    fiber = Column(Float)
    
    user = relationship("User", back_populates="meal_logs")

class WorkoutLog(Base):
    __tablename__ = "workout_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    date = Column(DateTime, default=func.now())
    workout_type = Column(String)  # strength, cardio, flexibility, etc.
    exercises = Column(Text)  # JSON string of exercises
    duration_minutes = Column(Integer)
    calories_burned = Column(Float)
    difficulty_rating = Column(Integer)  # 1-10 scale
    notes = Column(Text)
    
    user = relationship("User", back_populates="workout_logs")

class HealthMetric(Base):
    __tablename__ = "health_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    date = Column(DateTime, default=func.now())
    weight = Column(Float)
    body_fat_percentage = Column(Float)
    muscle_mass = Column(Float)
    resting_heart_rate = Column(Integer)
    sleep_hours = Column(Float)
    stress_level = Column(Integer)  # 1-10 scale
    energy_level = Column(Integer)  # 1-10 scale
    recovery_score = Column(Float)  # 0-100 scale
    
    user = relationship("User", back_populates="health_metrics")

class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=func.now())
    message = Column(Text)
    response = Column(Text)
    message_type = Column(String)  # question, motivation, advice
    
    user = relationship("User", back_populates="chat_history")

class FoodDatabase(Base):
    __tablename__ = "food_database"
    
    id = Column(Integer, primary_key=True, index=True)
    fdc_id = Column(Integer, unique=True)
    description = Column(String)
    food_category = Column(String)
    calories_per_100g = Column(Float)
    protein_per_100g = Column(Float)
    carbs_per_100g = Column(Float)
    fat_per_100g = Column(Float)
    fiber_per_100g = Column(Float)
    sugar_per_100g = Column(Float)
    sodium_per_100g = Column(Float)
    brand_name = Column(String)
    ingredients = Column(Text)

class ExerciseDatabase(Base):
    __tablename__ = "exercise_database"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    category = Column(String)
    muscle_group = Column(String)
    equipment = Column(String)
    difficulty = Column(String)
    instructions = Column(Text)
    calories_per_minute = Column(Float)
    
# Database Operations
class DatabaseManager:
    def __init__(self):
        self.session = SessionLocal()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
    
    def create_user(self, username, email, password_hash, salt):
        """Create a new user"""
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            salt=salt
        )
        self.session.add(user)
        self.session.commit()
        return user
    
    def get_user_by_username(self, username):
        """Get user by username"""
        return self.session.query(User).filter(User.username == username).first()
    
    def get_user_by_email(self, email):
        """Get user by email"""
        return self.session.query(User).filter(User.email == email).first()
    
    def update_user_profile(self, user_id, profile_data):
        """Update user profile information"""
        user = self.session.query(User).filter(User.id == user_id).first()
        if user:
            for key, value in profile_data.items():
                if hasattr(user, key):
                    setattr(user, key, value)
            self.session.commit()
        return user
    
    def log_meal(self, user_id, meal_data):
        """Log a meal for a user"""
        meal_log = MealLog(user_id=user_id, **meal_data)
        self.session.add(meal_log)
        self.session.commit()
        return meal_log
    
    def log_workout(self, user_id, workout_data):
        """Log a workout for a user"""
        workout_log = WorkoutLog(user_id=user_id, **workout_data)
        self.session.add(workout_log)
        self.session.commit()
        return workout_log
    
    def log_health_metric(self, user_id, health_data):
        """Log health metrics for a user"""
        health_metric = HealthMetric(user_id=user_id, **health_data)
        self.session.add(health_metric)
        self.session.commit()
        return health_metric
    
    def get_user_meal_history(self, user_id, days=30):
        """Get user's meal history"""
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        return self.session.query(MealLog).filter(
            MealLog.user_id == user_id,
            MealLog.date >= cutoff_date
        ).order_by(MealLog.date.desc()).all()
    
    def get_user_workout_history(self, user_id, days=30):
        """Get user's workout history"""
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        return self.session.query(WorkoutLog).filter(
            WorkoutLog.user_id == user_id,
            WorkoutLog.date >= cutoff_date
        ).order_by(WorkoutLog.date.desc()).all()
    
    def get_food_by_query(self, query, limit=20):
        """Search food database"""
        return self.session.query(FoodDatabase).filter(
            FoodDatabase.description.contains(query)
        ).limit(limit).all()
    
    def get_exercises_by_criteria(self, muscle_group=None, equipment=None, difficulty=None):
        """Get exercises by criteria"""
        query = self.session.query(ExerciseDatabase)
        if muscle_group:
            query = query.filter(ExerciseDatabase.muscle_group == muscle_group)
        if equipment:
            query = query.filter(ExerciseDatabase.equipment == equipment)
        if difficulty:
            query = query.filter(ExerciseDatabase.difficulty == difficulty)
        return query.all()

# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    create_tables()
    print("Database tables created successfully!")