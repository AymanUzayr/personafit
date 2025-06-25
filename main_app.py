"""
PersonaFit - Main Streamlit Application
"""
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime as dt

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from auth import AuthManager, show_login_page
from meal_recommender import show_meal_recommender
from workout_recommender import render_workout_interface
from health_prediction import render_health_prediction
from llm_chatbot import render_chatbot_interface
from database import DatabaseManager

# Configure Streamlit page
st.set_page_config(
    page_title="PersonaFit - Your AI Fitness Coach",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, immersive design
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #06d6a0;
        --dark-bg: #0f1419;
        --card-bg: #1a1f2e;
    }
    
    /* Hide default Streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #fff, #a8edea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.9;
        margin-bottom: 2rem;
    }
    
    /* Navigation cards */
    .nav-card {
        background: linear-gradient(145deg, #ffffff, #f8f9ff);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid #e5e7eb;
        cursor: pointer;
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .nav-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(99, 102, 241, 0.1);
        border-color: #6366f1;
    }
    
    .nav-card-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .nav-card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
    }
    
    .nav-card-desc {
        color: #6b7280;
        font-size: 0.9rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, #f8fafc, #e2e8f0);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3);
    }
    
    /* Success/Info messages */
    .stSuccess {
        background: linear-gradient(90deg, #06d6a0, #00b894);
        border-radius: 10px;
    }
    
    .stInfo {
        background: linear-gradient(90deg, #74b9ff, #0984e3);
        border-radius: 10px;
    }
    
    /* Animation classes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

def render_hero_section():
    """Render hero section for logged-in users"""
    if 'user_name' in st.session_state:
        user_name = st.session_state.user_name
        st.markdown(f"""
        <div class="hero-section fade-in-up">
            <h1 class="hero-title">Welcome back, {user_name}! 💪</h1>
            <p class="hero-subtitle">Ready to crush your fitness goals today?</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="hero-section fade-in-up">
            <h1 class="hero-title">PersonaFit</h1>
            <p class="hero-subtitle">Your AI-Powered Personal Trainer, Nutritionist & Wellness Coach</p>
        </div>
        """, unsafe_allow_html=True)

def render_navigation_dashboard():
    """Render main navigation dashboard"""
    st.markdown("### 🎯 Choose Your Focus")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="nav-card fade-in-up">
            <div class="nav-card-icon">🍽️</div>
            <div class="nav-card-title">Nutrition Planner</div>
            <div class="nav-card-desc">USDA-powered meal recommendations tailored to your goals</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Open Meal Planner", key="meal_nav", use_container_width=True):
            st.session_state.current_page = 'meals'
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="nav-card fade-in-up">
            <div class="nav-card-icon">🏋️</div>
            <div class="nav-card-title">Workout Builder</div>
            <div class="nav-card-desc">Personalized exercise routines based on your fitness level</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Open Workout Planner", key="workout_nav", use_container_width=True):
            st.session_state.current_page = 'workouts'
            st.rerun()
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        <div class="nav-card fade-in-up">
            <div class="nav-card-icon">📊</div>
            <div class="nav-card-title">Health Analytics</div>
            <div class="nav-card-desc">Progress forecasting and recovery optimization</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("View Analytics", key="analytics_nav", use_container_width=True):
            st.session_state.current_page = 'analytics'
            st.rerun()
    
    with col4:
        st.markdown("""
        <div class="nav-card fade-in-up">
            <div class="nav-card-icon">🤖</div>
            <div class="nav-card-title">AI Coach</div>
            <div class="nav-card-desc">Chat with your personal AI fitness and nutrition expert</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Chat with AI", key="chat_nav", use_container_width=True):
            st.session_state.current_page = 'chat'
            st.rerun()

def render_sidebar():
    """Render sidebar navigation"""
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/6366f1/ffffff?text=PersonaFit", width=200)
        
        # Always show welcome message (no login required)
        st.success(f"👋 Welcome, {st.session_state.get('user_name', 'Demo User')}!")
        
        # Navigation menu
        st.markdown("### 📱 Navigation")
        
        if st.button("🏠 Dashboard", use_container_width=True):
            st.session_state.current_page = 'dashboard'
            st.rerun()
        
        if st.button("🍽️ Nutrition", use_container_width=True):
            st.session_state.current_page = 'meals'
            st.rerun()
        
        if st.button("🏋️ Workouts", use_container_width=True):
            st.session_state.current_page = 'workouts'
            st.rerun()
        
        if st.button("📊 Analytics", use_container_width=True):
            st.session_state.current_page = 'analytics'
            st.rerun()
        
        if st.button("🤖 AI Coach", use_container_width=True):
            st.session_state.current_page = 'chat'
            st.rerun()
        
        st.markdown("---")
        
        # Quick stats
        db = DatabaseManager()
        try:
            user_id = st.session_state.user_id
            recent_workouts = db.get_user_workout_history(user_id, days=7)
            recent_meals = db.get_user_meal_history(user_id, days=7)
            
            st.markdown("### 📈 This Week")
            st.metric("Workouts", len(recent_workouts))
            st.metric("Meals Logged", len(recent_meals))
            
            if recent_workouts:
                avg_difficulty = sum(getattr(w, 'difficulty_rating', 0) or 0 for w in recent_workouts) / len(recent_workouts)
                st.metric("Avg Difficulty", f"{avg_difficulty:.1f}/10")
        except:
            pass
        
        st.markdown("---")
        
        # Reset button instead of logout
        if st.button("🔄 Reset Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def main():
    """Main application function"""
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'dashboard'
    
    # Set default user_id if not present (remove authentication requirement)
    if 'user_id' not in st.session_state:
        st.session_state.user_id = 1  # Default user ID
        st.session_state.user_name = "Demo User"
    
    # Render sidebar
    render_sidebar()
    
    # Render current page
    page = st.session_state.current_page
    
    if page == 'dashboard':
        render_hero_section()
        render_navigation_dashboard()
        
        # Show recent activity
        st.markdown("---")
        st.markdown("### 📱 Recent Activity")
        
        db = DatabaseManager()
        user_id = st.session_state.user_id
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏋️ Recent Workouts")
            recent_workouts = db.get_user_workout_history(user_id, days=5)
            
            if recent_workouts:
                for workout in recent_workouts[:3]:
                    from datetime import datetime as dt
                    date_val = getattr(workout, 'date', None)
                    if date_val is not None and isinstance(date_val, dt):
                        date_str = date_val.strftime('%Y-%m-%d')
                    else:
                        date_str = 'N/A'
                    duration = getattr(workout, 'duration_minutes', 'N/A')
                    difficulty = getattr(workout, 'difficulty_rating', 'N/A')
                    st.write(f"• {date_str} - {duration}min (Difficulty: {difficulty}/10)")
            else:
                st.info("No recent workouts. Start your first workout!")
        
        with col2:
            st.markdown("#### 🍽️ Recent Meals")
            recent_meals = db.get_user_meal_history(user_id, days=5)
            
            if recent_meals:
                for meal in recent_meals[:3]:
                    date_val = getattr(meal, 'date', None)
                    if date_val is not None and isinstance(date_val, dt):
                        date_str = date_val.strftime('%Y-%m-%d')
                    else:
                        date_str = 'N/A'
                    meal_type = getattr(meal, 'meal_type', 'N/A').title()
                    calories = getattr(meal, 'total_calories', 0)
                    st.write(f"• {date_str} - {meal_type}: {calories:.0f} cal")
            else:
                st.info("No recent meals logged. Plan your nutrition!")
    
    elif page == 'meals':
        show_meal_recommender()
    
    elif page == 'workouts':
        render_workout_interface()
    
    elif page == 'analytics':
        render_health_prediction()
    
    elif page == 'chat':
        render_chatbot_interface()

if __name__ == "__main__":
    main()
