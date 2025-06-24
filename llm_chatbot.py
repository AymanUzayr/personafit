"""
PersonaFit LLM Chatbot - User Q&A & Daily Motivation (GroqCloud)
"""
import streamlit as st
from groq import Groq  # Updated client import
import random
from datetime import datetime
from database import DatabaseManager
import os 

class FitnessBot:
    def __init__(self):
        self.db = DatabaseManager()
        self.system_prompt = ("""
You are PersonaFit AI, a knowledgeable fitness and nutrition coach. 
You provide helpful, encouraging, and scientifically-backed advice about:
- Exercise techniques and workout planning
- Nutrition and meal planning
- Recovery and injury prevention
- Motivation and goal setting

Keep responses concise but informative. Always prioritize safety and recommend consulting 
healthcare professionals for medical concerns. Be encouraging and supportive.
        """)

        # Initialize GroqCloud client
        if 'groqcloud_api_key' in st.secrets:
            self.client = Groq(api_key= os.getenv('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'))
        else:
            self.client = None

    def get_daily_motivation(self, user_name=None, user_goals=None):
        """Generate daily motivation message"""
        motivational_quotes = [
            "💪 Every workout brings you closer to your goals!",
            "🌟 Progress, not perfection. You've got this!",
            "🔥 Your only competition is who you were yesterday.",
            "💯 Consistency beats intensity. Keep showing up!",
            "⚡ Strong body, strong mind. Let's make today count!",
            "🎯 Small steps daily lead to big changes yearly.",
            "🚀 Challenge yourself - you're stronger than you think!",
            "💎 Diamonds are made under pressure. Shine today!",
            "🏆 Champions are made in training, not in comfort zones.",
            "✨ Your future self is counting on what you do today!"
        ]
        base_quote = random.choice(motivational_quotes)
        return f"Good morning, {user_name}! {base_quote}" if user_name else base_quote

    def get_contextual_motivation(self, user_id):
        """Get personalized motivation based on user's recent activity"""
        try:
            recent_workouts = self.db.get_user_workout_history(user_id, days=7)
            if not recent_workouts:
                return "🌅 Ready to start your fitness journey? Every expert was once a beginner!"
            count = len(recent_workouts)
            avg_diff = sum(w['difficulty_rating'] for w in recent_workouts) / count
            if count >= 5:
                return f"🔥 Amazing consistency! {count} workouts this week. You're unstoppable!"
            if count >= 3:
                return f"💪 Great work this week! {count} workouts down. Keep the momentum!"
            if avg_diff > 7:
                return "🏆 I see you pushing hard! Remember, recovery is just as important as training."
            return "⭐ Every step counts! Ready to make today even better than yesterday?"
        except:
            return self.get_daily_motivation()

    def chat_with_ai(self, user_message, user_id=None, chat_history=None):
        """Chat with GroqCloud AI assistant"""
        if not self.client:
            return "I'm currently unavailable. Please configure the GroqCloud API key to chat with me!"

        # Build context
        context = []
        context.append({"role": "system", "content": f"{self.system_prompt}"})
        if user_id:
            # Fetch user profile from the database
            from database import User
            user_obj = self.db.session.query(User).filter(User.id == user_id).first()
            user_profile = user_obj.__dict__ if user_obj else None
            recent = self.db.get_user_workout_history(user_id, days=7)
            ctx = []
            if user_profile:
                ctx.append(f"User goals: {user_profile.get('fitness_goal', 'general')}")
            if recent:
                ctx.append(f"Recent workouts: {len(recent)} this week.")
            if ctx:
                context.append({"role": "system", "content": ' '.join(ctx)})

        # Add history
        if chat_history:
            context.extend(chat_history[-6:])
        # Add user message
        context.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model="groq-1.0-chat",
                messages=context,
                max_tokens=300,
                temperature=0.7
            )
            content = response.choices[0].message.content if response.choices and response.choices[0].message and hasattr(response.choices[0].message, 'content') else None
            if content is not None:
                return content.strip()
            else:
                return "Sorry, I couldn't generate a response at this time."
        except Exception as e:
            return f"Sorry, I'm having trouble responding right now. Error: {e}"

    def get_quick_tips(self, category):
        tips = {
            'workout': [...], 'nutrition': [...], 'recovery': [...], 'motivation': [...]
        }
        return random.choice(tips.get(category, tips['motivation']))


def render_chatbot_interface():
    """Render Streamlit chatbot interface"""
    st.header("🤖 PersonaFit AI Coach")
    
    bot = FitnessBot()
    
    # Daily motivation section
    st.subheader("🌅 Daily Motivation")
    
    if 'user_id' in st.session_state:
        user_id = st.session_state.user_id
        user_name = st.session_state.get('user_name', 'Champion')
        motivation = bot.get_contextual_motivation(user_id)
    else:
        motivation = bot.get_daily_motivation()
    
    st.success(motivation)
    
    # Quick tips section
    st.subheader("💡 Quick Tips")
    
    tip_category = st.selectbox(
        "Choose a category:",
        ['workout', 'nutrition', 'recovery', 'motivation']
    )
    
    if st.button("Get Tip"):
        tip = bot.get_quick_tips(tip_category)
        st.info(tip)
    
    # Chat interface
    st.subheader("💬 Chat with PersonaFit AI")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message['role'] == 'user':
                st.write(f"**You:** {message['content']}")
            else:
                st.write(f"**PersonaFit AI:** {message['content']}")
    
    # Chat input
    user_input = st.text_input(
        "Ask me anything about fitness, nutrition, or wellness:",
        placeholder="e.g., How many sets should I do for muscle growth?",
        key="chat_input"
    )
    
    if st.button("Send", type="primary") and user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get AI response
        user_id = st.session_state.get('user_id')
        response = bot.chat_with_ai(user_input, user_id, st.session_state.chat_history)
        
        # Add AI response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Clear input and rerun to show new messages
        st.session_state.chat_input = ""
        st.rerun()
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Common questions
    st.subheader("❓ Common Questions")
    
    common_questions = [
        "How often should I work out?",
        "What should I eat before a workout?",
        "How do I build muscle effectively?",
        "What's the best way to lose weight?",
        "How important is sleep for fitness?",
        "Should I do cardio or strength training first?"
    ]
    
    selected_question = st.selectbox("Or choose a common question:", [""] + common_questions)
    
    if selected_question and st.button("Ask This Question"):
        # Add question to chat
        st.session_state.chat_history.append({"role": "user", "content": selected_question})
        
        # Get response
        user_id = st.session_state.get('user_id')
        response = bot.chat_with_ai(selected_question, user_id, st.session_state.chat_history)
        
        # Add response to chat
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        st.rerun()

if __name__ == "__main__":
    render_chatbot_interface()
