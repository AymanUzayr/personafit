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
        """Initialize the fitness chatbot"""
        self.db = DatabaseManager()
        
        # System prompt for the AI
        self.system_prompt = ("""
You are PersonaFit, a friendly and knowledgeable fitness AI coach. Your role is to:

- Provide evidence-based fitness and nutrition advice
- Offer personalized workout and meal recommendations
- Help users stay motivated and track their progress
- Answer questions about exercise form, nutrition, and wellness
- Encourage healthy habits and sustainable lifestyle changes

Key areas of expertise:
- Exercise programming and workout design
- Nutrition fundamentals and meal planning
- Recovery and injury prevention
- Goal setting and progress tracking
- Motivation and mindset

Keep responses concise but informative. Always prioritize safety and recommend consulting 
healthcare professionals for medical concerns. Be encouraging and supportive.
        """)

        # Initialize GroqCloud client (optional)
        self.client = None
        try:
            if 'groqcloud_api_key' in st.secrets:
                self.client = Groq(api_key=st.secrets['groqcloud_api_key'])
        except:
            # If secrets are not available, continue without API
            pass

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
        """Chat with GroqCloud AI assistant or provide fallback responses"""
        if not self.client:
            # Provide intelligent fallback responses
            return self.get_fallback_response(user_message)

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
                model="llama-3.3-70b-versatile",
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

    def get_fallback_response(self, user_message):
        """Provide intelligent fallback responses when API is not available"""
        message_lower = user_message.lower()
        
        # Workout-related responses
        if any(word in message_lower for word in ['workout', 'exercise', 'training', 'gym']):
            if 'how often' in message_lower:
                return "For optimal results, aim for 3-5 workouts per week. Beginners can start with 3 days, while more experienced individuals can handle 4-5 days. Remember to include rest days for recovery!"
            elif 'sets' in message_lower or 'reps' in message_lower:
                return "For muscle growth: 3-4 sets of 8-12 reps. For strength: 3-5 sets of 1-6 reps. For endurance: 2-3 sets of 15+ reps. Always prioritize proper form over weight!"
            elif 'cardio' in message_lower:
                return "Cardio is great for heart health! Aim for 150 minutes of moderate cardio or 75 minutes of vigorous cardio per week. You can split this into 30-minute sessions 5 days a week."
            else:
                return "Great question about workouts! A balanced routine should include strength training, cardio, and flexibility work. Start with compound movements like squats, deadlifts, and push-ups. Remember to warm up properly and listen to your body!"
        
        # Nutrition-related responses
        elif any(word in message_lower for word in ['eat', 'food', 'nutrition', 'diet', 'meal']):
            if 'before workout' in message_lower:
                return "Eat a light meal 2-3 hours before working out, or a small snack 30-60 minutes before. Focus on carbs for energy and some protein. Examples: banana with peanut butter, Greek yogurt with berries, or a protein smoothie."
            elif 'protein' in message_lower:
                return "Protein is essential for muscle repair! Aim for 0.8-1.2g of protein per pound of body weight daily. Good sources include lean meats, fish, eggs, dairy, legumes, and plant-based proteins."
            elif 'weight loss' in message_lower:
                return "Sustainable weight loss comes from a calorie deficit (burning more than you consume) combined with regular exercise. Focus on whole foods, adequate protein, and don't cut calories too drastically. Aim for 1-2 pounds per week."
            else:
                return "Nutrition is key to your fitness journey! Focus on whole foods, plenty of vegetables, lean proteins, and healthy fats. Stay hydrated and try to eat balanced meals throughout the day. Remember, consistency beats perfection!"
        
        # Recovery-related responses
        elif any(word in message_lower for word in ['recovery', 'rest', 'sleep', 'sore']):
            if 'sleep' in message_lower:
                return "Sleep is crucial for recovery and muscle growth! Aim for 7-9 hours of quality sleep per night. Your body repairs and builds muscle during deep sleep, so prioritize good sleep hygiene."
            elif 'sore' in message_lower:
                return "Muscle soreness is normal, especially when starting or increasing intensity. Gentle stretching, light activity, and proper nutrition can help. If soreness lasts more than 3-4 days, consider taking an extra rest day."
            else:
                return "Recovery is just as important as training! Include rest days in your routine, prioritize sleep, stay hydrated, and consider activities like stretching, foam rolling, or light walking on rest days."
        
        # Motivation-related responses
        elif any(word in message_lower for word in ['motivation', 'motivated', 'tired', 'hard']):
            return "Remember why you started! Progress takes time, and every workout counts. Focus on consistency over perfection. Celebrate small wins, and remember that showing up is half the battle. You're stronger than you think! 💪"
        
        # General fitness questions
        elif any(word in message_lower for word in ['goal', 'plan', 'routine', 'program']):
            return "Setting clear, achievable goals is important! Start with specific, measurable goals like 'work out 3 times per week' or 'run a 5K in 3 months.' Break big goals into smaller milestones and track your progress!"
        
        # Default response
        else:
            return "Thanks for your question! I'm here to help with fitness, nutrition, and wellness advice. Feel free to ask about workouts, meal planning, recovery, or motivation. Remember, consistency and patience are key to long-term success! 🌟"

    # def get_quick_tips(self, category):
    #     tips = {
    #         'workout': [...], 'nutrition': [...], 'recovery': [...], 'motivation': [...]
    #     }
    #     return random.choice(tips.get(category, tips['motivation']))

    def generate_next_workout(self, user_id, workout_history):
        """
        Generate a personalized next workout suggestion using the LLM.
        """
        prompt = (
            "Based on the user's recent workout history, suggest a specific next workout. "
            "Include workout type, focus, intensity, and 3-5 exercises with sets/reps or duration. "
            "Be concise and motivating. Here is the recent workout log:\n"
            f"{workout_history}\n"
            "Format the response as HTML for display in a card."
        )
        if self.client:
            response = self.chat_with_ai(prompt, user_id)
            return response
        else:
            # Fallback: static example
            return (
                "<b>Strength</b> Upper body with emphasis on Back "
                "<span class='pf-badge'>Moderate to high</span>"
                "<div style='margin-top:1rem;font-weight:600;'>Suggested Exercises:</div>"
                "<ol class='pf-list'>"
                "<li>Deadlifts - 4 sets of 8 reps</li>"
                "<li>Bench Press - 3 sets of 10 reps</li>"
                "<li>Pull-ups - 3 sets to failure</li>"
                "</ol>"
                "<div class='pf-card-footer'>📅 Tomorrow &nbsp; ⚡ 60 min</div>"
            )

def render_chatbot_interface():
    """Render Streamlit chatbot interface"""
    # --- Fix input clearing bug: check flag before any widgets ---
    if st.session_state.get("clear_input", False):
        st.session_state.chat_input = ""
        st.session_state.clear_input = False
        st.rerun()

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
    
    # # Quick tips section
    # st.subheader("💡 Quick Tips")
    
    # tip_category = st.selectbox(
    #     "Choose a category:",
    #     ['workout', 'nutrition', 'recovery', 'motivation']
    # )
    
    # if st.button("Get Tip"):
    #     tip = bot.get_quick_tips(tip_category)
    #     st.info(tip)
    
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
        # Set a flag to clear input and rerun
        st.session_state.clear_input = True
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
