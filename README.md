# PersonaFit

**PersonaFit** is an AI-powered fitness, nutrition, and wellness platform that provides personalized workout and meal recommendations, health analytics, and an interactive AI coach—all in a modern Streamlit web app.

---

## 🚀 Features
- **Personalized Workout Planner:**
  - Generates custom routines based on your goals, equipment, and fitness level
  - Supports both duration-based (cardio) and reps/sets-based (strength/flexibility) logging
- **Meal Recommender:**
  - Suggests meals tailored to your dietary preferences and goals
  - Integrates with USDA food databases
- **Health Analytics:**
  - Tracks workout history, calories burned, and progress trends
  - Visualizes your fitness journey
- **AI Coach:**
  - Chatbot for motivation, Q&A, and smart advice
- **Wearable Integration:**
  - (Planned) Support for importing data from fitness trackers

---

## 🛠️ Setup Instructions

### 1. **Clone the Repository**
```sh
git clone https://github.com/AymanUzayr/personafit.git
cd personafit
```

### 2. **Create a Virtual Environment**
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. **Install Dependencies**
```sh
pip install -r requirements.txt
```

### 4. **Set Up the Database**
```sh
python database.py  # Creates tables if not present
```

### 5. **Run the Application**
```sh
streamlit run main_app.py
```

---

## 💡 Use Case
PersonaFit is designed for anyone who wants a smart, all-in-one fitness companion:
- **Beginners**: Get started with safe, effective routines and meal plans.
- **Enthusiasts**: Track progress, optimize workouts, and get AI-powered insights.
- **Coaches/Nutritionists**: Use as a digital assistant for client planning and analytics.
- **Researchers/Developers**: Extend the platform for new fitness, nutrition, or AI features.

---

## 🔭 Future Scope
- **Wearable Device Integration:** Import and analyze data from smartwatches and fitness trackers.
- **Mobile App:** Native mobile experience for on-the-go tracking and notifications.
- **Social & Community Features:** Share progress, join challenges, and connect with friends.
- **Advanced AI Coaching:** Deeper personalization using LLMs and user feedback.
- **Automated Progress Forecasting:** Predict plateaus, suggest deloads, and optimize recovery.
- **Voice Assistant Integration:** Hands-free logging and coaching.

---

## 🧑‍💻 Expert Opinion & Recommendations
- **Modular Design:** The codebase is organized for easy extension—add new models, analytics, or UI modules as needed.
- **Data Privacy:** All user data is stored locally by default. For production, consider secure cloud storage and authentication.
- **Customization:** The app is highly customizable—tailor meal/workout logic, add new data sources, or integrate with external APIs.
- **Open Source:** Contributions are welcome! Please open issues or pull requests for new features, bug fixes, or improvements.

---

## 📬 Feedback & Contributions
- **Found a bug?** Open an issue on GitHub.
- **Want a new feature?** Submit a feature request or pull request.
- **Questions?** Contact the maintainer via GitHub.

---

**PersonaFit**: Your AI-powered partner for a healthier, stronger, and smarter you! 