"""
PersonaFit Authentication & User Management (Prototype Mode Enabled)
"""
import streamlit as st
import secrets
import hashlib
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import base64
from datetime import datetime, timedelta
import re
from database import DatabaseManager, User
from config import PASSWORD_SALT_LENGTH, SESSION_TIMEOUT_HOURS

# Toggle for skipping auth in development
DEV_MODE = True  # Set to False in production

class AuthManager:
    def __init__(self):
        self.db = DatabaseManager()

    def generate_salt(self):
        return secrets.token_hex(PASSWORD_SALT_LENGTH)

    def hash_password(self, password, salt):
        password_bytes = password.encode('utf-8')
        salt_bytes = salt.encode('utf-8')

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt_bytes,
            iterations=100000,
        )
        key = kdf.derive(password_bytes)
        return base64.urlsafe_b64encode(key).decode('utf-8')

    def verify_password(self, password, salt, stored_hash):
        return self.hash_password(password, salt) == stored_hash

    def validate_email(self, email):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def validate_password(self, password):
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        if not re.search(r'[A-Z]', password):
            return False, "Must contain at least one uppercase letter"
        if not re.search(r'[a-z]', password):
            return False, "Must contain at least one lowercase letter"
        if not re.search(r'\d', password):
            return False, "Must contain at least one number"
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Must contain at least one special character"
        return True, "Valid password"

    def register_user(self, username, email, password):
        with DatabaseManager() as db:
            if db.get_user_by_username(username):
                return False, "Username already exists"
            if db.get_user_by_email(email):
                return False, "Email already registered"
            if not self.validate_email(email):
                return False, "Invalid email format"

            is_valid, message = self.validate_password(password)
            if not is_valid:
                return False, message

            salt = self.generate_salt()
            password_hash = self.hash_password(password, salt)
            try:
                db.create_user(username, email, password_hash, salt)
                return True, "User registered successfully"
            except Exception as e:
                return False, f"Registration failed: {str(e)}"

    def login_user(self, username, password):
        with DatabaseManager() as db:
            user = db.get_user_by_username(username)
            if not user:
                return False, "Invalid username or password"
            if self.verify_password(password, user.salt, user.password_hash):
                st.session_state.user_id = user.id
                st.session_state.username = user.username
                st.session_state.login_time = datetime.now()
                st.session_state.authenticated = True
                return True, "Login successful"
            return False, "Invalid username or password"

    def logout_user(self):
        for key in ['user_id', 'username', 'login_time', 'authenticated']:
            st.session_state.pop(key, None)

    def is_authenticated(self):
        if DEV_MODE:
            return True
        if not st.session_state.get('authenticated', False):
            return False
        login_time = st.session_state.get('login_time')
        if login_time:
            if datetime.now() - login_time > timedelta(hours=SESSION_TIMEOUT_HOURS):
                self.logout_user()
                return False
        return True

    def get_current_user(self):
        if DEV_MODE:
            with DatabaseManager() as db:
                return db.get_user_by_username("testuser")
        if not self.is_authenticated():
            return None
        user_id = st.session_state.get('user_id')
        if user_id:
            with DatabaseManager() as db:
                return db.session.query(User).filter(User.id == user_id).first()
        return None

def show_login_page():
    if DEV_MODE:
        st.info("DEV MODE: Login screen is disabled.")
        return
    st.title("\U0001F3CB\ufe0f Welcome to PersonaFit")
    st.markdown("*Your Personal AI Fitness & Nutrition Coach*")
    tab1, tab2 = st.tabs(["Login", "Register"])
    auth_manager = AuthManager()

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if username and password:
                    success, message = auth_manager.login_user(username, password)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please fill in all fields")

    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Choose Username")
            new_email = st.text_input("Email Address")
            new_password = st.text_input("Create Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Register"):
                if all([new_username, new_email, new_password, confirm_password]):
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        success, message = auth_manager.register_user(
                            new_username, new_email, new_password)
                        if success:
                            st.success(message)
                            st.info("Please login with your new account")
                        else:
                            st.error(message)
                else:
                    st.error("Please fill in all fields")

def show_user_menu():
    auth_manager = AuthManager()
    user = auth_manager.get_current_user()
    if user:
        st.sidebar.markdown(f"\U0001F44B Welcome, **{user.username}**!")
        if not DEV_MODE and st.sidebar.button("Logout"):
            auth_manager.logout_user()
            st.rerun()
        st.sidebar.markdown("---")
    return user