import os
import csv
import cv2
import numpy as np
import face_recognition
import pickle
import time
import sqlite3
import hashlib
import secrets
import smtplib
from datetime import datetime, date, time as dtime, timedelta
from flask import Flask, Response, jsonify, request, send_from_directory, session, redirect, url_for
from flask_cors import CORS
from threading import Thread
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from itsdangerous import URLSafeTimedSerializer
from flask import render_template_string # String se HTML render karne ke liye

# Email Configuration
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'email': 'your_email@gmail.com',  # <<< IMPORTANT: Update with your actual Gmail address
    'password': 'your_app_password',  # <<< IMPORTANT: Update with your actual Gmail App Password
    'from_name': 'Face Attendance System'
}

# Lecture slots
LECTURE_SLOTS = [
    {"id": "09:00-09:45", "start": dtime(9, 0),  "end": dtime(9, 45)},
    {"id": "09:45-10:30", "start": dtime(9, 45), "end": dtime(10, 30)},
    {"id": "10:30-11:15", "start": dtime(10, 30), "end": dtime(11, 15)},
    {"id": "11:15-12:00", "start": dtime(11, 15), "end": dtime(12, 0)},
    {"id": "12:00-12:45", "start": dtime(12, 0), "end": dtime(12, 45)},
    {"id": "12:45-13:30", "start": dtime(12, 45), "end": dtime(13, 30)},
    {"id": "13:30-14:15", "start": dtime(13, 30), "end": dtime(14, 15)},
    {"id": "14:15-15:00", "start": dtime(14, 15), "end": dtime(15, 0)},
    {"id": "15:00-15:45", "start": dtime(15, 0), "end": dtime(15, 45)},
    {"id": "15:45-16:30", "start": dtime(15, 45), "end": dtime(16, 30)},
]

def today_dt_for(t: dtime) -> datetime:
    return datetime.combine(date.today(), t)

def find_current_slot(now: datetime | None = None):
    if now is None:
        now = datetime.now()
    now_t = now.time()
    for s in LECTURE_SLOTS:
        if s["start"] <= now_t < s["end"]:
            return s
    return None

def get_slot_by_id(slot_id: str):
    for s in LECTURE_SLOTS:
        if s["id"] == slot_id:
            return s
    return None

# Database functions
def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  role TEXT DEFAULT 'faculty',
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  is_active BOOLEAN DEFAULT 1)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS password_reset_tokens
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  token TEXT UNIQUE NOT NULL,
                  expires_at TIMESTAMP NOT NULL,
                  used BOOLEAN DEFAULT 0,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  faculty_name TEXT,
                  subject TEXT,
                  slot_id TEXT,
                  csv_filename TEXT,
                  started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  ended_at TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    return hash_password(password) == hashed

def send_email(to_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = f"{EMAIL_CONFIG['from_name']} <{EMAIL_CONFIG['email']}>"
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))
        
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['email'], EMAIL_CONFIG['password'])
        server.send_message(msg)
        server.quit()
        print(f"Email sent successfully to {to_email}")
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

# Video Stream Widget
class VideoStreamWidget:
    def __init__(self, src=0):
        print(f"Initializing camera at source: {src}...")
        try:
            if isinstance(src, int):
                self.capture = cv2.VideoCapture(src)
            else:
                self.capture = cv2.VideoCapture(src)

            if not self.capture.isOpened():
                print(f"!!! Error: Could not open camera at source: {src}")
                self.thread = None
                return

            self.status = False
            self.frame = None
            self.stopped = False
            self.thread = Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()
            print("Camera thread started.")
        except Exception as e:
            print(f"VideoStreamWidget init error: {e}")
            self.thread = None

    def update(self):
        while not self.stopped:
            try:
                if self.capture.isOpened():
                    (self.status, self.frame) = self.capture.read()
                time.sleep(.01)
            except Exception as e:
                print(f"Camera update error: {e}")
                break

    def read(self):
        try:
            if hasattr(self, 'frame') and self.frame is not None:
                return self.status, self.frame.copy()
        except Exception as e:
            print(f"Camera read error: {e}")
        return False, None

    def release(self):
        try:
            self.stopped = True
            if self.thread is not None:
                self.thread.join()
            if hasattr(self, 'capture') and self.capture.isOpened():
                self.capture.release()
                print("Camera resource released.")
        except Exception as e:
            print(f"Camera release error: {e}")

# Enhanced Face Attendance System
class FaceAttendanceSystem:
    def __init__(self, encodings_path: str = "encodings.pickle") -> None:
        self.encodings_path = encodings_path
        self.known_face_encodings: list[np.ndarray] = []
        self.known_face_names: list[str] = []
        self._load_known_faces_from_pickle()
        self.session_active = False
        self.expected_start_dt: datetime | None = None
        self.current_slot_id: str = ""
        self.current_faculty = ""
        self.current_subject = ""
        self.csv_filename = ""
        self.marked_attendance: set[str] = set()
        self.camera_widget = None

    def _load_known_faces_from_pickle(self):
        print("Loading known faces from pickle file...")
        try:
            if not os.path.exists(self.encodings_path):
                print(f"WARNING: Encodings file '{self.encodings_path}' not found. Face recognition will not work.")
                self.known_face_encodings = []
                self.known_face_names = []
                return

            with open(self.encodings_path, "rb") as f:
                data = pickle.load(f)
            self.known_face_encodings = data["encodings"]
            self.known_face_names = data["names"]
            print(f"Total faces loaded: {len(self.known_face_names)}")
        except Exception as e:
            print(f"ERROR: Failed to load encodings from '{self.encodings_path}': {e}")
            self.known_face_encodings = []
            self.known_face_names = []

    def start_new_session(self, faculty: str, subject: str, camera_source, slot_id: str | None = None, manual_start_time: str | None = None) -> bool:
        try:
            try:
                capture_source = int(camera_source)
            except (ValueError, TypeError):
                capture_source = camera_source

            self.camera_widget = VideoStreamWidget(src=capture_source)
            if self.camera_widget.thread is None: # Check if camera initialization failed
                print(f"Failed to initialize camera widget for source: {capture_source}")
                return False

            slot = None
            if slot_id:
                slot = get_slot_by_id(slot_id)
            if not slot:
                slot = find_current_slot()

            if manual_start_time:
                try:
                    h, m = map(int, manual_start_time.split(":"))
                    self.expected_start_dt = datetime.combine(date.today(), dtime(hour=h, minute=m))
                except Exception as e:
                    print(f"Error parsing manual start time: {e}. Using current time.")
                    self.expected_start_dt = datetime.now()
            elif slot:
                self.expected_start_dt = today_dt_for(slot["start"])
            else:
                self.expected_start_dt = datetime.now()

            self.current_slot_id = slot["id"] if slot else "NA"
            self.session_active = True
            self.current_faculty, self.current_subject = faculty, subject
            self.marked_attendance.clear()

            today_str = date.today().isoformat()
            safe_subject = "".join(c for c in subject if c.isalnum())
            safe_faculty = "".join(c for c in faculty if c.isalnum())
            safe_slot = "".join(c for c in self.current_slot_id if c.isalnum() or c in ("-", "_"))

            self.csv_filename = f"attendance_{safe_subject}_{safe_faculty}_{safe_slot}_{today_str}.csv"

            if os.path.exists(self.csv_filename):
                print(f"Restoring existing session from: {self.csv_filename}")
                with open(self.csv_filename, 'r', newline='', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    header = next(csv_reader, None)
                    name_idx = 2
                    if header:
                        try:
                            name_idx = header.index("Student Name")
                        except ValueError:
                            name_idx = 2
                    for row in csv_reader:
                        if row and len(row) > name_idx:
                            self.marked_attendance.add(row[name_idx])
                print(f"Restored {len(self.marked_attendance)} attendees.")
            else:
                with open(self.csv_filename, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(["Faculty", "Subject", "Student Name", "Timestamp", "LateMinutes", "Slot"])
                print(f"New session started. Attendance will be saved to: {self.csv_filename}")

            return True
        except Exception as e:
            print(f"Session start error: {e}")
            if self.camera_widget: # Ensure camera is released if an error occurs during setup
                self.camera_widget.release()
                self.camera_widget = None
            return False

    def stop_current_session(self) -> tuple[list, str]:
        final_attendees = []
        filename = ""
        try:
            if self.session_active:
                final_attendees = sorted(list(self.marked_attendance))
                filename = self.csv_filename

            self.session_active = False
            if self.camera_widget:
                self.camera_widget.release()
                self.camera_widget = None
            print("Session stopped successfully.")
        except Exception as e:
            print(f"Session stop error: {e}")
        return final_attendees, filename

    def _mark_attendance(self, name: str):
        try:
            if not self.session_active or name == "Unknown":
                return
            if name in self.marked_attendance:
                return

            self.marked_attendance.add(name)

            now = datetime.now()
            timestamp = now.strftime("%H:%M:%S")
            late_min = 0
            if self.expected_start_dt:
                delta_min = int((now - self.expected_start_dt).total_seconds() // 60)
                late_min = max(0, delta_min)

            with open(self.csv_filename, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([self.current_faculty, self.current_subject, name, timestamp, late_min, self.current_slot_id])

            print(f"✅ ATTENDANCE MARKED: {name} for {self.current_subject} (late {late_min} min, slot {self.current_slot_id})")
        except Exception as e:
            print(f"Mark attendance error: {e}")

# Flask app
app = Flask(__name__)
app.secret_key = 'face-attendance-secret-key-2025' # Keep this secret and strong in production

ts = URLSafeTimedSerializer(app.secret_key)

# More specific CORS setup to avoid browser policy issues
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize everything
init_db()

def get_name_from_roll_number(roll_number):
    """
    Finds a student's full name from their roll number.
    ASSUMPTION: Your face encoding names are in 'Name_RollNumber' format.
    Example: If a face is named 'Vijay_101', this function will return 'Vijay_101'
             if the input roll_number is '101'.
    """
    try:
        for name in face_attendance.known_face_names:
            # Check if the name ends with _RollNumber or is exactly RollNumber
            # This makes it more flexible if names are just roll numbers
            if name.endswith(f"_{roll_number}") or name == roll_number:
                return name
        print(f"No known face name found for roll number: {roll_number}")
        return None
    except Exception as e:
        print(f"Error finding name for roll number {roll_number}: {e}")
        return None

face_attendance = FaceAttendanceSystem()

def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# Routes with embedded HTML (avoiding template issues)
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login')
def login():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Face Attendance</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; display: flex; align-items: center; justify-content: center;
        }
        .login-container {
            background: white; padding: 40px; border-radius: 10px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1); width: 100%; max-width: 400px;
        }
        .logo { text-align: center; margin-bottom: 30px; }
        .logo h1 { color: #333; font-size: 28px; margin-bottom: 10px; }
        .logo p { color: #666; font-size: 14px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; color: #333; font-weight: 500; }
        input[type="text"], input[type="password"] {
            width: 100%; padding: 12px; border: 2px solid #e1e5e9; border-radius: 8px;
            font-size: 16px; transition: border-color 0.3s;
        }
        input:focus { outline: none; border-color: #667eea; }
        .btn {
            width: 100%; padding: 12px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: 600;
            cursor: pointer; transition: transform 0.2s;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .links { text-align: center; margin-top: 20px; }
        .links a { color: #667eea; text-decoration: none; margin: 0 10px; }
        .links a:hover { text-decoration: underline; }
        .alert { padding: 10px; margin-bottom: 15px; border-radius: 5px; display: none; }
        .alert-error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .alert-success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="logo">
            <h1>Face Attendance</h1>
            <p>Sign in to your account</p>
        </div>
        <div id="alert" class="alert"></div>
        <form id="loginForm">
            <div class="form-group">
                <label for="username">Username</label>
                <input type="text" id="username" required>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" required>
            </div>
            <button type="submit" class="btn" id="loginBtn">Sign In</button>
        </form>
        <div class="links">
            <a href="/register">Create Account</a>
            <a href="/forgot-password">Forgot Password?</a>
        </div>
    </div>
    <script>
        const loginBtn = document.getElementById('loginBtn');
        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('username').value.trim();
            const password = document.getElementById('password').value.trim();
            if (!username || !password) {
                showAlert('Please fill in all fields.', 'error');
                return;
            }
            loginBtn.disabled = true;
            loginBtn.textContent = 'Signing In...';
            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                const data = await response.json();
                if (data.status === 'success') {
                    showAlert('Login successful! Redirecting...', 'success');
                    setTimeout(() => { window.location.href = '/dashboard'; }, 1500);
                } else {
                    showAlert(data.message || 'Login failed', 'error');
                }
            } catch (error) {
                showAlert('Login failed. Please try again.', 'error');
            } finally {
                loginBtn.disabled = false;
                loginBtn.textContent = 'Sign In';
            }
        });
        function showAlert(message, type) {
            const alert = document.getElementById('alert');
            alert.className = `alert alert-${type}`;
            alert.textContent = message;
            alert.style.display = 'block';
            setTimeout(() => { alert.style.display = 'none'; }, 5000);
        }
    </script>
</body>
</html>'''

@app.route('/register')
def register():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Register - Face Attendance</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; display: flex; align-items: center; justify-content: center;
        }
        .register-container {
            background: white; padding: 40px; border-radius: 10px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1); width: 100%; max-width: 400px;
        }
        .logo { text-align: center; margin-bottom: 30px; }
        .logo h1 { color: #333; font-size: 28px; margin-bottom: 10px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; color: #333; font-weight: 500; }
        input[type="text"], input[type="email"], input[type="password"] {
            width: 100%; padding: 12px; border: 2px solid #e1e5e9; border-radius: 8px;
            font-size: 16px; transition: border-color 0.3s;
        }
        input:focus { outline: none; border-color: #667eea; }
        .btn {
            width: 100%; padding: 12px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: 600;
            cursor: pointer; transition: transform 0.2s;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .links { text-align: center; margin-top: 20px; }
        .links a { color: #667eea; text-decoration: none; }
        .alert { padding: 10px; margin-bottom: 15px; border-radius: 5px; display: none; }
        .alert-error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .alert-success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    </style>
</head>
<body>
    <div class="register-container">
        <div class="logo">
            <h1>Face Attendance</h1>
            <p>Create your account</p>
        </div>
        <div id="alert" class="alert"></div>
        <form id="registerForm">
            <div class="form-group">
                <label for="username">Username</label>
                <input type="text" id="username" required>
            </div>
            <div class="form-group">
                <label for="email">Email</label>
                <input type="email" id="email" required>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" required>
            </div>
            <button type="submit" class="btn" id="registerBtn">Create Account</button>
        </form>
        <div class="links">
            <a href="/login">Already have an account? Sign In</a>
        </div>
    </div>
    <script>
        const registerBtn = document.getElementById('registerBtn');
        document.getElementById('registerForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('username').value.trim();
            const email = document.getElementById('email').value.trim();
            const password = document.getElementById('password').value.trim();
            if (!username || !email || !password) {
                showAlert('Please fill in all fields.', 'error');
                return;
            }
            registerBtn.disabled = true;
            registerBtn.textContent = 'Creating Account...';
            try {
                const response = await fetch('/api/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, email, password })
                });
                const data = await response.json();
                if (data.status === 'success') {
                    showAlert('Account created successfully! Redirecting to login...', 'success');
                    setTimeout(() => { window.location.href = '/login'; }, 2000);
                } else {
                    showAlert(data.message || 'Registration failed', 'error');
                }
            } catch (error) {
                showAlert('Registration failed. Please try again.', 'error');
            } finally {
                registerBtn.disabled = false;
                registerBtn.textContent = 'Create Account';
            }
        });
        function showAlert(message, type) {
            const alert = document.getElementById('alert');
            alert.className = `alert alert-${type}`;
            alert.textContent = message;
            alert.style.display = 'block';
            setTimeout(() => { alert.style.display = 'none'; }, 5000);
        }
    </script>
</body>
</html>'''

@app.route('/forgot-password')
def forgot_password():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forgot Password - Face Attendance</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; display: flex; align-items: center; justify-content: center;
        }
        .container {
            background: white; padding: 40px; border-radius: 10px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1); width: 100%; max-width: 400px;
        }
        .logo { text-align: center; margin-bottom: 30px; }
        .logo h1 { color: #333; font-size: 28px; margin-bottom: 10px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; color: #333; font-weight: 500; }
        input[type="email"] {
            width: 100%; padding: 12px; border: 2px solid #e1e5e9; border-radius: 8px;
            font-size: 16px; transition: border-color 0.3s;
        }
        input:focus { outline: none; border-color: #667eea; }
        .btn {
            width: 100%; padding: 12px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: 600;
            cursor: pointer; transition: transform 0.2s;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .links { text-align: center; margin-top: 20px; }
        .links a { color: #667eea; text-decoration: none; }
        .alert { padding: 10px; margin-bottom: 15px; border-radius: 5px; display: none; }
        .alert-error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .alert-success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    </style>
</head>
<body>
    <div class="container">
        <div class="logo">
            <h1>Face Attendance</h1>
            <p>Enter your email to reset password</p>
        </div>
        <div id="alert" class="alert"></div>
        <form id="forgotForm">
            <div class="form-group">
                <label for="email">Email Address</label>
                <input type="email" id="email" required>
            </div>
            <button type="submit" class="btn" id="forgotBtn">Send Reset Link</button>
        </form>
        <div class="links">
            <a href="/login">Back to Login</a>
        </div>
    </div>
    <script>
        const forgotBtn = document.getElementById('forgotBtn');
        document.getElementById('forgotForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('email').value.trim();
            if (!email) {
                showAlert('Please enter your email address.', 'error');
                return;
            }
            forgotBtn.disabled = true;
            forgotBtn.textContent = 'Sending...';
            try {
                const response = await fetch('/api/forgot-password', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email })
                });
                const data = await response.json();
                if (data.status === 'success') {
                    showAlert(data.message, 'success');
                    setTimeout(() => { window.location.href = '/login'; }, 3000);
                } else {
                    showAlert(data.message || 'Password reset failed', 'error');
                }
            } catch (error) {
                showAlert('Password reset failed. Please try again.', 'error');
            } finally {
                forgotBtn.disabled = false;
                forgotBtn.textContent = 'Send Reset Link';
            }
        });
        function showAlert(message, type) {
            const alert = document.getElementById('alert');
            alert.className = `alert alert-${type}`;
            alert.textContent = message;
            alert.style.display = 'block';
            setTimeout(() => { alert.style.display = 'none'; }, 5000);
        }
    </script>
</body>
</html>'''

@app.route('/reset-password/<token>')
def reset_password(token):
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reset Password - Face Attendance</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; display: flex; align-items: center; justify-content: center;
        }}
        .container {{
            background: white; padding: 40px; border-radius: 10px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1); width: 100%; max-width: 400px;
        }}
        .logo {{ text-align: center; margin-bottom: 30px; }}
        .logo h1 {{ color: #333; font-size: 28px; margin-bottom: 10px; }}
        .form-group {{ margin-bottom: 20px; }}
        label {{ display: block; margin-bottom: 5px; color: #333; font-weight: 500; }}
        input[type="password"] {{
            width: 100%; padding: 12px; border: 2px solid #e1e5e9; border-radius: 8px;
            font-size: 16px; transition: border-color 0.3s;
        }}
        input:focus {{ outline: none; border-color: #667eea; }}
        .btn {{
            width: 100%; padding: 12px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: 600;
            cursor: pointer; transition: transform 0.2s;
        }}
        .btn:hover {{ transform: translateY(-2px); }}
        .btn:disabled {{ opacity: 0.6; cursor: not-allowed; transform: none; }}
        .links {{ text-align: center; margin-top: 20px; }}
        .links a {{ color: #667eea; text-decoration: none; }}
        .alert {{ padding: 10px; margin-bottom: 15px; border-radius: 5px; display: none; }}
        .alert-error {{ background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
        .alert-success {{ background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="logo">
            <h1>Face Attendance</h1>
            <p>Enter your new password</p>
        </div>
        <div id="alert" class="alert"></div>
        <form id="resetForm">
            <div class="form-group">
                <label for="password">New Password</label>
                <input type="password" id="password" required>
            </div>
            <div class="form-group">
                <label for="confirmPassword">Confirm Password</label>
                <input type="password" id="confirmPassword" required>
            </div>
            <button type="submit" class="btn" id="resetBtn">Reset Password</button>
        </form>
        <div class="links">
            <a href="/login">Back to Login</a>
        </div>
    </div>
    <script>
        const token = "{token}";
        const resetBtn = document.getElementById('resetBtn');
        document.getElementById('resetForm').addEventListener('submit', async (e) => {{
            e.preventDefault();
            const password = document.getElementById('password').value.trim();
            const confirmPassword = document.getElementById('confirmPassword').value.trim();
            if (!password || !confirmPassword) {{
                showAlert('Please fill in both password fields.', 'error');
                return;
            }}
            if (password !== confirmPassword) {{
                showAlert('Passwords do not match.', 'error');
                return;
            }}
            if (password.length < 6) {{
                showAlert('Password must be at least 6 characters long.', 'error');
                return;
            }}
            resetBtn.disabled = true;
            resetBtn.textContent = 'Resetting...';
            try {{
                const response = await fetch('/api/reset-password', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ token, password }})
                }});
                const data = await response.json();
                if (data.status === 'success') {{
                    showAlert('Password reset successful! Redirecting to login...', 'success');
                    setTimeout(() => {{ window.location.href = '/login'; }}, 2000);
                }} else {{
                    showAlert(data.message || 'Password reset failed', 'error');
                }}
            }} catch (error) {{
                showAlert('Password reset failed. Please try again.', 'error');
            }} finally {{
                resetBtn.disabled = false;
                resetBtn.textContent = 'Reset Password';
            }}
        }});
        function showAlert(message, type) {{
            const alert = document.getElementById('alert');
            alert.className = `alert alert-${{type}}`;
            alert.textContent = message;
            alert.style.display = 'block';
            setTimeout(() => {{ alert.style.display = 'none'; }}, 5000);
        }}
    </script>
</body>
</html>'''

@app.route('/dashboard')
@login_required
def dashboard():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard - Face Attendance</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f8fafc; color: #1a202c;
        }
        .header {
            background: white; padding: 1rem 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            display: flex; justify-content: space-between; align-items: center;
        }
        .logo h1 { color: #2d3748; font-size: 1.5rem; }
        .nav { display: flex; gap: 1rem; align-items: center; }
        .nav a {
            text-decoration: none; color: #4a5568; padding: 0.5rem 1rem;
            border-radius: 6px; transition: background-color 0.2s; cursor: pointer;
        }
        .nav a:hover { background: #edf2f7; }
        .nav a.active { background: #667eea; color: white; }
        .container { max-width: 1200px; margin: 2rem auto; padding: 0 2rem; }
        .welcome {
            background: white; padding: 2rem; border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 2rem;
        }
        .welcome h2 { color: #2d3748; margin-bottom: 0.5rem; }
        .welcome p { color: #718096; }
        .cards {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem;
        }
        .card {
            background: white; padding: 1.5rem; border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); transition: transform 0.2s;
        }
        .card:hover { transform: translateY(-2px); }
        .card h3 { color: #2d3748; margin-bottom: 1rem; }
        .card p { color: #718096; margin-bottom: 1rem; }
        .btn {
            display: inline-block; padding: 0.75rem 1.5rem; background: #667eea;
            color: white; text-decoration: none; border-radius: 6px; font-weight: 600;
            transition: background-color 0.2s;
        }
        .btn:hover { background: #5a67d8; }
        .logout-modal {
            display: none; position: fixed; z-index: 1000; left: 0; top: 0;
            width: 100%; height: 100%; background-color: rgba(0,0,0,0.5);
        }
        .modal-content {
            background-color: white; margin: 15% auto; padding: 20px;
            border-radius: 10px; width: 300px; text-align: center;
        }
        .modal-buttons {
            margin-top: 20px; display: flex; gap: 10px; justify-content: center;
        }
        .btn-confirm {
            background: #e53e3e; color: white; border: none; padding: 10px 20px;
            border-radius: 6px; cursor: pointer;
        }
        .btn-cancel {
            background: #718096; color: white; border: none; padding: 10px 20px;
            border-radius: 6px; cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">
            <h1>Face Attendance System</h1>
        </div>
        <nav class="nav">
            <a href="/dashboard" class="active">Dashboard</a>
            <a href="/attendance">Attendance</a>
            <a onclick="showLogoutModal()">Logout</a>
        </nav>
    </div>
    <div class="container">
        <div class="welcome">
            <h2>Welcome to Face Attendance System</h2>
            <p>Manage attendance sessions with face recognition and liveness detection.</p>
        </div>
        <div class="cards">
            <div class="card">
                <h3>Start New Session</h3>
                <p>Begin attendance session with live face recognition.</p>
                <a href="/attendance" class="btn">Start Session</a>
            </div>
            <div class="card">
                <h3>View Records</h3>
                <p>Access attendance records with timestamps.</p>
                <a href="/attendance" class="btn">View Records</a>
            </div>
            <div class="card">
                <h3>System Features</h3>
                <p>• Live face recognition<br>• Auto late calculation<br>• CSV export</p>
                <a href="/attendance" class="btn">Get Started</a>
            </div>
        </div>
    </div>
    <div id="logoutModal" class="logout-modal">
        <div class="modal-content">
            <h3>Confirm Logout</h3>
            <p>Are you sure you want to logout?</p>
            <div class="modal-buttons">
                <button class="btn-confirm" onclick="confirmLogout()">Yes, Logout</button>
                <button class="btn-cancel" onclick="hideLogoutModal()">Cancel</button>
            </div>
        </div>
    </div>
    <script>
        function showLogoutModal() {
            document.getElementById('logoutModal').style.display = 'block';
        }
        function hideLogoutModal() {
            document.getElementById('logoutModal').style.display = 'none';
        }
        async function confirmLogout() {
            try {
                await fetch('/api/logout');
                window.location.href = '/login';
            } catch (error) {
                window.location.href = '/login';
            }
        }
        window.onclick = function(event) {
            const modal = document.getElementById('logoutModal');
            if (event.target === modal) {
                hideLogoutModal();
            }
        }
    </script>
</body>
</html>'''

# Continue with attendance route (complete face recognition system)
@app.route('/attendance')
@login_required
def attendance():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Face Attendance</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      --primary: #2563eb; --bg: #0f172a; --panel: #111827; --muted: #6b7280;
      --ok: #10b981; --warn: #f59e0b; --err: #ef4444; --text: #e5e7eb; --soft: #1f2937;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      background: linear-gradient(180deg, #0b1020 0%, #0f172a 100%); color: var(--text); min-height: 100vh;
    }
    header {
      padding: 16px 20px; border-bottom: 1px solid #1f2937; background: rgba(17, 24, 39, 0.6);
      backdrop-filter: blur(6px); position: sticky; top: 0; z-index: 10;
      display: flex; justify-content: space-between; align-items: center;
    }
    header h1 { margin: 0; font-size: 18px; letter-spacing: 0.3px; display: flex; align-items: center; gap: 10px; }
    header h1 span.logo {
      display: inline-grid; place-items: center; width: 28px; height: 28px;
      background: radial-gradient(circle at 30% 30%, #60a5fa, #2563eb);
      border-radius: 8px; box-shadow: 0 0 24px rgba(37,99,235,0.4); font-weight: 700; color: #fff;
    }
    .nav { display: flex; gap: 1rem; align-items: center; }
    .nav a { color: #9ca3af; text-decoration: none; padding: 8px 12px; border-radius: 6px; cursor: pointer; }
    .nav a:hover { background: rgba(255,255,255,0.1); }
    .wrap { max-width: 1100px; margin: 20px auto; padding: 0 16px 24px; display: grid; grid-template-columns: 380px 1fr; gap: 18px; }
    @media (max-width: 980px) { .wrap { grid-template-columns: 1fr; } }
    .card { background: rgba(17, 24, 39, 0.8); border: 1px solid #1f2937; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.35); overflow: hidden; }
    .card h2 { font-size: 15px; margin: 0; padding: 14px 16px; border-bottom: 1px solid #1f2937; background: linear-gradient(180deg, rgba(31,41,55,0.6), rgba(17,24,39,0.6)); }
    .card .body { padding: 16px; }
    .grid { display: grid; gap: 12px; }
    .row { display: grid; gap: 8px; }
    label { font-size: 12px; color: var(--muted); }
    input[type="text"], select, input[type="time"] {
      width: 100%; background: var(--soft); color: var(--text);
      border: 1px solid #243244; border-radius: 8px; padding: 10px 12px; outline: none;
    }
    input[type="text"]::placeholder { color: #9ca3af; }
    .actions { display: flex; gap: 10px; }
    button { appearance: none; border: 0; cursor: pointer; padding: 10px 14px; border-radius: 8px; font-weight: 600; color: white; }
    .btn-primary { background: var(--primary); }
    .btn-stop { background: var(--err); }
    .btn-muted { background: #374151; }
    .hint { font-size: 12px; color: var(--muted); }
    .status { display: grid; gap: 6px; background: #0b1222; border: 1px dashed #1f2937; border-radius: 10px; padding: 12px; color: #a5b4fc; margin-top: 8px; }
    .video { display: grid; grid-template-rows: auto 1fr; gap: 10px; padding: 16px; }
    .video .screen {
      width: 100%; background: #000; border: 1px solid #1f2937; border-radius: 10px; overflow: hidden;
      min-height: 320px; display: grid; place-items: center;
    }
    .video .screen img { width: 100%; height: auto; display: block; }
    .attendance { display: grid; gap: 10px; }
    .pill {
      display: inline-flex; align-items: center; gap: 8px;
      border: 1px solid #1f2937; border-radius: 999px; padding: 6px 10px; background: #0b1222; color: #93c5fd; font-size: 12px;
    }
    .table { border: 1px solid #1f2937; border-radius: 10px; overflow: hidden; }
    table { width: 100%; border-collapse: collapse; color: #e5e7eb; font-size: 14px; background: rgba(17,24,39,0.6); }
    th, td { border-bottom: 1px solid #1f2937; padding: 10px 12px; text-align: left; }
    th { background: rgba(31,41,55,0.5); font-weight: 600; }
    tr:last-child td { border-bottom: 0; }
    .right { text-align: right; }
    .late { color: #f59e0b; font-weight: 600; }
    .ontime { color: #10b981; font-weight: 600; }
    a.link { color: #93c5fd; text-decoration: none; }
    a.link:hover { text-decoration: underline; }
    .logout-modal {
      display: none; position: fixed; z-index: 1000; left: 0; top: 0;
      width: 100%; height: 100%; background-color: rgba(0,0,0,0.7);
    }
    .modal-content {
      background: var(--panel); margin: 15% auto; padding: 20px; border-radius: 10px;
      width: 300px; text-align: center; border: 1px solid #1f2937;
    }
    .modal-content h3 { color: var(--text); margin-bottom: 10px; }
    .modal-content p { color: var(--muted); margin-bottom: 20px; }
    .modal-buttons { display: flex; gap: 10px; justify-content: center; }
    .btn-confirm {
      background: var(--err); color: white; border: none; padding: 10px 20px;
      border-radius: 6px; cursor: pointer; font-weight: 600;
    }
    .btn-cancel {
      background: var(--muted); color: white; border: none; padding: 10px 20px;
      border-radius: 6px; cursor: pointer; font-weight: 600;
    }
    .btn-confirm:hover { background: #c53030; }
    .btn-cancel:hover { background: #4a5568; }
  </style>
</head>
<body>
  <header>
    <h1><span class="logo">FA</span> Face Attendance</h1>
    <nav class="nav">
      <a href="/dashboard">Dashboard</a>
      <a href="/attendance">Attendance</a>
      <a onclick="showLogoutModal()">Logout</a>
    </nav>
  </header>
  <div class="wrap">
    <div class="card">
      <h2>Session Controls</h2>
      <div class="body">
        <div class="grid">
          <div class="row">
            <label>Faculty</label>
            <input type="text" id="faculty" placeholder="e.g., Rahul" />
          </div>
          <div class="row">
            <label>Subject</label>
            <input type="text" id="subject" placeholder="e.g., Math" />
          </div>
          <div class="row">
            <label>Camera</label>
            <select id="camera"></select>
            <div class="hint">Select USB index or enter Custom URL below.</div>
          </div>
          <div class="row">
            <label>Custom Camera URL (optional)</label>
            <input type="text" id="camera_url" placeholder="rtsp://user:pass@ip:554/stream" />
            <div class="hint">If filled, this overrides the dropdown.</div>
          </div>
          <div class="row">
            <label>Lecture Slot</label>
            <select id="slot"></select>
            <div class="hint">Auto-selected based on server time.</div>
          </div>
          <div class="row">
            <label>Manual Start Time (optional)</label>
            <input type="time" id="manual_time" />
            <div class="hint">If set, "late" is calculated from this HH:MM.</div>
          </div>
          <div class="actions">
            <button class="btn-primary" id="btnStart">Start Session</button>
            <button class="btn-stop" id="btnStop" disabled>Stop Session</button>
            <button class="btn-muted" id="btnRefresh">Refresh Attendance</button>
          </div>
          <div class="status" id="statusBox">
            <div><strong>Status:</strong> Idle</div>
            <div id="statusDetails" class="hint">Fill details, select camera and slot, then Start.</div>
          </div>
        </div>
      </div>
    </div>
    <div class="card">
      <h2>Live Stream & Attendance</h2>
      <div class="video">
        <div class="screen">
          <img id="stream" src="" alt="Video Stream will appear here" />
        </div>
        <div class="attendance">
          <div class="pill">Current CSV: <span id="csvName" style="margin-left:6px; color:#fff;"></span></div>
          <div class="pill">Slot: <span id="slotName" style="margin-left:6px; color:#fff;"></span></div>
          <div class="pill">Expected Start: <span id="expStart" style="margin-left:6px; color:#fff;"></span></div>
          <div id="downloadArea" style="margin-top:6px;"></div>
          <div class="table" id="tableWrap" style="display:none;">
            <table id="attTable">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Time</th>
                  <th class="right">Late (min)</th>
                </tr>
              </thead>
              <tbody id="attBody"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>
  <div id="logoutModal" class="logout-modal">
    <div class="modal-content">
      <h3>Confirm Logout</h3>
      <p>Are you sure you want to logout?</p>
      <div class="modal-buttons">
        <button class="btn-confirm" onclick="confirmLogout()">Yes, Logout</button>
        <button class="btn-cancel" onclick="hideLogoutModal()">Cancel</button>
      </div>
    </div>
  </div>
  <script>
    const $ = (q) => document.querySelector(q);
    const cameraSel = $('#camera'); const slotSel = $('#slot'); const streamImg = $('#stream');
    const statusBox = $('#statusBox'); const statusDetails = $('#statusDetails');
    const btnStart = $('#btnStart'); const btnStop = $('#btnStop'); const btnRefresh = $('#btnRefresh');
    const csvName = $('#csvName'); const slotName = $('#slotName'); const expStart = $('#expStart');
    const downloadArea = $('#downloadArea'); const tableWrap = $('#tableWrap'); const attBody = $('#attBody');
    let sessionActive = false; let currentCSV = ''; let currentSlot = ''; let expectedStart = '';

    function setStatus(msg, detail = '') {
      statusBox.querySelector('div').innerHTML = `<strong>Status:</strong> ${msg}`;
      statusDetails.textContent = detail;
    }

    function showLogoutModal() { document.getElementById('logoutModal').style.display = 'block'; }
    function hideLogoutModal() { document.getElementById('logoutModal').style.display = 'none'; }

    async function confirmLogout() {
      try { await fetch('/api/logout'); window.location.href = '/login'; } 
      catch (error) { window.location.href = '/login'; }
    }

    window.onclick = function(event) {
      const modal = document.getElementById('logoutModal');
      const qrModal = document.getElementById('qrModal');
      if (event.target === modal) { hideLogoutModal(); }
      if (event.target === qrModal) { qrModal.style.display = 'none'; }
    }

    async function loadCameras() {
      cameraSel.innerHTML = '<option value="">Loading…</option>';
      try {
        const res = await fetch('/api/cameras');
        const data = await res.json();
        cameraSel.innerHTML = '';
        (data.cameras || [0]).forEach((idx) => {
          const opt = document.createElement('option');
          opt.value = idx; opt.textContent = `Camera ${idx}`;
          cameraSel.appendChild(opt);
        });
      } catch { cameraSel.innerHTML = '<option value="0">Camera 0</option>'; }
    }

    async function loadSlots() {
      slotSel.innerHTML = '<option value="">Loading…</option>';
      try {
        const res = await fetch('/api/slots');
        const data = await res.json();
        slotSel.innerHTML = '';
        (data.slots || []).forEach((s) => {
          const opt = document.createElement('option');
          opt.value = s.id; opt.textContent = `${s.id}${s.isCurrent ? ' • current' : ''}`;
          if (s.isCurrent) opt.selected = true;
          slotSel.appendChild(opt);
        });
        if (!slotSel.value && data.autoSelected) { slotSel.value = data.autoSelected; }
      } catch { slotSel.innerHTML = '<option value="">No slots</option>'; }
    }

    function startStream() { streamImg.src = `/video_feed?ts=${Date.now()}`; }
    function stopStream() { streamImg.src = ''; }

    async function startSession() {
      const faculty = $('#faculty').value.trim(); const subject = $('#subject').value.trim();
      const url = $('#camera_url').value.trim();
      const camera_source = url !== '' ? url : (cameraSel.value !== '' ? cameraSel.value : 0);
      const slot_id = slotSel.value || null; const manual_time = $('#manual_time').value || null;
      if (!faculty || !subject) { setStatus('Error', 'Faculty and Subject are required.'); return; }
      setStatus('Starting…', 'Initializing camera and session'); btnStart.disabled = true;
      try {
        const res = await fetch('/api/start_session', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ faculty, subject, camera_source, slot_id, manual_start_time: manual_time })
        });
        const data = await res.json();
        if (data.status === 'success') {
          sessionActive = true; currentCSV = data.csv || ''; currentSlot = data.slot_id || ''; expectedStart = data.expected_start || '';
          csvName.textContent = currentCSV || '—'; slotName.textContent = currentSlot || '—'; expStart.textContent = expectedStart || '—';
          setStatus('Running', 'Session active. Face detection is active.');
          btnStop.disabled = false; startStream(); await loadAttendanceDetailed(); updateDownloadLink();
        } else { setStatus('Error', data.message || 'Failed to start session'); btnStart.disabled = false; }
      } catch (error) {
        console.error('Start session error:', error); setStatus('Error', 'Network or server issue starting session'); btnStart.disabled = false;
      }
    }

    async function stopSession() {
      if (!sessionActive) return; setStatus('Stopping…', 'Releasing camera and finalizing CSV'); btnStop.disabled = true;
      try {
        const res = await fetch('/api/stop_session', { method: 'POST' }); const data = await res.json();
        if (data.status === 'success') {
          sessionActive = false; setStatus('Stopped', 'Session closed. You can download the CSV.'); stopStream();
          if (data.filename) { currentCSV = data.filename; csvName.textContent = currentCSV; updateDownloadLink(); }
          await loadAttendanceDetailed();
        } else { setStatus('Error', data.message || 'Failed to stop session'); btnStop.disabled = false; }
      } catch (error) {
        console.error('Stop session error:', error); setStatus('Error', 'Network or server issue stopping session'); btnStop.disabled = false;
      }
      btnStart.disabled = false;
    }

    function updateDownloadLink() {
      downloadArea.innerHTML = '';
      if (currentCSV) {
        const a = document.createElement('a');
        a.href = `/api/download/${encodeURIComponent(currentCSV)}`; a.className = 'link'; a.textContent = 'Download CSV';
        a.setAttribute('download', currentCSV); downloadArea.appendChild(a);
      }
    }

    async function loadAttendanceDetailed() {
      try {
        const res = await fetch('/api/attendance_detailed'); if (!res.ok) throw new Error(); const items = await res.json();
        attBody.innerHTML = ''; if (items.length === 0) { tableWrap.style.display = 'none'; return; }
        tableWrap.style.display = '';
        for (const row of items) {
          const tr = document.createElement('tr');
          const tdName = document.createElement('td'); tdName.textContent = row.name || '';
          const tdTime = document.createElement('td'); tdTime.textContent = row.time || '';
          const tdLate = document.createElement('td'); tdLate.className = 'right ' + ((+row.late > 0) ? 'late' : 'ontime'); tdLate.textContent = (row.late ?? 0);
          tr.appendChild(tdName); tr.appendChild(tdTime); tr.appendChild(tdLate); attBody.appendChild(tr);
        }
      } catch (error) { console.error('Load attendance error:', error); }
    }

    btnStart.addEventListener('click', startSession);
    btnStop.addEventListener('click', stopSession);
    btnRefresh.addEventListener('click', loadAttendanceDetailed);

 

   
    
    // Initial load function
    (async function init() {
      setStatus('Idle', 'Fill details, select camera and slot, then Start.');
      await Promise.all([loadCameras(), loadSlots()]);
      setInterval(loadAttendanceDetailed, 10000); // Auto-refresh attendance list every 10 seconds
    })();
  </script>
  <div id="qrModal" class="logout-modal" style="display:none; align-items:center; justify-content:center;">
    <div class="modal-content" style="width:auto; padding: 2rem;">
        <h3>Scan to Mark Attendance</h3>
        <p>This QR code will expire in 90 seconds.</p>
        <img id="qrCodeImg" src="" alt="QR Code" style="max-width:250px; height:auto;"/>
        <div class="modal-buttons" style="margin-top:1rem;">
            <button class="btn-cancel" onclick="document.getElementById('qrModal').style.display='none'">Close</button>
        </div>
    </div>
</div>
</body>
</html>'''

# All API Routes (complete implementation)
@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        conn = sqlite3.connect('attendance.db')
        c = conn.cursor()
        c.execute('SELECT id, username, email, password_hash FROM users WHERE username = ? AND is_active = 1', (username,))
        user = c.fetchone()
        conn.close()
        
        if user and verify_password(password, user[3]):
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['email'] = user[2]
            print(f"User {username} logged in successfully.")
            return jsonify({'status': 'success', 'message': 'Login successful'})
        else:
            print(f"Failed login attempt for username: {username}")
            return jsonify({'status': 'error', 'message': 'Invalid username or password'}), 401
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'status': 'error', 'message': 'Login failed'}), 500

@app.route('/api/register', methods=['POST'])
def api_register():
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not all([username, email, password]):
            return jsonify({'status': 'error', 'message': 'All fields are required'}), 400
        
        password_hash = hash_password(password)
        
        conn = sqlite3.connect('attendance.db')
        c = conn.cursor()
        c.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                 (username, email, password_hash))
        conn.commit()
        conn.close()
        print(f"User {username} registered successfully.")
        return jsonify({'status': 'success', 'message': 'Registration successful'})
    except sqlite3.IntegrityError:
        print(f"Registration failed: Username '{username}' or email '{email}' already exists.")
        return jsonify({'status': 'error', 'message': 'Username or email already exists'}), 400
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'status': 'error', 'message': 'Registration failed'}), 500

@app.route('/api/forgot-password', methods=['POST'])
def api_forgot_password():
    try:
        data = request.get_json()
        email = data.get('email')
        
        conn = sqlite3.connect('attendance.db')
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE email = ?', (email,))
        user = c.fetchone()
        
        if user:
            token = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(hours=1)
            
            c.execute('INSERT INTO password_reset_tokens (user_id, token, expires_at) VALUES (?, ?, ?)',
                     (user[0], token, expires_at))
            conn.commit()
            
            reset_link = f"{request.url_root}reset-password/{token}"
            email_body = f"""
            <h2>Password Reset Request</h2>
            <p>Click the link below to reset your password:</p>
            <a href="{reset_link}">{reset_link}</a>
            <p>This link will expire in 1 hour.</p>
            """
            
            send_email(email, "Password Reset - Face Attendance System", email_body)
            print(f"Password reset link sent for email: {email}")
        else:
            print(f"Forgot password request for non-existent email: {email}")
        
        conn.close()
        # Always return a generic success message to prevent email enumeration
        return jsonify({'status': 'success', 'message': 'If email exists, reset link has been sent'})
    except Exception as e:
        print(f"Forgot password error: {e}")
        return jsonify({'status': 'error', 'message': 'Password reset failed'}), 500

@app.route('/api/reset-password', methods=['POST'])
def api_reset_password():
    try:
        data = request.get_json()
        token = data.get('token')
        password = data.get('password')
        
        conn = sqlite3.connect('attendance.db')
        c = conn.cursor()
        c.execute('SELECT user_id, expires_at FROM password_reset_tokens WHERE token = ? AND used = 0', (token,))
        token_data = c.fetchone()
        
        if not token_data:
            print(f"Reset password failed: Invalid or used token '{token}'")
            return jsonify({'status': 'error', 'message': 'Invalid or expired token'}), 400
        
        expires_at = datetime.fromisoformat(token_data[1])
        if datetime.now() > expires_at:
            print(f"Reset password failed: Token '{token}' has expired.")
            return jsonify({'status': 'error', 'message': 'Token has expired'}), 400
        
        password_hash = hash_password(password)
        c.execute('UPDATE users SET password_hash = ? WHERE id = ?', (password_hash, token_data[0]))
        c.execute('UPDATE password_reset_tokens SET used = 1 WHERE token = ?', (token,))
        conn.commit()
        conn.close()
        
        print(f"Password reset successful for user ID: {token_data[0]}")
        return jsonify({'status': 'success', 'message': 'Password reset successful'})
    except Exception as e:
        print(f"Reset password error: {e}")
        return jsonify({'status': 'error', 'message': 'Password reset failed'}), 500
    


        return jsonify({
            "status": "success", 
            "image": "data:image/png;base64," + base64.b64encode(img_bytes).decode('utf-8')
        })


@app.route('/api/logout')
def api_logout():
    session.clear()
    print("User logged out.")
    return jsonify({'status': 'success', 'message': 'Logged out successfully'})

@app.route('/api/cameras')
@login_required
def list_cameras():
    try:
        cameras = []
        # Check first 5 cameras, assuming 0 is usually the default webcam
        for index in range(5):  
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cameras.append(index)
                cap.release()
            else:
                cap.release() # Release even if not opened to prevent resource leaks
                # If a camera index fails, subsequent ones might also fail, so we can break
                # However, some systems might have non-contiguous indices, so keeping it to 5 is a balance.
                # For robustness, you might want to check more or handle specific errors.
                # break 
        print(f"Detected cameras: {cameras}")
        return jsonify({"cameras": cameras if cameras else [0]}) # Default to [0] if no cameras found
    except Exception as e:
        print(f"Camera list error: {e}")
        return jsonify({"cameras":[0], "message": "Error detecting cameras, defaulting to Camera 0." })

@app.route('/api/slots')
@login_required
def list_slots():
    try:
        now = datetime.now()
        current = find_current_slot(now)
        resp = []
        for s in LECTURE_SLOTS:
            resp.append({
                "id": s["id"],
                "start": s["start"].strftime("%H:%M"),
                "end": s["end"].strftime("%H:%M"),
                "isCurrent": (current is not None and s["id"] == current["id"])
            })
        print(f"Current slots: {resp}, auto-selected: {current['id'] if current else 'None'}")
        return jsonify({
            "slots": resp,
            "autoSelected": current["id"] if current else None,
            "serverNow": now.strftime("%H:%M")
        })
    except Exception as e:
        print(f"Slots error: {e}")
        return jsonify({"slots": [], "autoSelected": None, "serverNow": datetime.now().strftime("%H:%M"), "message": "Error loading lecture slots."})

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    last_known_locations, last_known_names, frame_counter = [], [], 0
    while True:
        try:
            if not face_attendance.session_active or face_attendance.camera_widget is None or not face_attendance.camera_widget.capture.isOpened():
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Camera Off - Start a Session", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', blank_frame)
                if not ret:
                    time.sleep(0.1)
                    continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.1)
                continue

            success, frame = face_attendance.camera_widget.read()
            if not success or frame is None:
                print("Failed to read frame from camera.")
                time.sleep(0.05)
                continue

            frame_counter += 1
            if frame_counter % 5 == 0: # Process every 5th frame for performance
                last_known_locations, last_known_names = [], []
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Face detection
                locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                encodings = face_recognition.face_encodings(rgb_small_frame, locations)

                for enc, location in zip(encodings, locations):
                    matches = face_recognition.compare_faces(face_attendance.known_face_encodings, enc, tolerance=0.5)
                    name = "Unknown"
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = face_attendance.known_face_names[first_match_index]
                        face_attendance._mark_attendance(name) # Mark attendance

                    last_known_locations.append(location)
                    last_known_names.append(name)

            # Draw bounding boxes and names on the frame
            for (top, right, bottom, left), name in zip(last_known_locations, last_known_names):
                top *= 2; right *= 2; bottom *= 2; left *= 2 # Scale back up locations
                
                if name in face_attendance.marked_attendance:
                    color, status = (0, 255, 0), "PRESENT" # Green for marked attendance
                else:
                    color, status = (255, 0, 0), "UNKNOWN" # Red for unknown or not yet marked
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
                cv2.rectangle(frame, (left, bottom - 60), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, status, (left + 6, bottom - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame to JPG.")
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Frame generation error: {e}")
            time.sleep(0.1) # Prevent busy-waiting on errors

@app.route('/api/start_session', methods=['POST'])
@login_required
def start_session():
    try:
        data = request.get_json()
        faculty = data.get('faculty')
        subject = data.get('subject')
        camera_source = data.get('camera_source', 0)
        slot_id = data.get('slot_id')
        manual_start_time = data.get('manual_start_time')

        if faculty and subject:
            print(f"Attempting to start session for Faculty: {faculty}, Subject: {subject}, Camera: {camera_source}, Slot: {slot_id}, Manual Time: {manual_start_time}")
            success = face_attendance.start_new_session(faculty, subject, camera_source, slot_id, manual_start_time)
            if success:
                print("Session started successfully.")
                return jsonify({
                    "status": "success",
                    "message": "Session started successfully.",
                    "slot_id": face_attendance.current_slot_id,
                    "expected_start": face_attendance.expected_start_dt.strftime("%H:%M") if face_attendance.expected_start_dt else None,
                    "csv": face_attendance.csv_filename
                })
            else:
                print(f"Failed to start session: Camera initialization failed for source {camera_source}.")
                return jsonify({"status": "error", "message": f"Failed to start camera at source {camera_source}. Please check if camera is in use or accessible."}), 500
        print("Failed to start session: Faculty and subject are required.")
        return jsonify({"status": "error", "message": "Faculty and subject are required."}), 400
    except Exception as e:
        print(f"Start session API error: {e}")
        return jsonify({"status": "error", "message": f"Session start failed due to an internal error: {str(e)}"}), 500

@app.route('/api/stop_session', methods=['POST'])
@login_required
def stop_session():
    try:
        final_list, filename = face_attendance.stop_current_session()
        print(f"Session stopped. Final attendees: {len(final_list)}, CSV: {filename}")
        return jsonify({
            "status": "success",
            "message": "Session stopped.",
            "final_attendance": final_list,
            "filename": filename
        })
    except Exception as e:
        print(f"Stop session API error: {e}")
        return jsonify({"status": "error", "message": f"Session stop failed due to an internal error: {str(e)}"}), 500

@app.route('/api/attendance_detailed')
@login_required
def api_attendance_detailed():
    try:
        items = []
        if face_attendance.csv_filename and os.path.exists(face_attendance.csv_filename):
            with open(face_attendance.csv_filename, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    items.append({
                        "faculty": row.get("Faculty"),
                        "subject": row.get("Subject"),
                        "name": row.get("Student Name"),
                        "time": row.get("Timestamp"),
                        "late": int(row.get("LateMinutes") or 0),
                        "slot": row.get("Slot"),
                    })
        # print(f"Returning {len(items)} detailed attendance records.") # Uncomment for verbose logging
        return jsonify(items)
    except Exception as e:
        print(f"Attendance detailed API error: {e}")
        return jsonify([], {"message": f"Error fetching detailed attendance: {str(e)}"})

@app.route('/api/attendance')
@login_required
def api_attendance():
    try:
        # This route is used by the frontend to get the list of marked attendees for the current session.
        # It's not reading from CSV directly, but from the in-memory set.
        # The detailed attendance is handled by api_attendance_detailed.
        return jsonify(list(face_attendance.marked_attendance))
    except Exception as e:
        print(f"Attendance API error: {e}")
        return jsonify([], {"message": f"Error fetching attendance list: {str(e)}"})

@app.route('/api/download/<path:filename>')
@login_required
def download_file(filename):
    try:
        directory = os.getcwd()
        print(f"Attempting to download file: {filename} from {directory}")
        return send_from_directory(directory, filename, as_attachment=True)
    except FileNotFoundError:
        print(f"Download failed: File '{filename}' not found.")
        return jsonify({"status": "error", "message": "File not found."}), 404
    except Exception as e:
        print(f"Download error: {e}")
        return jsonify({"status": "error", "message": "Download failed."}), 500


@app.route('/qr_mark/<token>')
def show_qr_attendance_page(token):
    """Student is redirected here after scanning the QR code."""
    subject_name = "Unknown Session" # Default subject name
    faculty_name = "Unknown Faculty"
    slot_id = "N/A"
    
    # Use a longer max_age for loading the page, as network latency might delay the scan
    # The actual submission will re-validate with a shorter age.
    qr_page_load_max_age = 300 # 5 minutes for page load
    
    try:
        # Token ko validate karein
        token_data = ts.loads(token, salt='qr-attendance-salt', max_age=qr_page_load_max_age)
        session_csv = token_data.get("csv_filename")
        subject_name = token_data.get("subject", "Unknown Session")
        faculty_name = token_data.get("faculty", "Unknown Faculty")
        slot_id = token_data.get("slot_id", "N/A")
        print(f"QR token validated for page load. Session CSV: {session_csv}, Subject: {subject_name}")
    except Exception as e:
        # Agar token expired ya invalid hai
        print(f"QR Token Error (page load): {e}")
        error_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>QR Error</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: sans-serif; display: grid; place-content: center; min-height: 100vh; background: #f8d7da; color: #721c24; text-align: center; padding: 20px; }
                h1 { color: #dc3545; }
                p { margin-top: 10px; }
            </style>
        </head>
        <body>
            <h1>❌ Error</h1>
            <p>This QR code is invalid or has expired. Please ask your faculty for a new one.</p>
        </body>
        </html>
        """
        return render_template_string(error_html), 400
    
    # Student ko roll number enter karne ke liye form dikhayein
    form_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mark Attendance</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: sans-serif; display: grid; place-content: center; min-height: 100vh; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); text-align: center; color: white; }
            .container { background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); width: 90%; max-width: 400px; color: #333; }
            h2 { margin-bottom: 1rem; color: #333; }
            p { margin-bottom: 1rem; color: #555; }
            input { width: 100%; padding: 10px; margin-bottom: 1rem; border: 1px solid #ccc; border-radius: 5px; box-sizing: border-box; }
            button { width: 100%; padding: 10px; border: none; background: #007bff; color: white; border-radius: 5px; cursor: pointer; font-size: 16px; transition: background-color 0.3s; }
            button:hover { background: #0056b3; }
            .message { margin-top: 1rem; padding: 10px; border-radius: 5px; display: none; }
            .success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Mark Your Attendance</h2>
            <p><strong>Faculty:</strong> {{ faculty }}</p>
            <p><strong>Subject:</strong> {{ subject }}</p>
            <p><strong>Slot:</strong> {{ slot }}</p>
            <form id="attendanceForm">
                <input type="text" name="roll_number" id="roll_number" placeholder="Enter Your Roll Number" required>
                <input type="hidden" name="token" value="{{ token }}">
                <button type="submit">Mark My Attendance</button>
            </form>
            <div id="responseMessage" class="message"></div>
        </div>
        <script>
            document.getElementById('attendanceForm').addEventListener('submit', async function(event) {
                event.preventDefault();
                const rollNumber = document.getElementById('roll_number').value.trim();
                const token = this.elements['token'].value;
                const responseMessageDiv = document.getElementById('responseMessage');

                if (!rollNumber) {
                    responseMessageDiv.innerHTML = 'Please enter your Roll Number.';
                    responseMessageDiv.className = 'message error';
                    responseMessageDiv.style.display = 'block';
                    return;
                }

                try {
                    const response = await fetch('/submit_qr_attendance', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded', // Form data
                        },
                        body: new URLSearchParams({
                            'roll_number': rollNumber,
                            'token': token
                        })
                    });
                    
                    // The backend returns HTML string directly, so parse it
                    const htmlResponse = await response.text(); 
                    responseMessageDiv.innerHTML = htmlResponse;
                    responseMessageDiv.classList.remove('hidden');
                    // Determine success/error class based on response status
                    responseMessageDiv.classList.add(response.ok ? 'success' : 'error');
                    
                    // Clear input on success
                    if (response.ok) {
                        document.getElementById('roll_number').value = '';
                    }

                } catch (error) {
                    console.error('Error submitting attendance:', error);
                    responseMessageDiv.innerHTML = '<h1>❌ Submission failed. Network error.</h1><p>Please check your internet connection and try again.</p>';
                    responseMessageDiv.classList.remove('hidden');
                    responseMessageDiv.classList.add('error');
                    responseMessageDiv.style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(form_html, subject=subject_name, faculty=faculty_name, slot=slot_id, token=token)


@app.route('/submit_qr_attendance', methods=['POST'])
def submit_qr_attendance():
    """Handles the form submission from the student's phone."""
    token = request.form.get('token')
    roll_number = request.form.get('roll_number')
    
    if not token or not roll_number:
        print("QR submission failed: Missing token or roll number.")
        return render_template_string("<h1>❌ Submission failed. Missing data.</h1>"), 400

    # Use a shorter max_age for submission to ensure quick expiration
    qr_submission_max_age = 90 # seconds (matches frontend display)
    
    try:
        # Token ko dobara validate karein to prevent misuse
        token_data = ts.loads(token, salt='qr-attendance-salt', max_age=qr_submission_max_age)
        session_csv_filename_from_token = token_data.get("csv_filename")
        subject_from_token = token_data.get("subject")
        print(f"QR submission: Token validated for CSV: {session_csv_filename_from_token}, Subject: {subject_from_token}, Roll: {roll_number}")
    except Exception as e:
        print(f"QR submission failed: Invalid or expired token. Error: {e}")
        return render_template_string("<h1>❌ Submission failed. The QR code has expired or is invalid.</h1><p>Please ask your faculty for a new one.</p>"), 400

    # Critical check: Ensure the session associated with the QR code is still active
    # and matches the currently running session in the FaceAttendanceSystem instance.
    if not face_attendance.session_active or face_attendance.csv_filename != session_csv_filename_from_token:
        print(f"QR submission failed: Session '{session_csv_filename_from_token}' is no longer active or does not match current session '{face_attendance.csv_filename}'.")
        return render_template_string("<h1>❌ Submission failed. The attendance session is no longer active.</h1><p>Please ask your faculty to start a new session.</p>"), 400

    # Roll number se student ka poora naam nikalein
    student_name = get_name_from_roll_number(roll_number)

    if not student_name:
        print(f"QR submission failed: Roll number '{roll_number}' not found in known faces.")
        return render_template_string(f"<h1>❌ Error</h1><p>Roll number '{roll_number}' not found in the system. Please check your ID.</p>"), 400

    # Check karein ki attendance pehle se marked to nahi
    if student_name in face_attendance.marked_attendance:
        print(f"QR submission: Attendance already marked for {student_name}.")
        return render_template_string(f"<h1>✅ Already Marked</h1><p>Hi {student_name}, your attendance is already marked for this session.</p>")

    # Existing function ka use karke attendance mark karein
    face_attendance._mark_attendance(student_name)
    print(f"QR submission: Attendance marked successfully for {student_name}.")
    
    return render_template_string(f"<h1>✅ Success!</h1><p>Hi {student_name}, your attendance has been marked successfully.</p>")

if __name__ == '__main__':
    print(" Starting Complete Face Attendance System...")
    print(" Features: Login, Register, Face Recognition, CSV Export, QR Code Attendance")
    print(" Visit: http://localhost:5000")
    
    # Create test user (idempotent)
    conn = None
    try:
        conn = sqlite3.connect('attendance.db')
        c = conn.cursor()
        # Check if admin user already exists
        c.execute('SELECT id FROM users WHERE username = ?', ('admin',))
        if c.fetchone() is None:
            c.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                     ('admin', 'admin@test.com', hash_password('admin123')))
            conn.commit()
            print(" Test user created: username=admin, password=admin123")
        else:
            print(" Test user 'admin' already exists.")
    except sqlite3.Error as e:
        print(f"Database error during test user creation: {e}")
    except Exception as e:
        print(f"Unexpected error during test user creation: {e}")
    finally:
        if conn:
            conn.close()
    
    # Run the Flask app
    # use_reloader=False is important when using cv2.VideoCapture in a separate thread
    # as reloader would try to re-import and re-initialize camera, leading to errors.
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
