Driver Drowsiness Detection & Alert System 

A real-time computer vision–based driver drowsiness detection system that monitors facial cues to identify fatigue and proactively alerts the driver and emergency contacts.
The system combines eye closure, yawning, and head posture analysis, supports real-time alerts via Telegram, and provides post-drive analytics for deeper insights.

📌 Problem Statement

Driver drowsiness is a major cause of road accidents and often goes unnoticed until it becomes dangerous. Existing solutions are either expensive, intrusive, or rely on a single indicator, leading to false positives.
This project addresses that gap by building a low-cost, camera-based software solution that:
•	Detects early signs of fatigue
•	Reduces false alarms using temporal smoothing
•	Sends real-time alerts with context (image, graph, location)
•	Logs data for post-drive analysis

🎯 Key Features

•	 Eye Aspect Ratio (EAR) for eye-closure detection
•	 Mouth Aspect Ratio (MAR) for yawning detection
•	 Head-drop & tilt detection with personalized baseline calibration
•	 Temporal smoothing using consecutive-frame logic
•	 Local alarm system
•	 Telegram alerts with snapshots, graphs & location
•	 Real-time visualization (EAR / MAR / head tilt)
•	 CSV-based logging & automatic data cleaning
•	 Post-drive analytics & session summary
•	 System Architecture (High Level)

🧩 Project Modules Explained

1️⃣ Video Capture & Face Detection
•	Uses OpenCV to capture webcam frames
•	dlib’s frontal face detector identifies faces
•	Facial landmarks extracted using a 68-point landmark model

2️⃣ Eye Aspect Ratio (EAR) Module
•	Measures vertical vs horizontal eye distances
•	Detects prolonged eye closure
•	Uses consecutive-frame thresholding to avoid noise
•	EAR < threshold for N frames → Drowsiness

3️⃣ Mouth Aspect Ratio (MAR) Module
•	Detects yawning using mouth landmarks
•	Triggers events only after sustained mouth opening

4️⃣ Head Drop & Tilt Detection
•	Computes chin-to-eye-center displacement
•	Normalized by face height
•	Uses a baseline calibration phase for personalization
•	Detects nodding-off behavior

5️⃣ Temporal Smoothing Logic
•	Prevents false positives by requiring:
•	Eye closure for multiple consecutive frames
•	Sustained yawning
•	Repeated head-drop frames

6️⃣ Alert System
When drowsiness is detected:
•	🔊 Local alarm is triggered
•	📸 Face snapshot is captured
•	📊 EAR trend graph is generated
•	📍 GPS location is fetched
•	🤖 Telegram message + images are sent

7️⃣ Logging & Data Cleaning
•	Logs EAR, MAR, head tilt, and events once per second
•	Automatically cleans malformed or duplicate rows
•	Ensures timestamps and numeric consistency

8️⃣ Real-time Visualization
Live Matplotlib plots for:
•	EAR
•	MAR
•	Head tilt
Helps monitor system behavior during execution

9️⃣ Post-drive Analytics
•	After the session ends:
•	Generates stacked time-series graphs
•	Computes event frequency per minute
•	Sends a session summary via Telegram


⚙️ Installation & Setup
 Prerequisites
•	Python 3.8+
•	Webcam
•	Internet connection (for Telegram & GPS)

📦 Install Dependencies
•	pip install opencv-python dlib imutils numpy pandas matplotlib scipy pygame geocoder requests
•	⚠️ Installing dlib may require CMake and Visual Studio Build Tools (Windows) or build-essential (Linux).
•	📥 Download Facial Landmark Model

Download:
•	shape_predictor_68_face_landmarks.dat
•	Place it in the project directory.

🔐 Telegram Configuration
Step 1: Create a Telegram Bot
•	Use @BotFather
•	Get your BOT_TOKEN
Step 2: Get Chat ID
•	Message your bot
•	Use a chat ID fetcher or Telegram API
Step 3: Configure in Code
•	BOT_TOKEN = "YOUR_BOT_TOKEN"
•	CHAT_ID = "YOUR_CHAT_ID"

▶️ How to Run
•	python drowsysmooth_driver.py
•	Press q to exit
•	Closing the app triggers post-drive analysis automatically

📊 Outputs & Alerts
Sent to Telegram
•	🚨 Alert message
•	📸 Driver snapshot
•	📊 EAR trend graph
•	📈 Session summary graphs

Stored Locally
•	CSV logs
•	Images & graphs

🚀 Future Improvements
•	Replace dlib with MediaPipe / CNN-based models
•	Add fatigue scoring using ML
•	Improve head-pose estimation (pitch/yaw/roll)
•	Mobile or embedded deployment
•	Cloud-based dashboard

🧠 Learning Outcomes
•	Real-time computer vision systems
•	Handling noisy data & false positives
•	System design & modular thinking
•	Integrating alerts, analytics, and visualization
•	Building software with real-world impact

📜 License
•	This project is for educational and research purposes.
•	Not intended for direct medical or automotive safety certification use.
