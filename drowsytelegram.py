import cv2
import dlib
import time
import csv
import requests
from scipy.spatial import distance as dist
from imutils import face_utils
from datetime import datetime
from pygame import mixer

# --- Initialize sound ---
mixer.init()
mixer.music.load("alarm.wav")  # use any alert sound file

# --- Telegram Config ---
BOT_TOKEN = ""
CHAT_ID = ""

def send_telegram_alert():
    message = "⚠️ Drowsiness detected! Please take a break."
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
        print("[TELEGRAM] Alert sent successfully!")
    except Exception as e:
        print("[TELEGRAM ERROR]:", e)

# --- EAR Calculation ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20

COUNTER = 0
ALARM_ON = False

# --- Setup CSV Logging ---
csv_file = open("drowsiness_log.csv", "a", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Event"])

# --- Dlib Face + Landmark Detector ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("🚗 Drowsy Driver Detection started...")

vs = cv2.VideoCapture(0)
while True:
    ret, frame = vs.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES and not ALARM_ON:
                ALARM_ON = True
                mixer.music.play()
                print("⚠️ Drowsiness detected!")

                # Log and send alert
                csv_writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Drowsiness Detected"])
                send_telegram_alert()

        else:
            COUNTER = 0
            ALARM_ON = False

    cv2.imshow("Drowsy Driver Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

csv_file.close()
vs.release()
cv2.destroyAllWindows()
