# drowsysmooth_driver.py
import os
import re
import csv
import cv2
import dlib
import time
import math
import requests
import threading
import numpy as np
import pandas as pd
import geocoder
import matplotlib.pyplot as plt

from collections import deque
from datetime import datetime
from imutils import face_utils
from scipy.spatial import distance as dist
from pygame import mixer

# ---------------------- Configuration / Thresholds ----------------------
ALARM_SOUND = "alarm.wav"
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
MAR_THRESH = 0.30
MAR_CONSEC_FRAMES = 15
HEAD_DROP_THRESH_RATIO = 0.05
HEAD_DROP_CONSEC_FRAMES = 15

ROLLING_WINDOW = 10
MAX_HISTORY = 200

BOT_TOKEN = ""
CHAT_ID = ""

LOG_PATH = "drowsiness_log.csv"
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# ---------------------- Helpers: Sound & Telegram ----------------------
try:
    mixer.init()
    if not os.path.exists(ALARM_SOUND):
        raise FileNotFoundError(f"{ALARM_SOUND} not found!")
    mixer.music.load(ALARM_SOUND)
except Exception as e:
    print("[SOUND WARN] Mixer/Alarm init failed:", e)

def play_alarm():
    try:
        if not mixer.music.get_busy():
            mixer.music.play(-1)
    except Exception as e:
        print("[SOUND ERROR]", e)

def stop_alarm():
    try:
        if mixer.music.get_busy():
            mixer.music.stop()
    except Exception as e:
        print("[SOUND ERROR]", e)

def send_telegram_message(message, parse_mode=None):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    if parse_mode:
        data["parse_mode"] = parse_mode
    try:
        requests.post(url, data=data, timeout=8)
    except Exception as e:
        print("[TELEGRAM ERROR]", e)

def send_telegram_image(image_path, caption=""):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    try:
        with open(image_path, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": CHAT_ID, "caption": caption}
            requests.post(url, files=files, data=data, timeout=10)
        print(f"[TELEGRAM] Image sent: {image_path}")
    except Exception as e:
        print("[TELEGRAM ERROR]:", e)

# ---------------------- GPS / Location ----------------------
def get_gps_location():
    try:
        g = geocoder.ip('me')
        if g.ok and g.latlng:
            lat, lon = g.latlng
            url = f"https://maps.google.com/?q={lat},{lon}"
            return lat, lon, url
    except Exception:
        pass
    try:
        r = requests.get("http://ip-api.com/json/", timeout=6).json()
        if r.get("status") == "success":
            lat = r.get("lat")
            lon = r.get("lon")
            url = f"https://maps.google.com/?q={lat},{lon}"
            return lat, lon, url
    except Exception:
        pass
    return None, None, "Location unavailable"

# ---------------------- CSV Logging & Cleaner ----------------------
def ensure_log_headers(path):
    headers = ["Timestamp", "EAR", "MAR", "HeadTilt", "Event", "EventType", "Latitude", "Longitude", "LocationURL"]
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def clean_drowsiness_log(log_path):
    try:
        if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
            return
        df = pd.read_csv(log_path, on_bad_lines="skip", engine="python")
        if df.empty:
            return
        df = df[df["Timestamp"] != "Timestamp"]
        df = df.dropna(subset=["Timestamp"])
        def clean_num(val):
            if isinstance(val, str):
                m = re.findall(r"-?\d+\.\d+|-?\d+", val)
                if m: return float(m[0])
                return None
            return val
        for col in ["EAR","MAR","HeadTilt"]:
            if col in df.columns: df[col] = df[col].apply(clean_num)
        df = df.dropna(subset=["EAR"], how="all")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])
        df = df.sort_values(by="Timestamp")
        df = df.drop_duplicates(subset=["Timestamp"], keep="last")
        df.to_csv(log_path, index=False)
        print(f"[LOG CLEAN] Cleaned and saved {len(df)} entries.")
    except Exception as e:
        print("[LOG CLEAN ERROR]:", e)

ensure_log_headers(LOG_PATH)
clean_drowsiness_log(LOG_PATH)
csv_file = open(LOG_PATH, "a", newline="")
csv_writer = csv.writer(csv_file)

# ---------------------- EAR / MAR / Head computations ----------------------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C != 0 else 0.0

def mouth_aspect_ratio(mouth):
    try:
        A = dist.euclidean(mouth[13], mouth[19])
        B = dist.euclidean(mouth[14], mouth[18])
        C = dist.euclidean(mouth[15], mouth[17])
        D = dist.euclidean(mouth[12], mouth[16])
        return (A + B + C) / (3.0 * D) if D != 0 else 0.0
    except Exception:
        return 0.0

# ---------------------- Dlib model init ----------------------
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
except Exception as e:
    raise RuntimeError("Failed to load dlib predictor. Ensure shape_predictor_68_face_landmarks.dat is present.") from e

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# ---------------------- Live plotting setup ----------------------
# ---------------------- Live plotting setup ----------------------
plt.ion()
fig, ax = plt.subplots()
ear_hist = deque(maxlen=MAX_HISTORY)
mar_hist = deque(maxlen=MAX_HISTORY)
tilt_hist = deque(maxlen=MAX_HISTORY)
time_hist = deque(maxlen=MAX_HISTORY)  # store seconds since start

line_ear, = ax.plot([], [], lw=2, label="EAR")
line_mar, = ax.plot([], [], lw=1, linestyle=':', label="MAR")
line_tilt, = ax.plot([], [], lw=1, linestyle='-.', label="Head Tilt")
ax.set_ylim(0, 1.0)
ax.set_xlim(0, MAX_HISTORY)
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Value")
ax.set_title("Real-time EAR / MAR / Head Tilt")
ax.legend()

def update_plot():
    try:
        xdata = list(time_hist)
        line_ear.set_data(xdata, list(ear_hist))
        line_mar.set_data(xdata, list(mar_hist))
        line_tilt.set_data(xdata, list(tilt_hist))
        if xdata:
            ax.set_xlim(max(0, xdata[0]), xdata[-1]+1)
        plt.pause(0.001)
    except Exception:
        pass

# ---------------------- State variables ----------------------
COUNTER = 0
ALARM_ON = False
yawn_counter = 0
head_drop_counter = 0
baseline_initialized = False
baseline_chin_eye = None
baseline_samples = []
BASELINE_FRAMES = 30

vs = cv2.VideoCapture(0)
if not vs.isOpened():
    raise RuntimeError("Cannot open webcam")

print(" Drowsy Driver Detection started...")

frame_count = 0
last_log_time = time.time()
start_time = time.time()

while True:
    ret, frame = vs.read()
    if not ret: break

    small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    ear = 0.0
    mar = 0.0
    head_tilt_angle = 0.0
    event_this_frame = None
    lat, lon, loc_url = None, None, "Location unavailable"

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR)/2.0

        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        chin = shape[8]
        eye_center = np.mean(np.concatenate((leftEye,rightEye)),axis=0)
        face_height = max(np.linalg.norm(chin - shape[27]), 1.0)
        chin_eye_offset = (chin[1]-eye_center[1])/face_height

        delta_x = chin[0]-eye_center[0]
        delta_y = chin[1]-eye_center[1]
        head_tilt_angle = math.degrees(math.atan2(delta_y, delta_x))

        if not baseline_initialized:
            baseline_samples.append(chin_eye_offset)
            if len(baseline_samples)>=BASELINE_FRAMES:
                baseline_chin_eye = float(np.mean(baseline_samples))
                baseline_initialized=True
                print(f"[BASELINE] head drop baseline set: {baseline_chin_eye:.2f}")

        for (x,y) in shape:
            cv2.circle(small, (x,y), 1, (0,255,0), -1)

    # ----------------------- Live plotting -----------------------
    current_seconds = int(time.time() - start_time)
    ear_hist.append(float(ear))
    mar_hist.append(float(mar))
    tilt_hist.append(float(head_tilt_angle))
    time_hist.append(current_seconds)

    frame_count +=1
    if frame_count % 5 == 0:
        update_plot()


    # ----------------------- Drowsiness detection -----------------------
    if 0<ear<EYE_AR_THRESH:
        COUNTER+=1
        if COUNTER>=EYE_AR_CONSEC_FRAMES and not ALARM_ON:
            ALARM_ON=True
            threading.Thread(target=play_alarm).start()
            event_this_frame="EyeClosure"
    else:
        COUNTER=0
        if ALARM_ON: stop_alarm()
        ALARM_ON=False

    if mar>MAR_THRESH:
        yawn_counter+=1
        if yawn_counter>=MAR_CONSEC_FRAMES:
            event_this_frame=event_this_frame or "Yawn"
    else:
        yawn_counter=0

    if baseline_initialized and rects:
        if chin_eye_offset-baseline_chin_eye>HEAD_DROP_THRESH_RATIO:
            head_drop_counter+=1
            if head_drop_counter>=HEAD_DROP_CONSEC_FRAMES:
                event_this_frame=event_this_frame or "HeadDrop"
        else:
            head_drop_counter=0

    # ----------------------- CSV Logging per second -----------------------
    current_time = time.time()
    if current_time - last_log_time >= 1.0:
        readable_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_writer.writerow([readable_ts,f"{ear:.3f}",f"{mar:.3f}",f"{head_tilt_angle:.1f}",event_this_frame or "",event_this_frame or "",
                             lat if lat else "", lon if lon else "", loc_url])
        csv_file.flush()
        last_log_time = current_time

    # ----------------------- Alerts -----------------------
    if event_this_frame:
        ts=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        try:
            lat, lon, loc_url=get_gps_location()
        except:
            lat, lon, loc_url=None,None,"Location unavailable"

        fname_frame=f"user_{event_this_frame}_{ts}.png"
        cv2.imwrite(fname_frame, frame)

        plt.figure(figsize=(6,3))
        plt.plot(list(ear_hist),label="EAR")
        plt.axhline(EYE_AR_THRESH,color='r',linestyle='--',label="EAR Threshold")
        plt.xlabel("Time (seconds)"); plt.ylabel("EAR"); plt.title(f"{event_this_frame} EAR Trend")
        plt.legend(); plt.tight_layout()
        fname_graph=f"ear_graph_{event_this_frame}_{ts}.png"
        plt.savefig(fname_graph); plt.close()

        message=(f"⚠️ *Drowsiness Alert*\nType: {event_this_frame}\nTime: {readable_ts}\nEAR: {ear:.3f} MAR: {mar:.3f}\nHead Tilt: {head_tilt_angle:.1f}°\nLocation: {loc_url}")
        threading.Thread(target=send_telegram_message,args=(message,"Markdown")).start()
        threading.Thread(target=send_telegram_image,args=(fname_frame,f"🚨 {event_this_frame} snapshot")).start()
        threading.Thread(target=send_telegram_image,args=(fname_graph,"📊 EAR Trend at Alert")).start()

        COUNTER=0; yawn_counter=0; head_drop_counter=0
        time.sleep(1)

    cv2.imshow("Drowsy Driver Detection", small)
    key=cv2.waitKey(1)&0xFF
    if key==ord("q"): break

# ---------------------- Cleanup ----------------------
csv_file.close()
vs.release()
cv2.destroyAllWindows()
plt.close("all")

# ---------------------- Post-drive analysis ----------------------
def send_analysis_graphs(log_path):
    try:
        df = pd.read_csv(log_path, on_bad_lines="skip", engine="python")
        df = df.dropna(subset=["Timestamp"])
        df = df[df["Timestamp"] != "Timestamp"]
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])

        def clean_num(val):
            if isinstance(val, str):
                m = re.findall(r"-?\d+\.\d+|-?\d+", val)
                if m: return float(m[0])
                return None
            return val

        for col in ["EAR", "MAR", "HeadTilt"]:
            if col in df.columns: df[col] = df[col].apply(clean_num)
        df = df.dropna(subset=["EAR"], how="all")

        # ---- Stacked subplots ----
        fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        # EAR
        axs[0].plot(df['Timestamp'], df['EAR'], color='blue', label='EAR')
        axs[0].axhline(EYE_AR_THRESH, color='red', linestyle='--', alpha=0.5, label='EAR Threshold')
        axs[0].set_ylabel('EAR')
        axs[0].legend(loc='upper right')
        axs[0].set_title('Drowsiness Metrics Over Time')

        # MAR
        axs[1].plot(df['Timestamp'], df['MAR'], color='orange', label='MAR')
        axs[1].axhline(MAR_THRESH, color='green', linestyle='--', alpha=0.5, label='Yawn Threshold')
        axs[1].set_ylabel('MAR')
        axs[1].legend(loc='upper right')

        # Head Tilt
        axs[2].plot(df['Timestamp'], df['HeadTilt'], color='purple', label='Head Tilt (°)')
        axs[2].set_ylabel('Head Tilt (°)')
        axs[2].set_xlabel('Time')
        axs[2].legend(loc='upper right')

        plt.tight_layout()
        combined_graph_path = f"post_combined_graph_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        plt.savefig(combined_graph_path)
        plt.close()
        send_telegram_image(combined_graph_path)

        # Frequency summary remains unchanged
        ...

    except Exception as e:
        print("[ANALYSIS ERROR]", e)

        # ---- Frequency summary ----
        if "EventType" in df.columns:
            drowsy_df = df[df["EventType"].notna() & (df["EventType"] != "")]
            if not drowsy_df.empty:
                drowsy_df["Minute"] = drowsy_df["Timestamp"].dt.floor("1min")
                freq = drowsy_df.groupby("Minute").size()
                plt.figure(figsize=(10,4))
                plt.bar(freq.index, freq.values, color="teal")
                plt.title("Drowsiness Events Over Time (per minute)")
                plt.xlabel("Time (Minute)"); plt.ylabel("Count")
                plt.tight_layout()
                freq_graph_path = f"drowsiness_freq_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
                plt.savefig(freq_graph_path)
                plt.close()
                send_telegram_image(freq_graph_path)

                summary_message = (
                    f"📊 *Session Summary*\n"
                    f"Total Events: {len(drowsy_df)}\n"
                    f"Session Duration: {df['Timestamp'].min().strftime('%H:%M:%S')} - {df['Timestamp'].max().strftime('%H:%M:%S')}\n"
                    f"Average EAR: {df['EAR'].mean():.3f}\n"
                    f"Average MAR: {df['MAR'].mean():.3f}\n"
                    f"Average Head Tilt: {df['HeadTilt'].mean():.1f}°"
                )
                send_telegram_message(summary_message, "Markdown")

    except Exception as e:
        print("[ANALYSIS ERROR]", e)

# Call post-drive analysis after script ends
send_analysis_graphs(LOG_PATH)