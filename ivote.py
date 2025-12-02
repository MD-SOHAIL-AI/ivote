import cv2
import os
import json
import csv
import logging
import time
from datetime import datetime
import numpy as np
from PIL import Image, ImageEnhance
import face_recognition
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QInputDialog, QMessageBox, QTextEdit
)
import sys
DATASET_PATH = "dataset"
TRAINER_PATH = "trainer"
VOTER_JSON = os.path.join(TRAINER_PATH, "voter_names.json")
ENCODINGS_FILE = os.path.join(TRAINER_PATH, "encodings.npz")
ATTENDANCE_CSV = os.path.join(TRAINER_PATH, "voters.csv")

SAMPLES_PER_PERSON = 10
TOLERANCE = 0.45
DUPLICATE_TOLERANCE = 0.38
RESIZE_SCALE = 0.5
FACE_DETECTION_MODEL = "hog"  # "hog" for CPU, "cnn" for GPU/high accuracy
COOLDOWN = 5 

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(TRAINER_PATH, exist_ok=True)

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Redirect logs to GUI
class QTextLogger(logging.Handler):
    def _init_(self, text_edit):
        super()._init_()
        self.text_edit = text_edit

    def emit(self, record):
        msg = self.format(record)
        self.text_edit.append(msg)

# ----------------------------
# Data Management
# ----------------------------
def load_voter_dict():
    if os.path.exists(VOTER_JSON):
        with open(VOTER_JSON, "r") as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}

def save_voter_dict(voter_dict):
    with open(VOTER_JSON, "w") as f:
        json.dump({str(k): v for k, v in voter_dict.items()}, f, indent=2)

def log_attendance(id_num, name, distance):
    header_needed = not os.path.exists(ATTENDANCE_CSV)
    with open(ATTENDANCE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["timestamp_local", "id", "name", "distance"])
        writer.writerow([datetime.now().isoformat(), id_num, name, round(distance, 4)])

# ----------------------------
# Image Processing
# ----------------------------
def enhance_image(frame, top, right, bottom, left):
    face_img = frame[top:bottom, left:right]
    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.5)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def preprocess_frame(frame):
    # Resize
    if RESIZE_SCALE != 1.0:
        frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
    # Convert to YUV and CLAHE for lighting correction
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return frame

# ----------------------------
# Registration
# ----------------------------
def register_voter(voter_id, voter_name, camera_index):
    voter_dict = load_voter_dict()
    if voter_id in voter_dict:
        logging.warning(f"ID {voter_id} already registered. Overwriting name.")
    voter_dict[voter_id] = voter_name
    save_voter_dict(voter_dict)

    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        logging.error("Cannot open camera.")
        return

    logging.info(f"Registering {voter_name} (ID: {voter_id}). ESC to stop.")
    count = 0
    collected_encodings = []

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        frame = preprocess_frame(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame, model=FACE_DETECTION_MODEL)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check for duplicate sample
            if collected_encodings:
                distances = face_recognition.face_distance(collected_encodings, face_encoding)
                if np.min(distances) < DUPLICATE_TOLERANCE:
                    cv2.putText(frame, "Duplicate frame skipped", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    continue

            # Save unique sample
            enhanced_face = enhance_image(frame, top, right, bottom, left)
            count += 1
            file_name = f"{voter_name}.{voter_id}.{count}.jpg"
            file_path = os.path.join(DATASET_PATH, file_name)
            cv2.imwrite(file_path, enhanced_face)
            collected_encodings.append(face_encoding)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"Saved {count}/{SAMPLES_PER_PERSON}", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if count >= SAMPLES_PER_PERSON:
                break

        cv2.imshow("Register Voter (Avoid Duplicates)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or count >= SAMPLES_PER_PERSON:
            break

    cam.release()
    cv2.destroyAllWindows()
    logging.info(f"Registration completed with {count} unique samples, duplicates skipped: {SAMPLES_PER_PERSON - count if SAMPLES_PER_PERSON-count>0 else 0}")

# ----------------------------
# Training
# ----------------------------
def train_recognizer():
    voter_dict = load_voter_dict()
    if not voter_dict:
        logging.error("No voters registered.")
        return

    encodings, names, ids = [], [], []
    image_files = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not image_files:
        logging.error("No images found in dataset.")
        return

    logging.info("Extracting face encodings...")
    for image_name in image_files:
        path = os.path.join(DATASET_PATH, image_name)
        parts = image_name.split(".")
        if len(parts) < 3:
            continue
        try:
            id_num = int(parts[1])
            name = voter_dict.get(id_num, None)
        except ValueError:
            continue
        if name is None:
            continue

        image = face_recognition.load_image_file(path)
        face_locations = face_recognition.face_locations(image, model=FACE_DETECTION_MODEL)
        if not face_locations:
            continue
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
        encodings.append(face_encoding)
        names.append(name)
        ids.append(id_num)

    np.savez(ENCODINGS_FILE, encodings=encodings, names=names, ids=ids)
    logging.info(f"Model trained with {len(encodings)} unique samples. Encodings saved.")

# ----------------------------
# Recognition
# ----------------------------
def recognize_voters(camera_index):
    if not os.path.exists(ENCODINGS_FILE):
        logging.error("No trained encodings found.")
        return

    data = np.load(ENCODINGS_FILE, allow_pickle=True)
    known_encodings = list(data["encodings"])
    known_names = list(data["names"])
    known_ids = list(data["ids"])

    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        logging.error("Cannot open camera.")
        return

    voted_ids = set()
    last_seen = {}  # cooldown tracker
    logging.info("Recognition started. Press 'q' to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        frame = preprocess_frame(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame, model=FACE_DETECTION_MODEL)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(distances) == 0:
                continue

            min_distance = np.min(distances)
            best_match_index = np.argmin(distances)
            if min_distance <= TOLERANCE:
                name = known_names[best_match_index]
                voter_id = known_ids[best_match_index]

                # Check cooldown
                now = time.time()
                if voter_id in last_seen and (now - last_seen[voter_id] < COOLDOWN):
                    text = f"{name} - Already voted"
                else:
                    log_attendance(voter_id, name, min_distance)
                    voted_ids.add(voter_id)
                    last_seen[voter_id] = now
                    text = f"{name} - Voted"
            else:
                text = "Unknown"

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Voter Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    logging.info("Recognition stopped.")

# ----------------------------
# GUI
# ----------------------------
dark_stylesheet = """
    QWidget {
        background-color: #121212;
        color: #e0e0e0;
    }
    QPushButton {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 5px;
        border-radius: 4px;
    }
    QPushButton:hover {
        background-color: #333333;
    }
    QTextEdit {
        background-color: #1e1e1e;
        color: #e0e0e0;
        border: 1px solid #333;
    }
    QInputDialog, QMessageBox {
        background-color: #121212;
        color: #e0e0e0;
    }
"""

class VoterFaceSystemGUI(QMainWindow):
    def _init_(self):
        super()._init_()
        self.camera_index = 1

        self.setWindowTitle('Voter Face System')
        self.setGeometry(200, 200, 450, 400)

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)

        # Buttons
        self.btn_select_camera = QPushButton('Select Camera Index (Current: 1)')
        self.btn_register = QPushButton('Register Voter')
        self.btn_train = QPushButton('Train Recognizer')
        self.btn_recognize = QPushButton('Run Recognition (Live)')
        self.btn_show_voters = QPushButton('Show Registered Voters')
        self.btn_exit = QPushButton('Exit')

        self.layout.addWidget(self.btn_select_camera)
        self.layout.addWidget(self.btn_register)
        self.layout.addWidget(self.btn_train)
        self.layout.addWidget(self.btn_recognize)
        self.layout.addWidget(self.btn_show_voters)
        self.layout.addWidget(self.btn_exit)

        # Output log
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.layout.addWidget(self.output_text)

        # Connect signals
        self.btn_select_camera.clicked.connect(self.select_camera)
        self.btn_register.clicked.connect(self.register_voter)
        self.btn_train.clicked.connect(self.train_recognizer)
        self.btn_recognize.clicked.connect(self.recognize_voters)
        self.btn_show_voters.clicked.connect(self.show_voters)
        self.btn_exit.clicked.connect(self.close)

        # Redirect logging
        text_logger = QTextLogger(self.output_text)
        logging.getLogger().addHandler(text_logger)

    def select_camera(self):
        val, ok = QInputDialog.getInt(self, "Camera Index", "Enter camera index:", self.camera_index, 0, 10, 1)
        if ok:
            self.camera_index = val
            self.btn_select_camera.setText(f'Select Camera Index (Current: {val})')
            self.output_text.append(f"Camera index set to {val}.")

    def register_voter(self):
        id_num, ok1 = QInputDialog.getInt(self, 'Input', 'Enter voter numeric ID (unique integer):')
        if not ok1:
            return
        name, ok2 = QInputDialog.getText(self, 'Input', 'Enter voter name:')
        if not ok2 or not name:
            return
        try:
            register_voter(id_num, name, self.camera_index)
            self.output_text.append(f"Registered voter {name} (ID {id_num}) successfully.")
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

    def train_recognizer(self):
        try:
            train_recognizer()
            self.output_text.append("Training completed successfully.")
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

    def recognize_voters(self):
        try:
            recognize_voters(self.camera_index)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

    def show_voters(self):
        voter_dict = load_voter_dict()
        if not voter_dict:
            self.output_text.append("No voters registered.")
        else:
            self.output_text.append("Registered voters:")
            for k in sorted(voter_dict.keys()):
                self.output_text.append(f"  ID: {k}  Name: {voter_dict[k]}")

# ----------------------------
# Main
# ----------------------------
def gui_main():
    app = QApplication(sys.argv)
    app.setStyleSheet(dark_stylesheet)
    window = VoterFaceSystemGUI()
    window.show()
    sys.exit(app.exec())

if _name_ == "_main_":
    gui_main()