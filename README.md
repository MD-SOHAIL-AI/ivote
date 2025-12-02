🗳️ iVote – Intelligent Voting Login System

iVote is an intelligent face-recognition–based voting authentication system built using Python, OpenCV, face_recognition, and PyQt6.
It enables secure and automatic voter login using facial biometrics, ensuring accuracy, anti-duplication, and real-time logging.

📌 Features
🔹 1. Voter Registration

Captures 10 unique face samples per voter

Prevents duplicate images using encoding similarity

Enhances face quality (contrast + sharpness)

Saves images automatically in dataset/

Stores voter info (ID + Name) in voter_names.json

🔹 2. Model Training

Extracts 128-D face encodings using face_recognition

Maps encodings to voter IDs and names

Saves trained model as encodings.npz for fast loading

🔹 3. Live Face Recognition

Detects and recognizes faces in real time

Compares live encodings with known voters

Logs:

Timestamp

Voter ID

Name

Distance (accuracy)

uses a cooldown system to prevent immediate re-voting repeating.

🔹 4. PyQt6 Graphical Interface

Dark mode UI

Buttons for:

Select Camera

Register Voter

Train Recognizer

Run Recognition

Show Registered Voters

Exit

Live output logs inside GUI

📁 Project Structure
iVote/
│── dataset/                 # Registered face samples
│── trainer/
│     ├── voter_names.json   # All registered voter IDs + names
│     ├── encodings.npz      # Saved trained model
│     └── voters.csv         # Attendance and recognition log
│── main.py                  # Main program
│── README.md                # Project documentation

🔧 Technologies Used

Python 3.x

OpenCV

face_recognition (dlib)

NumPy

Pillow (PIL)

PyQt6

CSV / JSON data storage

📦 Installation
1️⃣ Clone the Repository
git clone https://github.com/yourusername/iVote.git
cd iVote

2️⃣ Install Dependencies
pip install opencv-python
pip install face_recognition
pip install numpy
pip install pillow
pip install PyQt6


⚠ Note:
face_recognition requires dlib, which may need CMake and Visual Studio Build Tools (on Windows).

▶️ Running the Application

Start the GUI:

python main.py

🛠 How the System Works
1️⃣ Registration Process

User enters Voter ID and Name

System captures 10 unique face images

Duplicate frames are skipped using encoding distance threshold

Processed images saved in dataset/

2️⃣ Training Process

All dataset images are scanned

Face encodings extracted

Encodings + IDs + names saved into encodings.npz

3️⃣ Recognition Process

Camera reads a live frame

Frame is enhanced (CLAHE, resize)

Face encoding extracted

Matched with stored encodings

If matched:

Display name

Log vote into voters.csv

Apply cooldown to prevent multiple votes

📝 Attendance Log Format (voters.csv)
timestamp_local	id	name	distance
2025-12-02T14:30:21	101	Arjun	0.3321
🛡 Security Notes

Each voter can be recognized only once per cooldown window.

Voter IDs are unique and stored securely.

Face recognition thresholds are tuned for accuracy and anti-spoofing.

🎯 Use Cases

College election systems

Smart attendance

Secure identity verification

Face-based login systems

📌 Future Enhancements (Optional)

Anti-spoofing (blink detection / depth scan)

Cloud database integration

Mobile app version

Fingerprint + face multi-mode authentication

👤 Author

Your Name
Intelligent Voting System Developer
(Replace with your details)
