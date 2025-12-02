iVote – Intelligent Voting Login System

iVote is an intelligent face-recognition–based voting authentication system built using Python, OpenCV, face_recognition, and PyQt6.
It enables secure, automatic voter login using facial biometrics, ensuring accuracy, anti-duplication, and real-time vote logging.

Features
1. Voter Registration

Captures 10 unique face samples

Automatically skips duplicate frames using encoding distance

Enhances images (contrast and sharpness) for better training

Saves images in dataset/

Stores voter details in voter_names.json

2. Model Training

Extracts 128-D face encodings

Associates encodings with voter IDs and names

Saves the trained model in encodings.npz

3. Real-Time Face Recognition

Live camera-based recognition

Matches live encodings with stored encodings

Logs timestamp, voter ID, voter name, and accuracy distance

Includes a cooldown mechanism to prevent repeat voting

4. PyQt6 GUI Interface

Dark-themed, clean interface with buttons for:

Select Camera

Register Voter

Train Recognizer

Live Recognition

Show Registered Voters

Exit

Real-time log output is displayed inside the GUI window.

Project Structure
ivote/
│── dataset/                 # Captured face samples
│── trainer/
│     ├── voter_names.json   # Registered voter details
│     ├── encodings.npz      # Saved face encodings/model
│     └── voters.csv         # Recognition log
│── main.py                  # Main application code
│── README.md                # Project documentation

Technologies Used

Python 3.x

OpenCV

face_recognition (dlib)

NumPy

Pillow

PyQt6

CSV and JSON storage

Installation
1. Clone the Repository
git clone https://github.com/MD-SOHAIL-AI/ivote.git
cd ivote

2. Install Dependencies
pip install opencv-python
pip install face_recognition
pip install numpy
pip install pillow
pip install PyQt6


Note:
face_recognition requires dlib, which may need CMake and Visual Studio Build Tools on Windows.

Running the Application

Start the GUI:

python main.py

How the System Works
1. Registration

User enters voter ID and name

System captures 10 unique face samples

Duplicate frames are automatically skipped

Images are saved and voter details updated

2. Training

All images in dataset/ are processed

Face encodings are extracted

Data is saved into encodings.npz

3. Live Recognition

Detects live face from camera

Enhances the frame using CLAHE

Compares with stored encodings

On match, logs the vote and applies cooldown

Attendance Log Format

voters.csv contains:

timestamp_local	id	name	distance
Security Features

Duplicate frame prevention

Cooldown to block repeated votes

Secure JSON and NPZ storage

Unique voter ID mapping

Use Cases

College or university election systems

Secure attendance marking

Face-based login systems

Biometric identity verification

Author

SOHAIL
Developer – Intelligent Voting System
GitHub: MD-SOHAIL-AI
