# Real-Time Drowsiness Detection

A real-time drowsiness detection system using computer vision techniques built with Python, OpenCV, dlib, and Flask. The system monitors eye aspect ratio (EAR) via webcam to detect signs of drowsiness and triggers an alarm if the user appears sleepy.

## Features

- Real-time webcam monitoring
- Eye Aspect Ratio (EAR) based drowsiness detection
- On-screen and audio alerts
- Web interface using Flask

## Technologies Used

- Python
- OpenCV
- dlib
- pygame
- Flask

## Setup Instructions

1. Clone the repository  
   `git clone https://github.com/<your-username>/<your-repo>.git`

2. Install dependencies  
   `pip install -r requirements.txt`

3. Download the model  
   Download `shape_predictor_68_face_landmarks.dat` from  
   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2  
   and extract it into the project directory.

4. Run the Flask app  
   `python app.py`

5. Open the app in browser  
   Navigate to `http://127.0.0.1:5000/`

## Project Structure

- `app.py` : Main Flask application
- `templates/index.html` : Web interface
- `static/alert.mp3` : Alarm sound
- `shape_predictor_68_face_landmarks.dat` : dlib model

## Notes

- Adjust EAR threshold and frame count in code as needed
- Ensure webcam access is enabled

## License

This project is licensed under the MIT License.
