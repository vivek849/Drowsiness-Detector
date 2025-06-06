from flask import Flask, render_template, Response
import cv2
import dlib
from scipy.spatial import distance
import pygame
import os

app = Flask(__name__)

# Initialize pygame for alarm
pygame.mixer.init()
alarm_file = "static/alert.mp3"

def play_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(alarm_file)
        pygame.mixer.music.play(-1)

def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

# EAR calculation
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def draw_eye(frame, landmarks, start, end):
    eye = []
    for n in range(start, end):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        eye.append((x, y))
    return eye

# Load model
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(predictor_path)

EAR_THRESHOLD = 0.26
DROWSY_FRAMES = 30
frame_counter = 0
alarm_on = False

# Video stream generator
def gen_frames():
    global frame_counter, alarm_on
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)

            if len(faces) == 0:
                cv2.putText(frame, "FACE NOT DETECTED!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                if not alarm_on:
                    alarm_on = True
                    play_alarm()
            else:
                for face in faces:
                    landmarks = landmark_predictor(gray, face)
                    left_eye = draw_eye(frame, landmarks, 36, 42)
                    right_eye = draw_eye(frame, landmarks, 42, 48)

                    left_ear = calculate_EAR(left_eye)
                    right_ear = calculate_EAR(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    ear = round(ear, 2)

                    if ear < EAR_THRESHOLD:
                        frame_counter += 1
                        if frame_counter >= DROWSY_FRAMES:
                            cv2.putText(frame, "DROWSY!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
                            if not alarm_on:
                                alarm_on = True
                                play_alarm()
                    else:
                        frame_counter = 0
                        if alarm_on:
                            stop_alarm()
                            alarm_on = False

                    cv2.putText(frame, f"EAR: {ear}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
