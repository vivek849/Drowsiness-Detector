import cv2
import dlib

# Load pre-trained model
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(predictor_path)

# Load the real face image
image = cv2.imread("sample.jpg")  # Replace with your file name
if image is None:
    print("Error: Image not loaded. Check the file path.")
    exit()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect face(s)
faces = face_detector(gray)

# Loop through faces and draw landmarks
for face in faces:
    landmarks = landmark_predictor(gray, face)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

# Display the result
cv2.imshow("Facial Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
