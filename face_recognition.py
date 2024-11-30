import cv2
import pickle

# Initialize the recognizer and load the model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./trainer.yml')  # Make sure the path to the model is correct

# Load the label_to_user mapping
with open('label_to_user.pkl', 'rb') as f:
    label_to_user = pickle.load(f)

# Initialize the webcam and face detector
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        
        # Predict the label and confidence of the face
        label, confidence = recognizer.predict(face)
        user_id = label_to_user.get(label, "Unknown")  # Map label to user name
        
        # Draw rectangle around face and display user info
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{user_id} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Display the result
    cv2.imshow("Face Recognition", frame)
    
    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()
