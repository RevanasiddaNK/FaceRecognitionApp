import cv2
import os

# Initialize the webcam and Haar Cascade
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directory to save training images
user_id = input("Enter User ID: ")
training_data_dir = f"training_data/{user_id}"
os.makedirs(training_data_dir, exist_ok=True)

print("Press 'q' to quit after capturing images.")

image_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        cv2.imwrite(f"{training_data_dir}/face_{image_count}.jpg", face)
        image_count += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow("Face Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Saved {image_count} images to {training_data_dir}")
