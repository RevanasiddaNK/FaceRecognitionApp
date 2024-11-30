import cv2
import os
import numpy as np
import pickle

def capture_training_data(limit=200):
    """
    Captures training images for a user and saves them to the training_data directory.
    Stops automatically after capturing the specified number of images.
    """
    # Initialize webcam and Haar Cascade
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Directory to save training images
    user_id = input("Enter User ID: ")
    training_data_dir = f"training_data/{user_id}"
    os.makedirs(training_data_dir, exist_ok=True)

    print(f"Capturing up to {limit} images for User ID: {user_id}")
    image_count = 0

    while image_count < limit:
        ret, frame = cap.read()
        if not ret:
            print("Failed to access the webcam. Exiting.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{training_data_dir}/face_{image_count}.jpg", face)
            image_count += 1
            print(f"Captured image {image_count}/{limit}")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if image_count >= limit:
                break

        cv2.imshow("Face Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture interrupted by user. Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {image_count} images to {training_data_dir}")


def train_model():
    """
    Trains the LBPH Face Recognizer using the images in the training_data directory.
    Saves the model and the label-to-user mapping.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    training_data_dir = "training_data"
    face_images = []
    face_labels = []
    label_to_user = {}

    print("Training the model...")

    # Load training data
    for user_id, user_dir in enumerate(os.listdir(training_data_dir)):
        user_path = os.path.join(training_data_dir, user_dir)
        label_to_user[user_id] = user_dir  # Map user_id to directory name (user name)

        for image_file in os.listdir(user_path):
            image_path = os.path.join(user_path, image_file)
            face_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if face_image is not None:
                face_images.append(face_image)
                face_labels.append(user_id)

    # Train the recognizer
    recognizer.train(face_images, np.array(face_labels))
    print("Training completed!")

    # Save the trained model and label mapping
    recognizer.save('trainer.yml')
    with open('label_to_user.pkl', 'wb') as f:
        pickle.dump(label_to_user, f)

    print("Model saved as 'trainer.yml' and label mapping saved as 'label_to_user.pkl'")


if __name__ == "__main__":
    print("Starting the process...")

    # Step 1: Capture training data with a limit
    print("Step 1: Capturing training data")
    capture_training_data(limit=20)  # Specify the image capture limit here

    # Step 2: Train the model
    print("\nStep 2: Training the model")
    train_model()

    print("\nProcess completed! You can now use the model for face recognition.")
