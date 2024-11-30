import cv2
import numpy as np
import os
import pickle

# Initialize the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directory where training data is stored
training_data_dir = "training_data"

def prepare_training_data():
    face_images = []
    face_labels = []
    label_to_user = {}
    
    # Iterate through the directory and load face images
    for user_id, user_dir in enumerate(os.listdir(training_data_dir)):
        user_path = os.path.join(training_data_dir, user_dir)
        label_to_user[user_id] = user_dir  # Map user_id to directory name (user name)
        
        for image_file in os.listdir(user_path):
            image_path = os.path.join(user_path, image_file)
            face_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if face_image is not None:
                face_images.append(face_image)
                face_labels.append(user_id)
    
    return face_images, np.array(face_labels), label_to_user

# Train the model
faces, labels, label_to_user = prepare_training_data()
recognizer.train(faces, labels)
print("Training completed!")

# Save the trained recognizer model
recognizer.save('trainer.yml')

# Save the label_to_user mapping
with open('label_to_user.pkl', 'wb') as f:
    pickle.dump(label_to_user, f)

print("Model saved as 'trainer.yml' and label mapping saved as 'label_to_user.pkl'")
