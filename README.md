Face Recognition Using Google Teachable Machine

This project demonstrates how to implement a face recognition system using a model trained with Google Teachable Machine. It includes step-by-step instructions for setup, training, and execution in PyCharm.

Table of Contents:
Overview
Features
Prerequisites
Setting Up in PyCharm
How to Train the Model
Running the Project
Project Structure
Example Code
Credits

Overview:
This project uses a pre-trained face recognition model from Google Teachable Machine to identify individuals. It is implemented in Python and configured to run in PyCharm.

Features:
Train face recognition models with ease.
Real-time face recognition using a webcam.
Simple integration of Teachable Machine's TensorFlow model.


Below is a detailed README.md file template for your face recognition project using Google Teachable Machine, with specific instructions for setting up and running the project in PyCharm.

Face Recognition Using Google Teachable Machine
This project demonstrates how to implement a face recognition system using a model trained with Google Teachable Machine. It includes step-by-step instructions for setup, training, and execution in PyCharm.

Table of Contents
Overview
Features
Prerequisites
Setting Up in PyCharm
How to Train the Model
Running the Project
Project Structure
Example Code
Credits
Overview
This project uses a pre-trained face recognition model from Google Teachable Machine to identify individuals. It is implemented in Python and configured to run in PyCharm.

Features
Train face recognition models with ease.
Real-time face recognition using a webcam.
Simple integration of Teachable Machine's TensorFlow model.

Prerequisites

Python Version: Python 3.11 or later.
Libraries Required: TensorFlow, OpenCV, NumPy.
PyCharm IDE: Install the PyCharm Community Edition or Professional Edition

How to Train the Model:

Step 1: Use Google Teachable Machine:
Go to the Teachable Machine website.
![image](https://github.com/user-attachments/assets/0aa6243e-7886-4d18-8c30-509a03f4b1c7)
![image](https://github.com/user-attachments/assets/f1f0c5ae-c911-428d-a829-f5f097a14608)

Choose Image Project.
![image](https://github.com/user-attachments/assets/99d71f6e-acc9-43b6-af76-b2f63e8d24e9)


Step 2: Train Your Model:
Create classes for each individual to recognize.
![image](https://github.com/user-attachments/assets/3e18db63-b27c-42a6-9440-d7ece178766b)

Upload images or use your webcam to capture data for each class.
Click Train Model.

Step 3: Export and Download
Export the model by clicking Export Model.
![image](https://github.com/user-attachments/assets/602d455c-c036-43b6-bd5a-10f2efd68785)

Select the TensorFlow format.
Download the .h5 file and place it in the model/ folder

Project Structure:

face-recognition-teachable-machine/
│
├── model/               # Folder for the Teachable Machine model
│   └── model.h5         # Pre-trained model file
├── main.py              # Main application script
├── config.py            # Configuration settings (e.g., model path, class names)
├── requirements.txt     # List of required libraries
├── utils/               # Utility scripts for preprocessing or helper functions
└── README.md            # Documentation

Example Code:
main.py

Import Libraries:
import cv2
import tensorflow as tf
import numpy as np

Load the Trained Model:
model = tf.keras.models.load_model('model/model.h5')

Define Class Labels
class_labels = ['Person 1','Unknown']

Initialize the Webcam:
cap = cv2.VideoCapture(0)

Main Loop for Real-Time Recognition:
while True:
    ret, frame = cap.read()
    if not ret:
        break

Preprocess the Frame:
resized_frame = cv2.resize(frame, (224, 224))
normalized_frame = np.expand_dims(resized_frame / 255.0, axis=0)

Make Predictions:
predictions = model.predict(normalized_frame)
class_index = np.argmax(predictions)
confidence = predictions[0][class_index]

Display Results:
label = f"{class_labels[class_index]} ({confidence:.2f})"
cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('Face Recognition', frame)

 Exit the Application:
 if cv2.waitKey(1) & 0xFF == ord('q'):
    break

Release Resources:
cap.release()
cv2.destroyAllWindows()

![image](https://github.com/user-attachments/assets/6efc9262-a545-4903-86c3-11f5ce74464b)


Running the Project:
Click the Run button or use the shortcut Shift + F10.
![image](https://github.com/user-attachments/assets/742a40a5-1cc3-4f78-9fe1-2824b3a2061a)

The application will start and access your webcam to perform real-time face recognition. and click esc after getting the output

Summary of Workflow:
Initialize the webcam and load the trained model.
Continuously capture frames from the webcam.
Preprocess each frame (resize, normalize).
Pass the preprocessed frame to the model for predictions.
Display the predicted class and confidence score on the frame.
Exit the loop and clean up when the q key is pressed.
This code combines machine learning (model inference) and computer vision (video processing) to create a real-time face recognition system.
