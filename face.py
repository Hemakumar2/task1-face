from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels, stripping any newline characters
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Using DirectShow

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Grab the webcam's image
    ret, image = camera.read()

    # Check if frame was captured correctly
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Resize the raw image into (224-height, 224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the model's input shape
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predict the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    confidence_score = np.round(confidence_score * 100, 2)
    print(f"Class: {class_name}, Confidence Score: {confidence_score}%")

    # Listen to the keyboard for presses
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the ESC key
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()