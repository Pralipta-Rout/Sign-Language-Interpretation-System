import tkinter as tk
from tkinter import messagebox
from keras.models import model_from_json
import cv2
import numpy as np
import pyttsx3
import threading
from PIL import Image, ImageTk

# Load model
json_file = open("signlanguagemodel48x48.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("signlanguagemodel48x48.h5")

# Initialize for sign-to-text and sign-to-speech
recognized_text = ""
last_prediction = "blank"
stable_counter = 0
stable_threshold = 15  # Adjust based on your need
engine = pyttsx3.init()

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Create the GUI window
window = tk.Tk()
window.title("Sign Language Recognition System")

# Set window size
window.geometry("600x600")  # Adjust size as needed

# Define a label to show recognized text
recognized_label = tk.Label(window, text="Recognized Text: ", font=("Helvetica", 16))
recognized_label.pack(pady=20)

# Label to show recognized text
recognized_text_label = tk.Label(window, text="", font=("Helvetica", 16))
recognized_text_label.pack(pady=20)

# Clear text function
def clear_text():
    global recognized_text
    recognized_text = ""
    recognized_text_label.config(text=recognized_text)

# Button to clear the recognized text
clear_button = tk.Button(window, text="Clear Text", command=clear_text, font=("Helvetica", 14))
clear_button.pack(pady=10)

# Video capture and recognition function
def video_capture():
    global recognized_text, last_prediction, stable_counter
    cap = cv2.VideoCapture(0)
    label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'blank']

    while True:
        _, frame = cap.read()
        cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
        cropframe = frame[40:300, 0:300]
        cropframe_gray = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
        cropframe_resized = cv2.resize(cropframe_gray, (48, 48))
        cropframe_features = extract_features(cropframe_resized)

        # Check if hand is present (variance method)
        variance = np.var(cropframe_resized)

        if variance > 500:  # Hand detected
            pred = model.predict(cropframe_features)
            confidence = np.max(pred)
            prediction_label = label[pred.argmax()]

            if prediction_label == last_prediction and prediction_label != 'blank':
                stable_counter += 1
            else:
                stable_counter = 0
            last_prediction = prediction_label

            if stable_counter > stable_threshold:
                if prediction_label != 'blank':
                    recognized_text = prediction_label
                    recognized_text_label.config(text=recognized_text)
                    engine.say(prediction_label)
                    engine.runAndWait()
                    stable_counter = 0

        else:
            prediction_label = 'blank'
            stable_counter = 0  # Reset if no hand

        # Convert frame to ImageTk object to show in Tkinter window
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)
        frame_image = frame_image.resize((400, 300))  # Resize frame to fit Tkinter window
        frame_photo = ImageTk.PhotoImage(frame_image)

        # Update the video feed in the Tkinter window
        video_label.config(image=frame_photo)
        video_label.image = frame_photo

        # Add a small delay
        window.update_idletasks()
        window.update()

    cap.release()

# Create a label widget for displaying the video feed
video_label = tk.Label(window)
video_label.pack(pady=20)

# Run video capture in a separate thread to keep the GUI responsive
video_thread = threading.Thread(target=video_capture, daemon=True)
video_thread.start()

# Start the GUI loop
window.mainloop()