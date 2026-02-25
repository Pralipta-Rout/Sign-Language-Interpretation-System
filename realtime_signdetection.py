from keras.models import model_from_json
import cv2
import numpy as np
import pyttsx3

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
                recognized_text = prediction_label  # Only show the current recognized letter
                print("Recognized Text:", recognized_text)
                engine.say(prediction_label)
                engine.runAndWait()
                stable_counter = 0

    else:
        prediction_label = 'blank'
        stable_counter = 0  # Reset if no hand

    # Display prediction
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    if prediction_label == 'blank':
        cv2.putText(frame, "No Hand Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, f'{prediction_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()