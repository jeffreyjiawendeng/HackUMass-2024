import numpy as np
import cv2
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import model_from_json # type: ignore

def load_model():
    # Use a relative path to open `fer3.json`
    with open("fer3.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("fer3.weights.h5")  # Load weights from file
    return model

def preprocess_face(gray_img, face_coords):
    x, y, w, h = face_coords
    roi_gray = gray_img[y:y+w, x:x+h]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    img_pixels = image.img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255
    return img_pixels

def get_emotion_prediction(model, img_pixels):
    predictions = model.predict(img_pixels)
    max_index = np.argmax(predictions[0])
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    return emotions[max_index]
