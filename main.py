from cv2 import BORDER_CONSTANT, COLOR_BGR2GRAY, COLOR_BGR2RGB, FONT_HERSHEY_SIMPLEX, copyMakeBorder, cvtColor, rectangle, resize, putText, CascadeClassifier, VideoCapture
from json import load
from numpy import argmax, expand_dims
from PIL.ImageTk import PhotoImage
from PIL.Image import fromarray
from random import choice
import sys
from tensorflow.keras.models import model_from_json # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tkinter import NW, Canvas, Label, Frame, Tk
from os.path import abspath, dirname, join
from os import environ

environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', dirname(abspath(__file__)))
    return join(base_path, relative_path)

encouragement_file = resource_path("encouragement_test.json")
fer3_file = resource_path("fer3.json")
weights_file = resource_path("fer3.weights.h5")
haar_file = resource_path("haarcascade_frontalface_default.xml")

def load_model():
    with open(fer3_file, "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(weights_file)  
    return model

def preprocess_face(gray_img, face_coords):
    x, y, w, h = face_coords
    roi_gray = gray_img[y:y+w, x:x+h]
    roi_gray = resize(roi_gray, (48, 48))
    img_pixels = image.img_to_array(roi_gray)
    img_pixels = expand_dims(img_pixels, axis=0)
    img_pixels /= 255
    return img_pixels

def get_emotion_prediction(model, img_pixels):
    predictions = model.predict(img_pixels)
    max_index = argmax(predictions[0])
    emotions = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
    return emotions[max_index]

def load_encouragement_messages():
    with open(encouragement_file, 'r') as file:
        data = load(file)
    return data

class BrightSide:
    def __init__(self, root):
        self.root = root
        self.root.title("BrightSide")
        self.root.geometry("250x500") 
        self.root.resizable(False, False)

        self.root.geometry("+0+200")  
        self.root.attributes("-topmost", True)  

        self.model = load_model()
        self.face_haar_cascade = CascadeClassifier(haar_file)
        if self.face_haar_cascade.empty():
            raise ValueError("Haar Cascade file not loaded properly!")

        self.cap = VideoCapture(0)

        self.setup_ui()
        self.current_emotion = "neutral"
        self.not_happy_counter = 0
        self.frame_count = 0

        self.emotion_counts = {
            "angry": 0,
            "disgust": 0,
            "fear": 0,
            "happy": 0,
            "sad": 0,
            "surprise": 0,
            "neutral": 0
        }
        
        self.run_emotion_detection()

    def setup_ui(self):
        self.main_frame = Frame(self.root, bg="#FFE5B4", padx=10, pady=10)  
        self.main_frame.pack(fill="both", expand=True)

        self.canvas = Canvas(self.main_frame, width=230, height=200)  
        self.canvas.pack(pady=5)

        self.encouragement_text = Label(self.main_frame, text="Stay positive!", font=("Segoe UI", 12), bg="#FFE5B4", fg="#5A4E35", pady=5, wraplength=230)
        self.encouragement_text.pack(pady=5, padx=5)

        self.exit_button_canvas = Canvas(self.main_frame, width=100, height=100, bg="#FFE5B4", bd=0, highlightthickness=0)
        self.exit_button_canvas.pack(pady=10)

        self.exit_button = self.exit_button_canvas.create_oval(10, 10, 90, 90, fill="#FFB6C1", outline="#FF9AA2")
        self.exit_button_canvas.create_text(50, 50, text="Exit", font=("Segoe UI", 16, "bold"), fill="#5A4E35")
        self.exit_button_canvas.tag_bind(self.exit_button, "<Button-1>", self.exit_app)
        self.exit_button_canvas.tag_bind(self.exit_button, "<Enter>", self.on_button_hover)
        self.exit_button_canvas.tag_bind(self.exit_button, "<Leave>", self.on_button_hover_leave)

        self.encouragement_text.lift()

    def on_button_hover(self, event):
        self.exit_button_canvas.itemconfig(self.exit_button, fill="#FF9AA2")

    def on_button_hover_leave(self, event):
        self.exit_button_canvas.itemconfig(self.exit_button, fill="#FFB6C1")

    def run_emotion_detection(self):
        self.update_frame()
        self.root.after(10, self.run_emotion_detection)

    def update_frame(self):
        ret, test_img = self.cap.read()
        if not ret:
            return

        gray_img = cvtColor(test_img, COLOR_BGR2GRAY)
        faces_detected = self.face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        self.current_frame_counts = {
            "angry": 0,
            "disgust": 0,
            "fear": 0,
            "happy": 0,
            "sad": 0,
            "surprise": 0,
            "neutral": 0
        }

        for (x, y, w, h) in faces_detected:
            rectangle(test_img, (x, y), (x + w, y + h), (180, 229, 255), thickness=7)
            img_pixels = preprocess_face(gray_img, (x, y, w, h))
            predicted_emotion = get_emotion_prediction(self.model, img_pixels)
            self.current_emotion = predicted_emotion

            if predicted_emotion in self.current_frame_counts:
                self.current_frame_counts[predicted_emotion] += 1

            putText(test_img, predicted_emotion, (int(x), int(y)), FONT_HERSHEY_SIMPLEX, 1, (162, 154, 255), 2)

        for emotion in self.current_frame_counts:
            self.emotion_counts[emotion] += self.current_frame_counts[emotion]

        self.frame_count += 1
        resized_frame = self.resize_frame(test_img, 230, 200)  

        photo = self.cv2_to_tkinter(resized_frame)

        self.canvas.create_image(0, 0, image=photo, anchor=NW)
        self.canvas.image = photo

        if self.frame_count % 120 == 0:
            self.check_mood()

    def resize_frame(self, frame, target_width, target_height):
        height, width = frame.shape[:2]
        aspect_ratio = width / height

        if aspect_ratio > 1:  
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:  
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

        resized_frame = resize(frame, (new_width, new_height))

        pad_top = (target_height - new_height) // 2
        pad_bottom = target_height - new_height - pad_top
        pad_left = (target_width - new_width) // 2
        pad_right = target_width - new_width - pad_left

        padded_frame = copyMakeBorder(resized_frame, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, value=(180, 229, 255))

        return padded_frame

    def check_mood(self):
        max_emotion = max(self.emotion_counts, key=self.emotion_counts.get)
        if self.emotion_counts[max_emotion] == 0:
            max_emotion = ""

        encouragement_data = load_encouragement_messages()

        if max_emotion in encouragement_data:
            message = choice(encouragement_data[max_emotion])
            text_string = f"You're feeling {max_emotion}. Here's a word of encouragement: {message}"
        else:
            text_string = "Stay positive!"

        self.encouragement_text.config(text=text_string)
        self.emotion_counts = {emotion: 0 for emotion in self.emotion_counts}

    def cv2_to_tkinter(self, cv_image):
        rgb_image = cvtColor(cv_image, COLOR_BGR2RGB)
        pil_image = fromarray(rgb_image)
        return PhotoImage(pil_image)

    def exit_app(self, event=None):
        self.cap.release()
        self.root.quit()

def main():
    root = Tk()
    app = BrightSide(root)
    root.mainloop()

if __name__ == "__main__":
    main()
