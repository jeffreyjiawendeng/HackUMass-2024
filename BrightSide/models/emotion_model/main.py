import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
from utils import load_model, preprocess_face, get_emotion_prediction
from keras.preprocessing import image # type: ignore

def main():
    model = load_model()
    face_haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    if face_haar_cascade.empty():
        print("Error: Haar Cascade file not loaded properly!")

    cap = cv2.VideoCapture(0)

    while True:
        ret, test_img = cap.read()
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), thickness=7)
            img_pixels = preprocess_face(gray_img, (x, y, w, h))
            predicted_emotion = get_emotion_prediction(model, img_pixels)

            if predicted_emotion != "happy":
                predicted_emotion = "not happy"

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow("Facial Emotion Analysis", resized_img)

        if cv2.waitKey(10) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
