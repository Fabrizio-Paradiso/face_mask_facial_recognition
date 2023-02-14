import os
import sys
import cv2

root = os.path.abspath(".")
sys.path.insert(1, root)
from libraries.recognition import Recognition

recognition: Recognition = Recognition()
image_path = os.path.normpath(
    os.path.join(
        root,
        r"C:\Users\fabri\OneDrive\Escritorio\face_mask_facial_recognition\images\test\fabrizio.jpg",
    )
)


def main():
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = recognition.face_recognition.FACE_CASCADE.detectMultiScale(
        gray_image, scaleFactor=1.03, minNeighbors=4
    )
    recognition_crop = recognition.recognition_crop(faces, gray_image)
    recognition_features = recognition.hog.extract_features(recognition_crop)
    recognition_prediction = recognition.recognition_prediction(recognition_features)
    print(f"Recognition Prediction: {recognition_prediction}")


if __name__ == "__main__":
    main()
