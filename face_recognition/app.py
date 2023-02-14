import os
import sys
import cv2

root = os.path.abspath(".")
sys.path.insert(1, root)

from libraries.recognition import Recognition
from libraries.common import add_match_to_frame, create_bounding_box

recognition: Recognition = Recognition()
cam = cv2.VideoCapture(0)


def main():
    while True:
        _, frame = cam.read()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = recognition.face_recognition.FACE_CASCADE.detectMultiScale(
            gray_image, scaleFactor=1.3, minNeighbors=4
        )
        try:
            crop_image = recognition.recognition_crop(faces, gray_image)
            hog_features = recognition.hog.extract_features(crop_image)
            prediction = recognition.recognition_prediction(hog_features)
            frame = add_match_to_frame(frame, prediction)
            create_bounding_box(faces, frame, [0], prediction)
        except:
            pass

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.frame_release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
