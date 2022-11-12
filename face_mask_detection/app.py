import numpy
import os
import sys
import cv2
import pandas as pd

root = os.path.abspath(".")
sys.path.insert(1, root)
from libraries.common import header_face_mask
from libraries.face_mask import FaceMask
from libraries.hog import Hog
from libraries.webcam import Camera
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Mask:
    def __init__(self) -> None:
        self.svm_model_path = os.path.normpath(f"models/face_mask.xml")
        self.face_mask: FaceMask = FaceMask()
        self.hog: Hog = Hog()

        self.model = self.hog.load_svm_model(self.svm_model_path)

        self.mask = pd.read_csv("models/mask.csv")
        self.mask.columns = [*self.mask.columns[:-1], "Target Class"]

        self.scaler = StandardScaler()
        self.scaler.fit(self.mask.drop("Target Class", axis=1).values)
        self.scaled_data = self.scaler.transform(
            self.mask.drop("Target Class", axis=1).values
        )

        self.pca = PCA(n_components=2)
        self.pca.fit(self.scaled_data)

    def mask_crop(self, faces: tuple, gray_image: numpy.ndarray) -> numpy.ndarray:
        coordinates_roi = self.face_mask.geometrical_face_model_roi_coordinates(faces)
        crop_image = self.face_mask.resize_crop_roi(coordinates_roi, gray_image)
        return crop_image

    def mask_prediction(self, hog_features: tuple) -> int:
        scaled_features = self.scaler.transform(hog_features)
        pca_features = self.pca.transform(scaled_features)
        prediction = self.model.predict(pca_features)
        return prediction


mask: Mask = Mask()
cam: Camera = Camera()


def main():
    while True:
        frame = cam.get_frame()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = mask.face_mask.FACE_CASCADE.detectMultiScale(
            gray_image, scaleFactor=1.3, minNeighbors=4
        )
        try:
            crop_image = mask.mask_crop(faces, gray_image)
            hog_features = mask.hog.extract_features(crop_image)
            prediction = mask.mask_prediction(hog_features)
            header_face_mask(faces, frame, prediction)
        except:
            pass

        cv2.imshow("Face-Mask Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.frame_release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
