from libraries.face_mask import FaceMask
from libraries.hog import Hog
from libraries.data import Data
import os
import numpy


class Mask:
    def __init__(self) -> None:
        self.svm_model_path = os.path.normpath(f"models/face_mask.xml")
        self.svm_model_csv = "models/mask.csv"
        self.face_mask: FaceMask = FaceMask()
        self.hog: Hog = Hog()
        self.model = self.hog.load_svm_model(self.svm_model_path)
        self.pca_decomposition = 2
        self.data = Data(self.svm_model_csv, self.pca_decomposition)

    def mask_crop(self, faces: tuple, gray_image: numpy.ndarray) -> numpy.ndarray:
        """
        Get cropped image in Face-Mask ROI

        Args:
            faces (tuple): Coordinates, width and height from faces detected by Haar Cascade Frontal Face
            gray_image (numpy.ndarray): Face image in grayscale
        Return:
            Cropped image in Face-Mask ROI resized for prediction
        """
        coordinates_roi = self.face_mask.geometrical_face_model_roi_coordinates(faces)
        crop_image = self.face_mask.resize_crop_roi(coordinates_roi, gray_image)
        return crop_image

    def mask_prediction(self, hog_features: tuple) -> int:
        """
        Face-Mask prediction in Real-Time

        Args:
            hog_features: Features obtained with HOG
        Return:
            prediction: Prediction number based on Face-Mask Detection model
        """
        scaled_features = self.data.scaler.transform(hog_features)
        pca_features = self.data.pca.transform(scaled_features)
        prediction = self.model.predict(pca_features)
        return prediction
