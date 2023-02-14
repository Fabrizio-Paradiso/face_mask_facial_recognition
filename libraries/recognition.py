from collections import Counter
from libraries.face_recognition import FaceRecognition
from libraries.hog import Hog
from libraries.data import Data
import os
import numpy


class Recognition:
    def __init__(self) -> None:
        self.svm_model_path = os.path.normpath(f"models/face_recognition.xml")
        self.svm_model_csv = f"models/recognition.csv"
        self.face_recognition: FaceRecognition = FaceRecognition()
        self.hog: Hog = Hog()
        self.model = self.hog.load_svm_model(self.svm_model_path)
        self.pca_decomposition = 66
        self.threshold_recognition = -0.1
        self.data = Data(self.svm_model_csv, self.pca_decomposition)

    def recognition_crop(
        self, faces: tuple, gray_image: numpy.ndarray
    ) -> numpy.ndarray:
        """
        Get cropped image in Face-Recognition ROI

        Args:
            faces (tuple): Coordinates, width and height from faces detected by Haar Cascade Frontal Face
            gray_image (numpy.ndarray): Face image in grayscale
        Return:
            Cropped image in Face-Recognition ROI resized for prediction
        """
        coordinates_roi = self.face_recognition.get_face_recognition_roi_coordinates(
            faces
        )
        crop_image = self.face_recognition.resize_crop_roi(coordinates_roi, gray_image)
        return crop_image

    @staticmethod
    def __is_not_a_unique_true(iterable: numpy.array) -> bool:
        """

        Args:
            iterable (numpy.array): _description_

        Returns:
            bool: _description_
        """
        return numpy.count_nonzero(iterable) != 1

    def is_unknown_recognition(self, pca_features: tuple) -> bool:
        """
        Check if recognition from OneVsRest classifier is unknown

        Args:
            pca_features (tuple): Features obtained with PCA
        Returns:
            bool: True or False
        """
        decision_array = self.model.decision_function(pca_features)
        return self.__is_not_a_unique_true(
            (decision_array > self.threshold_recognition)[0]
        )

    def recognition_prediction(self, hog_features: tuple) -> tuple:
        """
        Face-Recognition prediction in Real-Time

        Args:
            hog_features (tuple): Features obtained with HOG
        Return:
            prediction (int) : Prediction number based on Face-Recognition model
            is_unknown (bool): True or False depends on decision function
        """
        scaled_features = self.data.scaler.transform(hog_features)
        pca_features = self.data.pca.transform(scaled_features)
        prediction = self.model.predict(pca_features)
        is_unknown = self.is_unknown_recognition(pca_features)
        return prediction, is_unknown
