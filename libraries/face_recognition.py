from config import FINAL_SIZE, XO1_FACTOR, XO2_FACTOR, YO1_FACTOR, YO2_FACTOR
from .hog import Hog
import cv2
import math
import numpy
import os
import sklearn.svm

class FaceRecognition:
    def __init__ (self) -> None:
        self.FACE_CASCADE = cv2.CascadeClassifier((cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
        self.hog: Hog = Hog()

    @classmethod
    def crop_dataset (cls, dataset_dir: str, export_dir: str) -> None:
        """
        Crop Dataset in ROI and export in desired path

        Args:
            dataset_dir (str): Directory where original images are located
            export_dir (str): Directory where cropped images are exported
        Return:
            None
        """
        for name_dir in os.listdir(dataset_dir):
            person_path = os.path.join(dataset_dir, name_dir)
            person_id = person_path.split(os.sep)[-1]
            for image_path in os.listdir(person_path):
                image_rgb = cv2.imread(os.path.join(person_path, image_path))
                h, w, _ = image_rgb.shape
                image_copy = image_rgb.copy()
                coordinates = cls.get_training_face_recognition_roi_coordinates(w, h)
                crop_image = image_copy [coordinates[2]:coordinates[3], coordinates[0]:coordinates[1]]
                cv2.imwrite(os.path.normpath(os.path.join(export_dir, person_id, image_path.split(".")[0]+"_crop.jpg")), crop_image)

    def face_name_recognition(self, faces: tuple, gray_image: numpy.ndarray, svm_model_multiclass: sklearn.svm._classes.LinearSVC) -> tuple:
        """
        Face-Name Recognition in Real-Time

        Args:
            faces (tuple): Coordinates, width and height from faces detected by Haar Cascade Frontal Face
            gray_image (numpy.ndarray): Face image in grayscale
            svm_model (sklearn.svm._classes.LinearSVC): Linear SVM Model in charge of Face-Name Recognition
        Return:
            prediction: Prediction label Face-Name Recognition
            hog_features: Features obtained with HOG
        """
        coordinates_roi = self.get_face_recognition_roi_coordinates(faces)
        crop_image = self.resize_crop_roi(coordinates_roi, gray_image)
        prediction, hog_features = self.hog.extract_features_and_predict(crop_image, svm_model_multiclass)
        return prediction, hog_features

    @staticmethod
    def get_face_recognition_roi_coordinates(faces: tuple) -> tuple:
        """
        Get ROI for Face-Name Recognition in Real-Time

        Args:
            faces (tuple): Coordinates, width and height from faces detected by Haar Cascade Frontal Face
        Return:
            Face-Name Recognition ROI
        """
        for (x, y, width, height) in faces:
            xo1 = math.floor(width*XO1_FACTOR)
            yo1 = math.floor(height*YO1_FACTOR)
            xo2 = math.ceil(width*XO2_FACTOR)
            yo2 = math.ceil(height*YO2_FACTOR)
        return [x+xo1, x+xo2, y+yo1, y+yo2]

    @staticmethod
    def get_training_face_recognition_roi_coordinates(width: int, height: int) -> tuple:
        """
        Get ROI for Face-Name Recognition considering image is cropped in face

        Args:
            width (int): Face bounding box width
            height (int): Face bounding box height
        Return:
            Face-Name Recognition ROI
        """
        xo1 = math.floor(width*XO1_FACTOR)
        yo1 = math.floor(height*YO1_FACTOR)
        xo2 = math.ceil(width*XO2_FACTOR)
        yo2 = math.ceil(height*YO2_FACTOR)
        return [xo1, xo2, yo1, yo2]

    @staticmethod
    def resize_crop_roi (coordinates: tuple, image: numpy.ndarray) -> numpy.ndarray:
        """
        Crop image in ROI and resize

        Args:
            coordinates (tuple): Coordinates Face-Name Recognition ROI
            image (numpy.ndarray): Original image
        Return:
            Cropped and resized image
        """
        crop_image = image[coordinates[2]:coordinates[3], coordinates[0]:coordinates[1]]
        crop_resized = cv2.resize (crop_image, dsize=FINAL_SIZE)
        return crop_resized