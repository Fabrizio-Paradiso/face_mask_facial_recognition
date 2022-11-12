from config import (
    BLUR_FILTER_SIZE,
    CROP_SIZE,
    MAX_FACE_HEIGHT,
    MIN_FACE_HEIGHT,
    MIN_NEIGHBORS_TRAIN,
    MIN_SIZE_TRAIN,
    RESIZE_SIZE,
    SCALE_FACTOR_TRAIN,
)
from .hog import Hog
from imutils import paths
import cv2
import math
import numpy
import os
import sklearn.svm


class FaceMask:
    def __init__(self) -> None:
        self.FACE_CASCADE = cv2.CascadeClassifier(
            (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        )
        self.CATEGORIES = ["mask", "non-mask"]
        self.hog: Hog = Hog()

    @staticmethod
    def _approximated_roi_coordinates(image: numpy.ndarray) -> tuple:
        """
        Get approximated ROI coordinates based on image size since haar cascade can not found face in image (only used in training)

        Args:
            image (numpy.ndarray): Train image of person
        Return:
            Coordinates of Face-Mask ROI
        """
        h, w, _ = image.shape
        x0 = math.floor(w / 8)
        y0 = math.floor(h * 6 / 10)
        x1 = math.ceil(w * 7 / 8)
        y1 = math.ceil(h)
        return [x0, y0, x1, y1]

    def crop_dataset_face_mask(self, dataset_dir: str, export_dir: str) -> None:
        """
        Crop Dataset in Face-Mask ROI and export in desired path

        Args:
            dataset_dir (str): Directory where original images are located
            export_dir (str): Directory where cropped images are exported
        Return:
            None
        """
        for category in self.CATEGORIES:
            dir = os.path.join(dataset_dir, category)
            self.export_roi_in_directory(dir, export_dir, prefix_name=f"roi_")

    def export_resize_image_in_directory(
        self, dir: str, export_dir: str, prefix_name: str
    ) -> None:
        """
        Resize Images in ROI and export in desired path

        Args:
            dir (str): Directory of face or mask images
            export_dir (str): Export directory where cropped images are exported
            prefix_name (str): Prefix to add for all images in directory
        Return:
            None
        """
        for image_path in paths.list_images(dir):
            image = cv2.imread(image_path)
            resize_image = cv2.resize(src=image, dsize=(32, 36))
            filename = str(os.path.basename(image_path.split(".")[0]))
            print(os.path.join(export_dir, f"{prefix_name}{filename}.jpg"))
            cv2.imwrite(
                os.path.join(export_dir, f"{prefix_name}{filename}.jpg"), resize_image
            )

    def export_roi_in_directory(
        self, dir: str, export_dir: str, prefix_name: str
    ) -> None:
        """
        Crop Test Images in ROI and export in desired path

        Args:
            dir (str): Directory of face or mask images
            export_dir (str): Export directory where cropped images are exported
            prefix_name (str): Prefix to add for all images in directory
        Return:
            None
        """
        for image_path in paths.list_images(dir):
            image = cv2.imread(image_path)
            crop_image = self.get_crop_based_on_geometrical_face_model(image)
            filename = str(os.path.basename(image_path.split(".")[0]))
            print(os.path.join(export_dir, f"{prefix_name}{filename}.jpg"))
            cv2.imwrite(
                os.path.join(export_dir, f"{prefix_name}{filename}.jpg"), crop_image
            )

    def face_mask_features(self, faces: tuple, gray_image: numpy.ndarray) -> tuple:
        """
        Face-Mask features in Real-Time

        Args:
            faces (tuple): Coordinates, width and height from faces detected by Haar Cascade Frontal Face
            gray_image (numpy.ndarray): Face image in grayscale
            svm_model (sklearn.svm._classes.LinearSVC): Linear SVM Model in charge of Face-Mask Detection
        Return:
            prediction: Prediction label Face-Mask Detection
            hog_features: Features obtained with HOG
        """
        coordinates_roi = self.geometrical_face_model_roi_coordinates(faces)
        crop_image = self.resize_crop_roi(coordinates_roi, gray_image)
        hog_features = self.hog.extract_features(crop_image)
        return hog_features

    @staticmethod
    def geometrical_face_model_roi_coordinates(
        faces: tuple, training_case: bool = False
    ) -> tuple:
        """
        Get ROI coordinates based on Geometrical Face Model (used in training and in Real-Time)

        Args:
            faces (tuple): Coordinates, width and height from faces detected by Haar Cascade Frontal Face
            training_case (bool): Boolean to enable size filtering for faces
        Return:
            Face-Mask ROI coordinates (tuple)
        """
        if training_case:
            faces = faces[0][3] > MIN_FACE_HEIGHT and faces[0][3] < MAX_FACE_HEIGHT

        for (x, y, w, h) in faces:
            x1 = int(w / 8) + x
            y1 = int(h / 2) + y
            x2 = int(w * 7 / 8) + x
            y2 = int(h * 8 / 9) + y

        return [x1, x2, y1, y2]

    def geometrical_face_model_roi_image(self, image: numpy.ndarray) -> numpy.ndarray:
        """
        Get Face-Mask ROI image based on Geometrical Face Model

        Args:
            image (numpy.ndarray): Face image
        Return:
            Cropped image in Face-Mask ROI
        """
        gray = self._preprocessing_image_to_gray(image)
        faces = self.FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR_TRAIN,
            minNeighbors=MIN_NEIGHBORS_TRAIN,
            minSize=MIN_SIZE_TRAIN,
        )

        try:
            coordinates = self.geometrical_face_model_roi_coordinates(
                faces, training_case=True
            )
        except:
            coordinates = self._approximated_roi_coordinates(image)
        finally:
            return self.resize_crop_roi(coordinates, image)

    def get_crop_based_on_geometrical_face_model(
        self, image: numpy.ndarray
    ) -> numpy.ndarray:
        height, width, _ = image.shape
        xo1 = int(width / 8)
        yo1 = int(height * 1 / 2)
        xo2 = int(width * 7 / 8)
        yo2 = int(height * 8 / 9)
        crop_image = image[yo1:yo2, xo1:xo2]

        return crop_image

    @staticmethod
    def _preprocessing_image_to_gray(image: numpy.ndarray) -> numpy.ndarray:
        """
        Preprocessing for training images

        Args:
            image (numpy.ndarray): Original image
        Return:
            Grayscale and resize image
        """
        image_resized = cv2.resize(src=image, dsize=RESIZE_SIZE)
        image_filtered = cv2.GaussianBlur(
            image_resized, BLUR_FILTER_SIZE, cv2.BORDER_DEFAULT
        )
        gray = cv2.cvtColor(image_filtered, cv2.COLOR_BGR2GRAY)
        return gray

    @staticmethod
    def resize_crop_roi(coordinates: tuple, image: numpy.ndarray) -> numpy.ndarray:
        """
        Crop image in ROI and resize

        Args:
            coordinates (tuple): Coordinates Face-Mask Detection ROI
            image (numpy.ndarray): Original image
        Return:
            Cropped and resized image
        """
        crop_image = image[
            coordinates[2] : coordinates[3], coordinates[0] : coordinates[1]
        ]
        crop_resized = cv2.resize(crop_image, dsize=(32, 36))
        return crop_resized
