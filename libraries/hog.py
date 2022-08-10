from imutils import paths
from sklearn.decomposition import PCA
from skimage.exposure import rescale_intensity
from skimage.feature import hog
import cv2
import joblib
import numpy
import os
import pandas
import sklearn.svm

class Hog:
    def __init__(self, block_norm: str = "L2", cells_per_block: tuple = (2,2), feature_vector: bool = True, orientations: int = 9, pixels_per_cell: tuple = (2,2)) -> None:
        self.BLOCK_NORM = block_norm
        self.CELLS_PER_BLOCK = cells_per_block
        self.DATA = []
        self.FEATURE_VECTOR = feature_vector
        self.LABELS = []
        self.ORIENTATIONS = orientations
        self.PIXELS_PER_CELL = pixels_per_cell

    def create_data_csv(self, dir_path: str, save_path:str) -> None:
        """
        Create DataFrame based on HOG features

        Args:
            dir_path (str): Directory where images are located
            save_path (str): Filename for .csv
        Return:
            prediction: Predicted label
            hog_features: Features obtained with HOG
        """
        df = pandas.DataFrame()
        data, labels = self._get_data_labels(dir_path)
        X_train = pandas.DataFrame(data)
        y_train = pandas.Series(labels)
        df = pandas.concat([X_train, y_train], axis=1, ignore_index=True)
        df.to_csv(save_path,index=False)

    def extract_features(self, gray_image: numpy.ndarray) -> tuple:
        """
        Extract features in ROI using HOG

        Args:
            gray_image (numpy.ndarray): Grayscale ROI image
        Return:
            hog_features: Features obtained with HOG
        """
        hog_features = hog(gray_image, self.ORIENTATIONS, self.PIXELS_PER_CELL, self.CELLS_PER_BLOCK, self.BLOCK_NORM)
        hog_features = hog_features.reshape(1,-1)
        return hog_features

    def _get_data_labels(self, training_path: str) -> tuple:
        """
        Get data and labels from HOG

        Args:
            training_path (str): Path where training images are located
        Return:
            Data and labels stored in an array and extracted for each image in training path
        """
        for image_path in paths.list_images(training_path):
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hog_ft = hog(gray, self.ORIENTATIONS, self.PIXELS_PER_CELL, self.CELLS_PER_BLOCK, self.BLOCK_NORM)
            self.DATA.append(hog_ft)
            self.LABELS.append(image_path.split(os.path.sep)[-2])
            print(image_path)
        return self.DATA, self.LABELS

    @staticmethod
    def load_svm_model (model_path: str) -> sklearn.svm._classes.LinearSVC:
        """
        Load SVM model

        Args:
            model_path (str): Path where SVM model is located
        Return:
            Linear SVM model
        """
        model = joblib.load(model_path)
        return model

    @staticmethod
    def visualize_hog_image(image_path: str):
        """"
        Visualize HOG image in a window in case it is necessary to adjust HOG parameters

        Args:
            image_path (str): Path where image is located
        Return:
            None
        """
        image = cv2.imread(os.path.normpath(image_path))
        image_resized = cv2.resize(src = image, dsize=(850,955))
        image_filtered = cv2.GaussianBlur(image_resized, (7,7), cv2.BORDER_DEFAULT)
        gray_image = cv2.cvtColor(image_filtered, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, (128, 144))
        (_, hog_image) = hog(gray_image,
                            orientations=9,
                            pixels_per_cell=(2, 2),
	                        cells_per_block=(2, 2),
                            block_norm="L1",
	                        visualize=True)
        hog_image = rescale_intensity(hog_image, out_range=(0, 255))
        hog_image = hog_image.astype("uint8")
        cv2.imshow("HOG Image", hog_image)
        cv2.waitKey(0)