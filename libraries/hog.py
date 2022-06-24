from imutils import paths
from skimage.exposure import rescale_intensity
from skimage.feature import hog
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import cv2
import joblib
import numpy
import os
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

    def extract_features_and_predict(self, gray_image: numpy.ndarray, svm_model: sklearn.svm._classes.LinearSVC) -> tuple:
        """
        Get prediction from trained model based on features

        Args:
            gray_image (numpy.ndarray): Grayscale ROI image
            model (sklearn.svm._classes.LinearSVC): Linear SVM model based on HOG features
        Return:
            prediction: Predicted label
            hog_features: Features obtained with HOG
        """
        hog_features = hog(gray_image, self.ORIENTATIONS, self.PIXELS_PER_CELL, self.CELLS_PER_BLOCK, self.BLOCK_NORM)
        hog_features = hog_features.reshape(1,-1)
        prediction = svm_model.predict(hog_features)
        return prediction, hog_features

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

    def get_error_linear_svm_model(self, svm_model: sklearn.svm._classes.LinearSVC, test_images_path: str) -> float:
        """
        Test a Linear SVM model based on HOG features with images

        Args:
            model (sklearn.svm._classes.LinearSVC): Linear SVM model for testing
            test_images_path (str): Path where testing images are located
        Return:
            Model error based on negative and positive cases over the total
        """
        hits = 0
        mistakes = 0
        total = len(os.listdir(test_images_path))
        for image_path in paths.list_images(test_images_path):
            label = os.path.basename(image_path).split("_")[0]
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction, _ = self.extract_features_and_predict(gray_image, svm_model)
            if(label == prediction[0]):
                hits += 1
            elif (label != prediction[0]):
                mistakes += 1
            print(f"Prediction:{prediction[0]} Label:{label} Hits:{hits} Mistakes:{mistakes} Total:{total}")
        method_error = (mistakes/total)*100
        return method_error

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

    def train_multi_class_svm_linear_model_and_save(self, c: float, svm_model_name: str, training_path: str) -> None:
        """
        Train a multi class Linear SVM model

        Args:
            c (float): SVM optimization parameter related with how much it wants to avoid misclassifying each training example
            svm_model_name (str): Filename for SVM multiclass model
            training_path (str): Path where training images are located
        Return:
            None
        """
        data, labels = self._get_data_labels(training_path)
        model = OneVsRestClassifier(LinearSVC(C=c, random_state=42))
        model.fit(data, labels)
        joblib.dump(model, svm_model_name)

    def train_svm_linear_model_and_save(self, c: float, svm_model_name: str, training_path: str) -> None:
        """
        Train a Linear SVM model

        Args:
            c (float): SVM optimization parameter related with how much it wants to avoid misclassifying each training example
            svm_model_name (str): Filename for SVM model
            training_path (str): Path where training images are located
        Return:
            None
        """
        data, labels = self._get_data_labels(training_path)
        model = LinearSVC(C=c, random_state=42)
        model.fit(data, labels)
        joblib.dump(model, svm_model_name)

    def visualize_hog_image(image_path: str):
        """"
        Visualize HOG image in a window in case it is necessary to adjust HOG parameters

        Args:
            image_path (str): Path where image is located
        Return:
            None
        """
        image = cv2.imread(os.path.normpath("static/assets/images/fabrizio.png"))
        image_resized = cv2.resize(src = image, dsize=(850,955))
        image_filtered = cv2.GaussianBlur(image_resized, (7,7), cv2.BORDER_DEFAULT)
        gray_image = cv2.cvtColor(image_filtered, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, (128, 144))
        (_, hog_image) = hog(gray_image,
                            orientations=9,
                            pixels_per_cell=(8, 8),
	                        cells_per_block=(2, 2),
                            block_norm="L1",
	                        visualize=True)
        hog_image = rescale_intensity(hog_image, out_range=(0, 255))
        hog_image = hog_image.astype("uint8")
        cv2.imshow("HOG Image", hog_image)
        cv2.waitKey(0)