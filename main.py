from flask import Flask, render_template, Response
from libraries.common import add_date_to_frame, add_match_to_frame, create_bounding_box, get_ip_address_pc, get_ip_address_raspberry
from libraries.hog import Hog
from libraries.face_mask import FaceMask
from libraries.face_recognition import FaceRecognition
from libraries.soup import Soup
from libraries.webcam import Camera
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cv2
import numpy
import os
import pandas as pd
import sys
import webbrowser

class Mask():
    def __init__(self) -> None:
        self.svm_model_path = os.path.normpath(f"models/face_mask.xml")
        self.face_mask: FaceMask = FaceMask ()
        self.hog: Hog = Hog()

        self.model =  self.hog.load_svm_model (self.svm_model_path)

        self.mask = pd.read_csv('models/mask.csv')
        self.mask.columns = [*self.mask.columns[:-1], 'Target Class']

        self.scaler = StandardScaler()
        self.scaler.fit(self.mask.drop('Target Class', axis=1).values)
        self.scaled_data = self.scaler.transform(self.mask.drop('Target Class',axis=1).values)

        self.pca = PCA(n_components=2)
        self.pca.fit(self.scaled_data)

    def mask_crop (self, faces: tuple, gray_image: numpy.ndarray) -> numpy.ndarray:
        coordinates_roi = self.face_mask.geometrical_face_model_roi_coordinates(faces)
        crop_image = self.face_mask.resize_crop_roi(coordinates_roi, gray_image)
        return crop_image

    def mask_prediction(self, hog_features: tuple) -> int:
        scaled_features = self.scaler.transform(hog_features)
        pca_features = self.pca.transform(scaled_features)
        prediction = self.model.predict(pca_features)
        return prediction

class Recognition():
    def __init__(self) -> None:
        self.svm_model_path = os.path.normpath(f"models/face_recognition.xml")

        self.face_recognition: FaceRecognition = FaceRecognition()
        self.hog: Hog = Hog()
        self.model = self.hog.load_svm_model (self.svm_model_path)

        self.recognition = pd.read_csv('models/recognition.csv')
        self.recognition.columns = [*self.recognition.columns[:-1], 'Target Class']

        self.scaler = StandardScaler()
        self.scaler.fit(self.recognition.drop('Target Class', axis=1).values)
        self.scaled_data = self.scaler.transform(self.recognition.drop('Target Class',axis=1).values)

        self.pca = PCA(n_components=66)
        self.pca.fit(self.scaled_data)

    def recognition_crop(self, faces: tuple, gray_image: numpy.ndarray) -> numpy.ndarray:
        coordinates_roi = self.face_recognition.get_face_recognition_roi_coordinates(faces)
        crop_image = self.face_recognition.resize_crop_roi(coordinates_roi, gray_image)
        return crop_image

    def recognition_prediction(self, hog_features: tuple) -> int:
        scaled_features = self.scaler.transform(hog_features)
        pca_features = self.pca.transform(scaled_features)
        prediction = self.model.predict(pca_features)
        return prediction

mask: Mask = Mask()
recognition: Recognition = Recognition()
cam: Camera = Camera()
face_recognition: FaceRecognition = FaceRecognition()
face_mask: FaceMask = FaceMask()
hog: Hog = Hog()
soup: Soup = Soup()
app = Flask(__name__, static_url_path="/static")

ip_address = None

if sys.argv[1]=="pc":
    from libraries.camera_opencv import Camera
    ip_address = get_ip_address_pc()
if sys.argv[1]=="raspberry":
    from libraries.camera_raspberry import Camera
    ip_address = get_ip_address_raspberry()

svm_model =  hog.load_svm_model(os.path.normpath(f"models/face_mask_model.xml"))
svm_model_multiclass = hog.load_svm_model(os.path.normpath(f"models/face_recognition_model.xml"))

def generator(cam):
    while True:
        frame = cam.get_frame()
        add_date_to_frame(frame)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_recognition.FACE_CASCADE.detectMultiScale(gray_image,
                                                               scaleFactor = 1.03,
                                                               minNeighbors = 4)
        try:
            mask_crop = mask.mask_crop(faces, gray_image)
            recognition_crop = recognition.recognition_crop(faces, gray_image)
            mask_features = mask.hog.extract_features(mask_crop)
            recognition_features = recognition.hog.extract_features(recognition_crop)
            mask_prediction = mask.mask_prediction(mask_features)
            recognition_prediction = recognition.recognition_prediction(recognition_features)
            add_match_to_frame(frame, recognition_prediction)
            create_bounding_box(faces, frame, mask_prediction, recognition_prediction)
        except:
            pass
        (flag, encoded_image) = cv2.imencode(".jpg", frame)
        if not flag:
            continue
        yield(b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encoded_image) + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generator(Camera()), mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    soup.insert_ip_address(ip_address)
    webbrowser.open_new(f"http://{ip_address}:5000/")
    app.run(host=ip_address)