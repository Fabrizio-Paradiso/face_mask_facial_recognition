from flask import Flask, render_template, Response
from libraries.common import add_date_to_frame, add_match_to_frame, create_bounding_box
from libraries.host_device import HostDevice
from libraries.hog import Hog
from libraries.face_mask import FaceMask
from libraries.face_recognition import FaceRecognition
from libraries.mask import Mask
from libraries.recognition import Recognition
from libraries.soup import Soup
from libraries.webcam import Camera

import cv2
import webbrowser

mask: Mask = Mask()
recognition: Recognition = Recognition()
cam: Camera = Camera()
face_recognition: FaceRecognition() = FaceRecognition()
face_mask: FaceMask() = FaceMask()
hog: Hog = Hog()
soup: Soup = Soup()
app = Flask(__name__, static_url_path="/static")
host_device = HostDevice()

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
    return Response(generator(host_device.cam), mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    soup.insert_ip_address(host_device.ip_address)
    webbrowser.open_new(f"http://{host_device.ip_address}:5000/")
    app.run(host=host_device.ip_address)
