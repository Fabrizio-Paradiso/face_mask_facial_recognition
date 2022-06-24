from flask import Flask, render_template, Response
from libraries.common import add_date_to_frame, add_match_to_frame, create_bounding_box, get_ip_address_pc, get_ip_address_raspberry
from libraries.hog import Hog
from libraries.face_mask import FaceMask
from libraries.face_recognition import FaceRecognition
from libraries.soup import Soup
import cv2
import os
import sys
import webbrowser

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

def generator(camera):
    while True:
        frame = camera.get_frame()
        add_date_to_frame(frame)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_recognition.FACE_CASCADE.detectMultiScale(gray_image,
                                                               scaleFactor = 1.03,
                                                               minNeighbors = 4)
        try:
            recognition_hog, _ = face_recognition.face_name_recognition(faces, gray_image, svm_model_multiclass)
            detection_hog, mask_features = face_mask.face_mask_detection(faces, gray_image, svm_model)
            add_match_to_frame(frame, recognition_hog)
            create_bounding_box(detection_hog, faces, frame, mask_features, recognition_hog, svm_model)
        except:
            pass
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        yield(b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

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


