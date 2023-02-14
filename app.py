from flask import Flask, render_template, Response
from libraries.host_device import HostDevice
from libraries.soup import Soup
from process import application_process

import cv2
import webbrowser

soup: Soup = Soup()
app = Flask(__name__, static_url_path="/static")
host_device = HostDevice()


def generator(cam):
    while True:
        frame = cam.get_frame()
        frame_stream = application_process(frame)
        (flag, encoded_image) = cv2.imencode(".jpg", frame_stream)
        if not flag:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encoded_image) + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generator(host_device.cam), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    soup.insert_ip_address(host_device.ip_address)
    webbrowser.open_new(f"http://{host_device.ip_address}:5000/")
    app.run(host=host_device.ip_address)
