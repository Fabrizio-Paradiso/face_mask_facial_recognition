from picamera.array import PiRGBArray
from picamera import PiCamera
from libraries.base_camera import BaseCamera

class Camera(BaseCamera):
    @staticmethod
    def frames():
        camera = PiCamera()
        camera.resolution = (640, 480)
        camera.framerate = 32
        rawCapture = PiRGBArray(camera, size=(640, 480))

        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            image = frame.array
            yield image
            rawCapture.truncate(0)

