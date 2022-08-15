from libraries.base_camera import BaseCamera
import cv2
import os
import numpy

class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source: int) -> None:
        """
        This function set video source for OpenCV Cam

        Args:
            None
        Return:
            None
        """
        Camera.video_source = source

    @staticmethod
    def frames() -> numpy.ndarray:
        """
        This function is in charge of return the current frame from the OpenCV cam.

        Args:
            None
        Return:
            frame (numpy.ndarray): Current frame
        """
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, frame = camera.read()
            yield frame
