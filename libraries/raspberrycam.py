
from datetime import datetime
from imutils.video.pivideostream import PiVideoStream
import cv2
import numpy
import time

class Camera(object):
    def __init__(self, file_type: str = ".jpg", flip_vertical: bool = False, frame_rate: int = 32, photo_string: str = "Screenshot", resolution: tuple = (320, 240)) -> None:
        self.file_type = file_type
        self.flip_vertical = flip_vertical
        self.frame_rate = frame_rate
        self.resolution = resolution
        self.photo_string = photo_string
        self.video_stream = PiVideoStream(self.resolution, self.frame_rate).start()
        time.sleep(2.0)

    def __del__(self) -> None:
        """
        This function indicates that the thread should be stopped

        Args:
            self: Camera(object)
        Return:
            None
        """
        self.video_stream.stop()

    def flip_vertically_if_needed(self, frame: numpy.ndarray):
        """
        This function could flip vertically the frame captured by Raspberry Cam

        Args:
            self: Camera(object)
            frame (numpy.ndarray): Frame captured by Raspberry Cam
        Return:
            frame (numpy.ndarray)
        """
        if self.flip_vertical:
            return numpy.flip(frame, 0)
        return frame

    def get_frame(self):
        """
        This function returns the frame captured by Raspberry Cam in numpy format

        Args:
            self: Camera(object)
        Return:
            frame (numpy.ndarray)
        """
        frame = self.flip_vertically_if_needed(self.video_stream.read())
        self.previous_frame = frame.copy()
        return frame

    def take_picture(self):
        """
        This function takes a picture from the frame captured by Raspberry Cam

        Args:
            self: Camera(object)
        Return:
            None
        """
        today_date = datetime.now().strftime("%m%d%Y-%H%M%S")
        cv2.imwrite(f"{self.photo_string}_{today_date}{self.file_type}", self.get_frame())
