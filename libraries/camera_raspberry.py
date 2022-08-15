from libraries.base_camera import BaseCamera
from PIL import Image
import io
import numpy
import cv2
import picamera
import time


class Camera(BaseCamera):
    @staticmethod
    def frames() -> numpy.ndarray:
        """
        This function is in charge of return the current frame from the Rasberry Cam.
        First of all, let camera warm up and then set capture_continuous()

        Args:
            None
        Return:
            frame (numpy.ndarray): Current frame
        """
        with picamera.PiCamera() as camera:
            time.sleep(2)
            stream = io.BytesIO()
            for _ in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
                stream.seek(0)

                # Convert bytes frame to PIL type image
                stream_pil = Image.open(stream)

                # Convert PIL image to numpy array BGR
                stream_numpy_bgr = numpy.asarray(stream_pil)

                # Convert numpy array from BGR to RGB
                stream_numpy_rgb = cv2.cvtColor(stream_numpy_bgr)

                # Flip frame vertically for Raspberry Cam
                stream_flip = numpy.flip(stream_numpy_rgb, axis=0)
                yield stream_flip
                stream.seek(0)
                stream.truncate()
