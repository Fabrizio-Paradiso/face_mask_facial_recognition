import cv2

class Camera():
    def __init__(self) -> None:
        self.frame = cv2.VideoCapture(0)

    def get_frame(self):
        """
        This function returns the frame captured by Web Cam in numpy format

        Args:
            self: Camera(object)
        Return:
            frame (numpy.ndarray)
        """
        return self.frame.read()[1]

    def frame_release(self) -> None:
        """
        This function indicates that the thread should be stopped

        Args:
            self: Camera(object)
        Return:
            None
        """
        self.frame.release()