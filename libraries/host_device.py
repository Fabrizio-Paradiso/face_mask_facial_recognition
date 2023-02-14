from libraries.common import get_ip_address_pc, get_ip_address_raspberry
import os


class HostDevice:
    def __init__(self) -> None:
        self.host_device = self.set_host_device()

    def set_host_device(self) -> None:
        """
        Set host device based on environment variables

        Args:
            None
        Return:
            None
        """
        if os.getenv("HOST_DEVICE") == "pc":
            from libraries.camera_opencv import Camera

            self.ip_address = get_ip_address_pc()
            self.cam: Camera() = Camera()

        if os.getenv("HOST_DEVICE") == "raspberry":
            from libraries.camera_raspberry import Camera

            self.ip_address = get_ip_address_raspberry()
            self.cam: Camera() = Camera()
