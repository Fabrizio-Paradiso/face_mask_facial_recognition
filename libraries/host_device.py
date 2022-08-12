from libraries.camera_opencv import Camera
from libraries.common import get_ip_address_pc, get_ip_address_raspberry
import os

class HostDevice():
    def __init__(self) -> None:
        self.host_device = self.set_host_device()

    def set_host_device(self)-> None:
        '''
        Set host device based on environment variables

        Args:
            None
        Return:
            None
        '''
        if os.getenv("HOST_DEVICE")=="pc":
            self.ip_address = get_ip_address_pc()
            self.cam : Camera() = Camera()
        if os.getenv("HOST_DEVICE")=="raspberry":
            from libraries.camera_raspberry import RaspberryCamera
            self.ip_address = get_ip_address_raspberry()
            self.cam : RaspberryCamera = RaspberryCamera()