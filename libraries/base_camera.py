import time
import threading

try:
    from greenlet import getcurrent as get_ident
except ImportError:
    try:
        from thread import get_ident
    except ImportError:
        from _thread import get_ident


class CameraEvent(object):
    """
    An Event-like class that signals all active clients when a new frame is
    available.
    """

    def __init__(self) -> None:
        self.events = {}

    def wait(self) -> None:
        """
        This function is invoked from each new client's thread to wait for the next frame,
        each entry has two elements, a threading.Event() and a timestap

        Args:
            None
        Return:
            None: Only waits for the next frame
        """
        ident = get_ident()
        if ident not in self.events:
            self.events[ident] = [threading.Event(), time.time()]
        return self.events[ident][0].wait()

    def set(self) -> None:
        """
        This function is invoked by the camera thread when a new frame is available,
        if client's event is not set it will set, but if it is set, it means the client
        did not process a previous frame, so if the event stays set for more than 5 second,
        it will asusume the client is gone and remove it

        Args:
            None
        Return:
            None
        """
        now = time.time()
        remove = None
        for ident, event in self.events.items():
            if not event[0].isSet():
                event[0].set()
                event[1] = now
            else:
                if now - event[1] > 5:
                    remove = ident
        if remove:
            del self.events[remove]

    def clear(self) -> None:
        """
        This function is invoked from each client's thread after a frame was processed

        Args:
            None
        Return:
            None
        """
        self.events[get_ident()][0].clear()


class BaseCamera(object):
    """
    Parameters definition from BaseCamera class

    Parameters:
        event: Instance of CameraEvent class
        frame: Current frame stored by background thread
        last_access: Last time client access to the camera
        thread: Background thread that reads frames from camera
    """

    event = CameraEvent()
    frame = None
    last_access = 0
    thread = None

    def __init__(self):
        """
        This function sets last time client access, starts the background camera thread if it isn't running yet
        and wait until first frame is available

        Args:
            None
        Return:
            None
        """
        if BaseCamera.thread is None:
            BaseCamera.last_access = time.time()
            BaseCamera.thread = threading.Thread(target=self._thread)
            BaseCamera.thread.start()
            BaseCamera.event.wait()

    @staticmethod
    def frames():
        """ "
        Generator that returns frames from the camera, must be implemented by camera subclasses

        Args:
            None
        Return:
            None
        """
        raise RuntimeError("Must be implemented by subclasses.")

    def get_frame(self):
        """
        This function waits for a signal from the camera thread and
        returns the current camera frame

        Args:
            None
        Return:
            None
        """
        BaseCamera.last_access = time.time()
        BaseCamera.event.wait()
        BaseCamera.event.clear()
        return BaseCamera.frame

    @classmethod
    def _thread(cls):
        """
        Handler camera background thread which sets get frame from camera subclass and
        stop camera thread if there is an inactivity of ten seconds

        Args:
            None
        Return:
            None
        """
        print("Starting camera thread.")
        frames_iterator = cls.frames()
        for frame in frames_iterator:
            BaseCamera.frame = frame
            BaseCamera.event.set()
            time.sleep(0)
            if time.time() - BaseCamera.last_access > 10:
                frames_iterator.close()
                print("Stopping camera thread due to inactivity.")
                break
        BaseCamera.thread = None
