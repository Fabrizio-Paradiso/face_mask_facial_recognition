from asyncio.log import logger
from libraries.common import add_date_to_frame, add_match_to_frame, create_bounding_box
from libraries.hog import Hog
from libraries.face_mask import FaceMask
from libraries.face_recognition import FaceRecognition
from libraries.mask import Mask
from libraries.recognition import Recognition

import cv2
import numpy

mask: Mask = Mask()
recognition: Recognition = Recognition()
face_recognition: FaceRecognition = FaceRecognition()
face_mask: FaceMask = FaceMask()
hog: Hog = Hog()

def application_process(frame: numpy.ndarray) -> None:
    add_date_to_frame(frame)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_recognition.FACE_CASCADE.detectMultiScale(gray_image,
                                                            scaleFactor = 1.03,
                                                            minNeighbors = 4)
    try:
        mask_crop = mask.mask_crop(faces, gray_image)
        recognition_crop = recognition.recognition_crop(faces, gray_image)
        mask_features = mask.hog.extract_features(mask_crop)
        recognition_features = recognition.hog.extract_features(recognition_crop)
        mask_prediction = mask.mask_prediction(mask_features)
        recognition_prediction = recognition.recognition_prediction(recognition_features)
        frame_stream = add_match_to_frame(frame, recognition_prediction)
        create_bounding_box(faces, frame_stream, mask_prediction, recognition_prediction)

    except:
        frame_stream = frame
        logger.info('No face found')

    return frame_stream