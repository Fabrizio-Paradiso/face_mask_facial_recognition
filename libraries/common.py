from config import FONT_SCALE, GREEN_COLOR, BLACK_COLOR, LINE_THICKNESS, MATCH_SCALE, RESIZE_SIZE, TEXT_COORDINATES, TEXT_FONT, WHITE_COLOR
from datetime import datetime
from imutils import paths
import cv2
import json
import numpy
import os
import socket
import subprocess

def add_date_to_frame(frame: numpy.ndarray) -> None:
    """
    Add Date in left top corner of frame

    Args:
        frame (numpy.ndarray): Frame captured by camera in Real-Time
    Return:
        None
    """
    cv2.putText(frame, str(datetime.now().replace(microsecond=0)), TEXT_COORDINATES, TEXT_FONT, FONT_SCALE, GREEN_COLOR, LINE_THICKNESS)

def add_match_to_frame(frame: numpy.ndarray, prediction_recognition: int, text_coordinates: str = (520, 360)) -> None:
    """
    Add Match Found image in right bottom corner of frame

    Args:
        frame (numpy.ndarray): Frame captured by camera in Real-Time
        recognition_hog (tuple): Decision about recognition made by SVM multi-class model
        text_coordinates (tuple): Text location coordinates
    Return:
        None
    """
    image_path = json_id_to_image_person(prediction_recognition)
    match_image = cv2.imread(f"{image_path}")
    size = 100
    logo = cv2.resize(match_image, (size, size))
    added_image = cv2.addWeighted(frame[-size-10:-10, -size-10:-10, :], 0, logo[0:100,0:100, :], 1, 0)
    frame[-size-10:-10, -size-10:-10, :] = added_image
    cv2.putText(frame, f"Match found", text_coordinates, TEXT_FONT, MATCH_SCALE, BLACK_COLOR, LINE_THICKNESS)

def blur_and_resize_images_in_directory(path_directory: str):
    for image_path in paths.list_images(path_directory):
        image = cv2.imread(image_path)
        image_filtered = cv2.GaussianBlur(image, (3,3), cv2.BORDER_DEFAULT)
        image_resized = cv2.resize(src = image_filtered, dsize=RESIZE_SIZE)
        cv2.imwrite(image_path, image_resized)
        print(image_path)


def header_face_mask(faces: tuple, frame: numpy.ndarray, prediction: int) -> None:
    """
    Bounding box for Face-Mask Detection

    Args:
        Inherit from create_bounding_box()
    Return:
        None
    """
    prediction_label = json_id_to_mask_label(prediction[0])
    cv2.rectangle(frame, (faces[0][0]-1, faces[0][1]-52), (faces[0][0]+faces[0][2]+1, faces[0][1]-20), WHITE_COLOR, -1)
    cv2.putText(frame, f"{prediction_label}", (faces[0,0], faces[0,1]-30), TEXT_FONT, FONT_SCALE, BLACK_COLOR, LINE_THICKNESS)

def header_face_recognition(faces: tuple, frame: numpy.ndarray, prediction: int) -> None:
    """
    Header for Face-Name Recognition

    Args:
        Inherit from create_bounding_box()
    Return:
        None
    """
    prediction_label = json_id_to_recognition_label(prediction[0])
    cv2.rectangle(frame, (faces[0,0]-1, faces[0,1]-20), (faces[0,0]+faces[0,2]+1, faces[0,1]+5), WHITE_COLOR, -1)
    cv2.putText(frame, f"{prediction_label}", (faces[0,0], faces[0,1]), TEXT_FONT, FONT_SCALE, BLACK_COLOR, LINE_THICKNESS)

def face_bounding_box(faces: tuple, frame: numpy.ndarray) -> None:
    """
    Bounding box for face detection

    Args:
        Inherit from create_bounding_box()
    Return:
        None
    """
    cv2.rectangle(frame, (faces[0,0], faces[0,1]), (faces[0,0]+faces[0,2], faces[0,1]+faces[0,3]), WHITE_COLOR, 2)

def create_bounding_box(faces: tuple, frame: numpy.ndarray, mask_prediction: tuple, recognition_prediction: tuple) -> None:
    """
    Create bounding box for user interface

    Args:
        faces (tuple): Coordinates, width and height from faces detected by Haar Cascade Frontal Face
        frame (numpy.ndarray): Frame captured by camera in Real-Time
        mask_prediction (tuple): Prediction about wearing mask in integer
        recognition_prediction (tuple): Prediction about person in integer
    Return:
        None
    """
    face_bounding_box(faces, frame)
    header_face_recognition(faces, frame, recognition_prediction)
    header_face_mask(faces, frame, mask_prediction)

def get_ip_address_raspberry() -> str:
    """
    Get Raspberry IP address from LAN

    Args:
        None
    Return:
        Raspberry PI Local IP address (str)
    """
    return subprocess.check_output(["hostname", "-I"]).decode("utf-8").strip()

def get_ip_address_pc() -> str:
    """
    Get PC IP address from LAN

    Args:
        None
    Return:
        PC Local IP address (str)
    """
    return socket.gethostbyname(socket.gethostname())

def json_id_to_mask_label(id: int) -> str:
    """
    Find the label for Face-Mask detection searching by id

    Args:
        id (int): Name to make the query in JSON file
    Return:
        Image path located in static folder
    """
    with open("static/json/mask.json") as json_file:
        items = json.load(json_file)

    item = next((item for item in items if item["id"] == id), None)
    return str(item["label"])

def json_id_to_recognition_label(id: int) -> str:
    """
    Find the label for Face-Recognition searching by id

    Args:
        id (int): Name to make the query in JSON file
    Return:
        Image path located in static folder
    """
    with open("static/json/people.json") as json_file:
        items = json.load(json_file)

    item = next((item for item in items if item["id"] == id), None)
    return str(item["name"])

def json_id_to_image_person(id: int) -> str:
    """
    Find the image path in json searching by person's name

    Args:
        name (str): Name to make the query in JSON file
    Return:
        Image path located in static folder
    """
    with open("static/json/people.json") as json_file:
        people = json.load(json_file)

    person = next((person for person in people if person["id"] == id), None)
    return str(person["image"])

def rename_images(path_images: str, prefix_name: str) -> None:
    """
    Rename all files in a directory

    Args:
        path_images (str): Directory where images are located
        prefix_name (str): Prefix to add for all images in directory
    Return:
        None
    """
    files = os.listdir(path_images)
    for index, file in enumerate(files):
        index_image = str(index)
        os.rename(os.path.join(path_images,file), os.path.join(path_images,f"{prefix_name}_{index_image}.jpg"))

def show_face_mask_roi(frame: numpy.ndarray, coordinates: tuple) -> None:
    """
    Show face mask ROI in face's bounding box

    Args:
        frame (str): Frame captured by camera
        coordinates (tuple): Face-Mask ROI coordinates
    Return:
        None
    """
    cv2.rectangle(frame, (coordinates[0], coordinates[2]), (coordinates[1], coordinates[3]), GREEN_COLOR, LINE_THICKNESS)

