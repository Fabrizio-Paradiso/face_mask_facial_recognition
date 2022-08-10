from config import BLUR_FILTER_SIZE, FONT_SCALE, GREEN_COLOR, LINE_THICKNESS, MARGIN_SVM_DECISION, MATCH_SCALE, RED_COLOR, RESIZE_SIZE, TEXT_COORDINATES, TEXT_FONT, WHITE_COLOR
from datetime import datetime
from imutils import paths
import cv2
import json
import numpy
import os
import sklearn.svm
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
    cv2.putText(frame, str(datetime.now()), TEXT_COORDINATES, TEXT_FONT, FONT_SCALE, GREEN_COLOR, LINE_THICKNESS)

def add_match_to_frame(frame: numpy.ndarray, recognition_hog: tuple, text_coordinates: str = (520, 360)) -> None:
    """
    Add Match Found image in right bottom corner of frame

    Args:
        frame (numpy.ndarray): Frame captured by camera in Real-Time
        recognition_hog (tuple): Decision about recognition made by SVM multi-class model
        text_coordinates (tuple): Text location coordinates
    Return:
        None
    """
    image_path = json_name_to_image_person(str(recognition_hog[0]))
    match_image = cv2.imread(f"{image_path}")
    size = 100
    logo = cv2.resize(match_image, (size, size))
    gray_logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_logo, 1, 255, cv2.THRESH_BINARY)
    roi = frame[-size-10:-10, -size-10:-10]
    roi[numpy.where(mask)] = 0
    cv2.putText(frame, f"Match found", text_coordinates, TEXT_FONT, MATCH_SCALE, GREEN_COLOR, LINE_THICKNESS)
    roi += logo

def bounding_box_face_mask(detection_hog: tuple, faces: tuple, frame: numpy.ndarray, mask_features: tuple, svm_model: sklearn.svm._classes.LinearSVC) -> None:
    """
    Bounding box for Face-Mask Detection

    Args:
        Inherit from create_bounding_box()
    Return:
        None
    """
    cv2.rectangle(frame, (faces[0,0]-1, faces[0,1]-52), (faces[0,0]+faces[0,2]+1, faces[0,1]-20), WHITE_COLOR, -1)
    if abs(svm_model.decision_function(mask_features)) > MARGIN_SVM_DECISION:
        cv2.putText(frame, f"{detection_hog[0]}", (faces[0,0], faces[0,1]-30), TEXT_FONT, FONT_SCALE, RED_COLOR, LINE_THICKNESS)

    elif abs(svm_model.decision_function(mask_features)) < MARGIN_SVM_DECISION:
        cv2.putText(frame, f"{detection_hog[0]}", (faces[0,0], faces[0,1]-30), TEXT_FONT, FONT_SCALE, GREEN_COLOR, LINE_THICKNESS)

def bounding_box_face_name_recognition(faces: tuple, frame: numpy.ndarray, recognition_hog: tuple) -> None:
    """
    Bounding box for Face-Name Recognition

    Args:
        Inherit from create_bounding_box()
    Return:
        None
    """
    cv2.rectangle(frame, (faces[0,0], faces[0,1]), (faces[0,0]+faces[0,2], faces[0,1]+faces[0,3]), GREEN_COLOR, 2)
    cv2.rectangle(frame, (faces[0,0]-1, faces[0,1]-20), (faces[0,0]+faces[0,2]+1, faces[0,1]+5), GREEN_COLOR, -1)
    cv2.putText(frame, f"{recognition_hog[0]}", (faces[0,0], faces[0,1]), TEXT_FONT, FONT_SCALE, WHITE_COLOR, LINE_THICKNESS)

def create_bounding_box(detection_hog: tuple, faces: tuple, frame: numpy.ndarray, mask_features: tuple, recognition_hog: tuple, svm_model: sklearn.svm._classes.LinearSVC,) -> None:
    """
    Create bounding box for user interface

    Args:
        detection_hog (tuple): Decision about Decision about Face-Mask Detection made by SVM linear model
        faces (tuple): Coordinates, width and height from faces detected by Haar Cascade Frontal Face
        frame (numpy.ndarray): Frame captured by camera in Real-Time
        mask_features (tuple): Features obtained with HOG in Mask ROI
        model (sklearn.svm._classes.LinearSVC): SVM Linear Model in charge of Face-Mask Detection
        recognition_hog (tuple): Decision about Face-Name Recognition made by SVM multi-class linear model
    Return:
        None
    """
    bounding_box_face_name_recognition(faces, frame, recognition_hog)
    bounding_box_face_mask(detection_hog, faces, frame, mask_features, svm_model)

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

def json_name_to_image_person(name: str) -> str:
    """
    Find the image path in json searching by person's name

    Args:
        name (str): Name to make the query in JSON file
    Return:
        Image path located in static folder
    """
    with open("static/json/people.json") as json_file:
        people = json.load(json_file)

    person = next((person for person in people if person["name"] == name), None)
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

def resize_images_in_directory(path_directory: str, final_size: tuple):
        for image_path in paths.list_images(path_directory):
            image = cv2.imread(image_path)
            image_resized = cv2.resize(src = image, dsize=RESIZE_SIZE)
            image_filtered = cv2.GaussianBlur(image_resized, BLUR_FILTER_SIZE, cv2.BORDER_DEFAULT)
            image_resized = cv2.resize(image_filtered, final_size)
            cv2.imwrite(image_path, image_resized)
            print(image_path)

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

