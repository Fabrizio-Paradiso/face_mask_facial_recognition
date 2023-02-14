import os
import sys

root = os.path.abspath(".")
sys.path.insert(1, root)
from libraries.face_recognition import FaceRecognition

# training_path = os.path.normpath(os.path.join(root, f"C:/Users/fabri/OneDrive/Escritorio/Proyecto/Dataset/images/Face-Recognition/Step 1 - Selected Images/9"))
# export_path = os.path.normpath(os.path.join(root, f"C:/Users/fabri/OneDrive/Escritorio/Proyecto/Dataset/images/Face-Recognition/Step 2 - Eyes Roi Images/9"))
training_path = os.path.normpath(
    os.path.join(
        root,
        f"C:/Users/fabri/OneDrive/Escritorio/Proyecto/Dataset/images/Face-Recognition/Step 2 - Eyes Roi Images/9",
    )
)
export_path = os.path.normpath(
    os.path.join(
        root,
        f"C:/Users/fabri/OneDrive/Escritorio/Proyecto/Dataset/images/Face-Recognition/Step 3 - Resize Images/9",
    )
)


def main():
    # face_recognition : FaceRecognition = FaceRecognition()
    # face_recognition.export_roi_in_directory(training_path, export_path, 'roi_')
    face_recognition: FaceRecognition = FaceRecognition()
    face_recognition.export_resize_image_in_directory(
        training_path, export_path, "resize_"
    )


if __name__ == "__main__":
    main()
