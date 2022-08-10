import os
import sys
root = os.path.abspath(".")
sys.path.insert(1, root)
# from libraries.common import blur_and_resize_images_in_directory
from libraries.face_mask import FaceMask

# training_path = os.path.normpath(os.path.join(root, f"C:/Users/fabri/OneDrive/Escritorio/Proyecto/Dataset/images/Face-Mask/Step 4 - Selected Train Test Images/no-mask"))
# export_path = os.path.normpath(os.path.join(root, f"C:/Users/fabri/OneDrive/Escritorio/Proyecto/Dataset/images/Face-Mask/Step 5 - Train Test Roi Images/no-mask"))
training_path = os.path.normpath(os.path.join(root, f"C:/Users/fabri/OneDrive/Escritorio/Proyecto/Dataset/images/Face-Mask/Step 5 - Train Test Roi Images/no-mask"))
export_path = os.path.normpath(os.path.join(root, f"C:/Users/fabri/OneDrive/Escritorio/Proyecto/Dataset/images/Face-Mask/Step 6 - Resize Images/no-mask"))

def main():
    # face_mask : FaceMask = FaceMask()
    # face_mask.export_roi_in_directory(training_path, export_path, 'roi_no-mask_')
    face_mask : FaceMask = FaceMask()
    face_mask.export_resize_image_in_directory(training_path, export_path, 'resize_')


if __name__ == "__main__":
    main()