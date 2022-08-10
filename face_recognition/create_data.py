import os
import sys
root = os.path.abspath(".")
sys.path.insert(1, root)
from libraries.hog import Hog

dir_path = os.path.normpath(os.path.join(root, f"images/face_recognition"))
save_path = 'notebooks/recognition.csv'

def main():
    hog: Hog = Hog()
    hog.create_data_csv(dir_path, save_path)

if __name__ == "__main__":
    main()