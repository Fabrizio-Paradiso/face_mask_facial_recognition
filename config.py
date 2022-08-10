import cv2

# Settings for Crop
BLUR_FILTER_SIZE    = (7,7)
CROP_SIZE           = (86,96)
MAX_FACE_HEIGHT     = 800
MIN_FACE_HEIGHT     = 550
RESIZE_SIZE         = (850,995)

# Settings for Haar Cascade for Detection in Real Time
MIN_NEIGHBORS       = 4
MIN_SIZE            = (100,100)
SCALE_FACTOR        = 1.3

# Settings for Haar Cascade for Training
MIN_NEIGHBORS_TRAIN = 1
MIN_SIZE_TRAIN      = (450,450)
SCALE_FACTOR_TRAIN  = 1.01

# Settings for Histogram of Oriented Gradients
BLOCK_NORM          = "L2"
CELLS_PER_BLOCK     = (2,2)
FEATURE_VECTOR      = True
ORIENTATIONS        = 10
PIXELS_PER_CELL     = (2,2)

# Settings for Model
MARGIN_SVM_DECISION = 0.05

# Settings for Bounding Box
FONT_SCALE          = 0.7
GREEN_COLOR         = (0,255,0)
MATCH_SCALE         = 0.6
LINE_THICKNESS      = 2
RED_COLOR           = (0,0,255)
BLACK_COLOR         = (0,0,0)
TEXT_COLOR          = (0,255,0)
TEXT_COORDINATES    = (10,35)
TEXT_FONT           = cv2.FONT_HERSHEY_SIMPLEX
WHITE_COLOR         = (255,255,255)

# Settings for Face Recognition ROI
XO1_FACTOR = 1/7
XO2_FACTOR = 19/21
YO1_FACTOR = 1/8
YO2_FACTOR = 1/2

# Settings for Recognition in Real Time
FINAL_SIZE = (86,96)

