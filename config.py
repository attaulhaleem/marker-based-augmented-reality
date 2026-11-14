import cv2 as cv

# Input file paths
MARKER_PATH = "data/marker.png"
SUBSTITUTE_PATH = "data/substitute.png"

# Output file paths
FRAME_PATH = "data/frame.png"

# Settings
MARKER_MAX_SIDE = 1000
CAMERA_INDEX = 1
CAPTURE_NEW_FRAME = False

# Detectors
DETECTOR_NAME = "SIFT"

DETECTORS = {
    "SIFT": {"create": cv.SIFT_create, "settings": {}, "norm": cv.NORM_L2},
}

# Matchers
MATCHER_NAME = "BF"

MATCHERS = {
    "BF": {"create": cv.BFMatcher, "settings": {}},
}