import cv2 as cv
from config import DETECTORS


def build_detector(name, **overrides):
    detector = DETECTORS[name]
    return detector["create"](**{**detector["settings"], **overrides})


def extract_features(detector, image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return detector.detectAndCompute(gray, None)   # descriptors is None if nothing found
