import cv2 as cv

from config import DETECTORS, MATCHERS


def build_detector(name, **overrides):
    """
    Create the feature detector registered as `name` in DETECTORS.
    Keyword `overrides` replace the registered default settings.
    """
    entry = DETECTORS[name]
    return entry["create"](**{**entry["settings"], **overrides})


def extract_features(detector, image):
    """
    Detect keypoints in `image` (colour or grayscale) and compute their
    descriptors. Returns (keypoints, descriptors); descriptors is None
    if no keypoints are found.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return detector.detectAndCompute(gray, None)


def build_matcher(matcher_name, detector_name, **overrides):
    """
    Create the descriptor matcher registered as `matcher_name` in MATCHERS,
    using the distance norm of the detector registered as `detector_name`.
    Keyword `overrides` replace the registered default settings.
    """
    entry = MATCHERS[matcher_name]
    norm = DETECTORS[detector_name]["norm"]
    return entry["create"](normType=norm, **{**entry["settings"], **overrides})


def match_descriptors(matcher, query_descriptors, train_descriptors, k=2):
    """
    For each query descriptor, find its `k` nearest train descriptors.
    Returns one list of up to `k` DMatch per query descriptor, best first,
    or an empty list if either descriptor set is None.
    """
    if query_descriptors is None or train_descriptors is None:
        return []
    return matcher.knnMatch(query_descriptors, train_descriptors, k=k)
