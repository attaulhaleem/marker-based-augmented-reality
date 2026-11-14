import cv2 as cv

from config import DETECTORS, FILTERS, MATCHERS


def build_from_registry(registry, name, **overrides):
    """
    Create the object registered as `name` in `registry`.
    Keyword `overrides` replace the registered default settings.
    Raises KeyError if `name` is not registered.
    """
    entry = registry[name]
    settings = {**entry["settings"], **overrides}
    return entry["create"](**settings)


def build_detector(detector_name, **overrides):
    """
    Create the feature detector registered as `detector_name` in DETECTORS.
    Keyword `overrides` replace the registered default settings.
    """
    return build_from_registry(DETECTORS, detector_name, **overrides)


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
    norm = DETECTORS[detector_name]["norm"]
    return build_from_registry(MATCHERS, matcher_name, normType=norm, **overrides)


def match_descriptors(matcher, query_descriptors, train_descriptors, k=2):
    """
    For each query descriptor, find its `k` nearest train descriptors.
    Returns one list of up to `k` DMatch per query descriptor, best first,
    or an empty list if either descriptor set is None.
    """
    if query_descriptors is None or train_descriptors is None:
        return []
    return matcher.knnMatch(query_descriptors, train_descriptors, k=k)


def build_filter(filter_name, **overrides):
    """
    Create the match filter registered as `filter_name` in FILTERS.
    Keyword `overrides` replace the registered default settings.
    """
    return build_from_registry(FILTERS, filter_name, **overrides)


def filter_matches(match_filter, knn_matches):
    """
    Filter `knn_matches` (one list of DMatch candidates per query
    descriptor) with `match_filter`. Returns the kept matches as a flat
    list of DMatch.
    """
    return match_filter.match_filter(knn_matches)
