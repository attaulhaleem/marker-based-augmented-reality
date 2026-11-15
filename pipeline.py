import cv2 as cv
import numpy as np

from config import DETECTORS, ESTIMATORS, FILTERS, MATCHERS


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


def apply_detector(detector, image):
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
    return build_from_registry(MATCHERS, matcher_name, norm=norm, **overrides)


def apply_matcher(matcher, query_descriptors, train_descriptors):
    """
    For each query descriptor, find its nearest train descriptors with
    `matcher`. Returns one list of DMatch candidates per query
    descriptor, best first (how many depends on the matcher), or an
    empty list if either descriptor set is None.
    """
    if query_descriptors is None or train_descriptors is None:
        return []
    return matcher.knn_match(query_descriptors, train_descriptors)


def build_filter(filter_name, **overrides):
    """
    Create the match filter registered as `filter_name` in FILTERS.
    Keyword `overrides` replace the registered default settings.
    """
    return build_from_registry(FILTERS, filter_name, **overrides)


def apply_filter(match_filter, knn_matches):
    """
    Filter `knn_matches` (one list of DMatch candidates per query
    descriptor) with `match_filter`. Returns the kept matches as a flat
    list of DMatch.
    """
    return match_filter.match_filter(knn_matches)


def build_estimator(estimator_name, **overrides):
    """
    Create the homography estimator registered as `estimator_name` in
    ESTIMATORS. Keyword `overrides` replace the registered default
    settings.
    """
    return build_from_registry(ESTIMATORS, estimator_name, **overrides)


def apply_estimator(estimator, query_keypoints, train_keypoints, matches):
    """
    Estimate, with `estimator`, the homography that maps the query
    image onto the train image from `matches` (a flat list of DMatch).
    Returns (homography, inlier_mask), where inlier_mask has one row per
    match (1 for an inlier).
    Raises HomographyNotFoundError if no homography can be estimated.
    """
    query_points = []
    train_points = []
    for match in matches:
        query_points.append(query_keypoints[match.queryIdx].pt)
        train_points.append(train_keypoints[match.trainIdx].pt)

    return estimator.estimate(np.float32(query_points), np.float32(train_points))
