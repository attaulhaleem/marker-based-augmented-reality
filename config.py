import cv2 as cv

import estimators
import filters
import matchers

# Input file paths
MARKER_PATH = "data/marker.png"
SUBSTITUTE_PATH = "data/substitute.png"

# Output file paths
FRAME_PATH = "data/frame.png"

# Settings
MARKER_MAX_SIDE = 1000
CAMERA_INDEX = 1
CAPTURE_NEW_FRAME = False
# How much of the frame shows through the substitute: 0 replaces the
# marker completely, 1 leaves the frame unchanged
FRAME_ALPHA = 0.2

# Technique registries. Every entry has "create" (called with the
# settings to build the technique) and "settings" (its defaults, which
# the GUI can override). Some entries also say what the technique can
# be combined with:
# - DETECTORS "norm": the distance that compares its descriptors.
# - MATCHERS "filters": the filters that can use its matches.
# The selected techniques are checked at the bottom of this file.

# Detectors
DETECTOR_NAME = "SIFT"

DETECTORS = {
    "SIFT": {"create": cv.SIFT_create, "settings": {}, "norm": cv.NORM_L2},
    "ORB": {
        "create": cv.ORB_create,
        "settings": {"nfeatures": 2000},
        "norm": cv.NORM_HAMMING,
    },
    "AKAZE": {
        "create": cv.xfeatures2d.AKAZE_create,
        "settings": {},
        "norm": cv.NORM_HAMMING,
    },
    "BRISK": {
        "create": cv.xfeatures2d.BRISK_create,
        "settings": {},
        "norm": cv.NORM_HAMMING,
    },
}

# Matchers
MATCHER_NAME = "BF"

MATCHERS = {
    "BF": {
        "create": matchers.BruteForceMatcher,
        "settings": {"cross_check": False},
        "filters": ("RATIO", "GLOBAL"),
    },
    "BF_CROSS": {
        "create": matchers.BruteForceMatcher,
        "settings": {"cross_check": True},
        # Cross-check leaves one candidate, the ratio test needs two
        "filters": ("GLOBAL",),
    },
    "FLANN": {
        "create": matchers.FlannMatcher,
        "settings": {
            "trees": 5,
            "checks": 50,
            "table_number": 6,
            "key_size": 12,
            "multi_probe_level": 1,
        },
        "filters": ("RATIO", "GLOBAL"),
    },
}

# Filters
FILTER_NAME = "RATIO"

FILTERS = {
    "RATIO": {
        "create": filters.RatioFilter,
        "settings": {"ratio": 0.7},
    },
    "GLOBAL": {
        "create": filters.GlobalThresholdFilter,
        "settings": {"fraction": 0.3},
    },
}

# Estimators
ESTIMATOR_NAME = "RANSAC"

ESTIMATORS = {
    "RANSAC": {
        "create": estimators.HomographyEstimator,
        "settings": {
            "method": cv.RANSAC,
            "reprojection_threshold": 3.0,
            "min_inlier_count": 15,
        },
    },
}

# Fail early if the selected techniques can't be used together
if FILTER_NAME not in MATCHERS[MATCHER_NAME]["filters"]:
    raise ValueError(
        f"Filter {FILTER_NAME} can't be used with matcher {MATCHER_NAME}. "
        f"Its valid filters are {MATCHERS[MATCHER_NAME]['filters']}"
    )
