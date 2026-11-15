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
# Seconds a technique selector has to rest before the GUI switches to
# the technique, so dragging past the others doesn't switch to each
SELECTOR_DELAY = 0.5
# Label of the slider that selects the technique of each stage
SELECTOR_LABELS = {
    "detector": "Detector",
    "matcher": "Matcher",
    "filter": "Filter",
    "estimator": "Estimator",
}

# Technique registries. Every entry has "create" (called with the
# settings to build the technique) and "settings" (its defaults, which
# the GUI can override). Some entries also say what the technique can
# be combined with:
# - DETECTORS "norm": the distance that compares its descriptors.
# - MATCHERS "filters": the filters that can use its matches.
# An entry can also have "controls": the settings the GUI shows a
# slider for, each with its "label", "minimum", "maximum" and "step".
# The selected techniques and the controls are checked at the bottom
# of this file.

# Detectors
DETECTOR_NAME = "SIFT"

DETECTORS = {
    "SIFT": {
        "create": cv.SIFT_create,
        # nfeatures 0 keeps every keypoint
        "settings": {"nfeatures": 0, "contrastThreshold": 0.04},
        "norm": cv.NORM_L2,
        "controls": {
            "nfeatures": {
                "label": "Features",
                "minimum": 0,
                "maximum": 5000,
                "step": 1,
            },
            "contrastThreshold": {
                "label": "Contrast",
                "minimum": 0.01,
                "maximum": 0.2,
                "step": 0.01,
            },
        },
    },
    "ORB": {
        "create": cv.ORB_create,
        "settings": {"nfeatures": 2000},
        "norm": cv.NORM_HAMMING,
        "controls": {
            "nfeatures": {
                "label": "Features",
                "minimum": 100,
                "maximum": 5000,
                "step": 1,
            },
        },
    },
    "AKAZE": {
        "create": cv.xfeatures2d.AKAZE_create,
        "settings": {"threshold": 0.001},
        "norm": cv.NORM_HAMMING,
        "controls": {
            "threshold": {
                "label": "Threshold",
                "minimum": 0.0001,
                "maximum": 0.01,
                "step": 0.0001,
            },
        },
    },
    "BRISK": {
        "create": cv.xfeatures2d.BRISK_create,
        "settings": {"thresh": 30},
        "norm": cv.NORM_HAMMING,
        "controls": {
            "thresh": {
                "label": "Threshold",
                "minimum": 5,
                "maximum": 100,
                "step": 1,
            },
        },
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
        "controls": {
            "checks": {
                "label": "Checks",
                "minimum": 10,
                "maximum": 200,
                "step": 1,
            },
        },
    },
}

# Filters
FILTER_NAME = "RATIO"

FILTERS = {
    "RATIO": {
        "create": filters.RatioFilter,
        "settings": {"ratio": 0.7},
        "controls": {
            "ratio": {
                "label": "Ratio",
                "minimum": 0.5,
                "maximum": 0.95,
                "step": 0.01,
            },
        },
    },
    "GLOBAL": {
        "create": filters.GlobalThresholdFilter,
        "settings": {"fraction": 0.3},
        "controls": {
            "fraction": {
                "label": "Fraction",
                "minimum": 0.05,
                "maximum": 1.0,
                "step": 0.01,
            },
        },
    },
}

# Estimators
ESTIMATOR_NAME = "RANSAC"

# Every estimator has the same two sliders
ESTIMATOR_CONTROLS = {
    "reprojection_threshold": {
        "label": "Max Error",
        "minimum": 0.5,
        "maximum": 10.0,
        "step": 0.1,
    },
    "min_inlier_count": {
        "label": "Min Inliers",
        "minimum": 4,
        "maximum": 100,
        "step": 1,
    },
}

ESTIMATORS = {
    "RANSAC": {
        "create": estimators.HomographyEstimator,
        "settings": {
            "method": cv.RANSAC,
            "reprojection_threshold": 3.0,
            "min_inlier_count": 15,
        },
        "controls": ESTIMATOR_CONTROLS,
    },
    "LEAST_SQUARES": {
        "create": estimators.HomographyEstimator,
        "settings": {
            "method": 0,
            "reprojection_threshold": 3.0,
            "min_inlier_count": 15,
        },
        "controls": ESTIMATOR_CONTROLS,
    },
    "LMEDS": {
        "create": estimators.HomographyEstimator,
        "settings": {
            "method": cv.LMEDS,
            "reprojection_threshold": 3.0,
            "min_inlier_count": 15,
        },
        "controls": ESTIMATOR_CONTROLS,
    },
    "PROSAC": {
        "create": estimators.HomographyEstimator,
        "settings": {
            "method": cv.RHO,
            "reprojection_threshold": 3.0,
            "min_inlier_count": 15,
        },
        "controls": ESTIMATOR_CONTROLS,
    },
}


def get_valid_filter_names(matcher_name):
    """
    Return the names of the filters that can use the matches of the
    matcher registered as `matcher_name`, as a tuple.
    """
    return MATCHERS[matcher_name]["filters"]


# Fail early if the selected techniques can't be used together
if FILTER_NAME not in get_valid_filter_names(MATCHER_NAME):
    raise ValueError(
        f"Filter {FILTER_NAME} can't be used with matcher {MATCHER_NAME}. "
        f"Its valid filters are {get_valid_filter_names(MATCHER_NAME)}"
    )


def check_controls(registry):
    """
    Check the "controls" of every entry in `registry`.
    Raises ValueError if a control has no setting of the same name, if
    its minimum is not below its maximum, or if the setting's default
    is outside that range.
    """
    for technique_name, entry in registry.items():
        for setting_name, control in entry.get("controls", {}).items():
            if setting_name not in entry["settings"]:
                raise ValueError(
                    f"Control {setting_name} of {technique_name} has no "
                    f"setting of that name"
                )

            minimum = control["minimum"]
            maximum = control["maximum"]
            if minimum >= maximum:
                raise ValueError(
                    f"Control {setting_name} of {technique_name} needs a "
                    f"minimum below its maximum, got {minimum} and {maximum}"
                )

            default = entry["settings"][setting_name]
            if default < minimum or default > maximum:
                raise ValueError(
                    f"Setting {setting_name} of {technique_name} is {default}, "
                    f"outside its control range {minimum} to {maximum}"
                )


def check_control_labels(registries):
    """
    Check that no two sliders that can be shown together have the same
    label, since a slider is found by its label. `registries` maps each
    stage name to its registry.
    Raises ValueError if a control label is also a selector label, is
    used twice by one technique, or is used in two stages.
    """
    label_stage_names = {}
    for stage_name, registry in registries.items():
        for technique_name, entry in registry.items():
            technique_labels = []
            for control in entry.get("controls", {}).values():
                label = control["label"]
                if label in SELECTOR_LABELS.values():
                    raise ValueError(
                        f"Control label {label} of {technique_name} is "
                        f"also a selector label"
                    )
                if label in technique_labels:
                    raise ValueError(
                        f"Control label {label} is used twice by {technique_name}"
                    )
                if label_stage_names.get(label, stage_name) != stage_name:
                    raise ValueError(
                        f"Control label {label} of {technique_name} is also "
                        f"used in the {label_stage_names[label]} stage"
                    )
                technique_labels.append(label)
                label_stage_names[label] = stage_name


# The registry of each pipeline stage, in pipeline order
REGISTRIES = {
    "detector": DETECTORS,
    "matcher": MATCHERS,
    "filter": FILTERS,
    "estimator": ESTIMATORS,
}

# Fail early if a control doesn't fit its setting or a label is reused
for registry in REGISTRIES.values():
    check_controls(registry)
check_control_labels(REGISTRIES)
