import cv2 as cv

import estimators
import filters
import matchers
import views

# Input file paths
MARKER_PATH = "data/marker1.png"
SUBSTITUTE_PATH = "data/substitute1.png"

# Output file paths
FRAME_PATH = "data/frames/test_frame.png"
# Folder where the live app saves the camera frames
FRAMES_FOLDER = "data/frames"

# Benchmark frames: the saved camera frames the report compares the
# techniques on. Every entry has the "condition" the frame tests, the
# path of the "frame" and the path of the "marker" it shows.
BENCHMARK_FRAMES = {
    "easy": {
        "condition": "Reference",
        "frame": "data/frames/easy.png",
        "marker": "data/marker1.png",
    },
    "far": {
        "condition": "Viewpoint: Distance",
        "frame": "data/frames/far.png",
        "marker": "data/marker1.png",
    },
    "rotated": {
        "condition": "Viewpoint: Rotation",
        "frame": "data/frames/rotated.png",
        "marker": "data/marker1.png",
    },
    "tilt": {
        "condition": "Viewpoint: Tilt",
        "frame": "data/frames/tilt.png",
        "marker": "data/marker1.png",
    },
    "tilt_steep": {
        "condition": "Viewpoint: Steep Tilt",
        "frame": "data/frames/tilt_steep.png",
        "marker": "data/marker1.png",
    },
    "bright": {
        "condition": "Lighting: Bright",
        "frame": "data/frames/bright.png",
        "marker": "data/marker1.png",
    },
    "dim": {
        "condition": "Lighting: Dim",
        "frame": "data/frames/dim.png",
        "marker": "data/marker1.png",
    },
    "occluded": {
        "condition": "Occlusion",
        "frame": "data/frames/occluded.png",
        "marker": "data/marker1.png",
    },
    "clutter": {
        "condition": "Background Clutter",
        "frame": "data/frames/clutter.png",
        "marker": "data/marker1.png",
    },
    "weak_marker2": {
        "condition": "Marker: Small Text",
        "frame": "data/frames/weak_marker2.png",
        "marker": "data/marker2.png",
    },
    "weak_marker3": {
        "condition": "Marker: Flat Shapes",
        "frame": "data/frames/weak_marker3.png",
        "marker": "data/marker3.png",
    },
    "weak_marker4": {
        "condition": "Marker: Text Only",
        "frame": "data/frames/weak_marker4.png",
        "marker": "data/marker4.png",
    },
}
# Benchmark sweeps: the settings the report varies one at a time, keyed by
# the heading of their table. Every entry has the "stage" and the
# "technique" the setting belongs to (the technique replaces the
# selected one of that stage), the "setting" and the "values" it takes.
BENCHMARK_SWEEPS = {
    "Features": {
        "stage": "detector",
        "technique": "SIFT",
        "setting": "nfeatures",
        # 0 keeps every keypoint
        "values": (500, 1000, 2000, 0),
    },
    "Ratio": {
        "stage": "filter",
        "technique": "LOCAL",
        "setting": "ratio",
        "values": (0.5, 0.6, 0.7, 0.8, 0.9),
    },
    "Fraction": {
        "stage": "filter",
        "technique": "GLOBAL",
        "setting": "fraction",
        "values": (0.3, 0.4, 0.6, 0.8),
    },
}
# A filtered match counts as an inlier in the benchmark if the homography
# maps its marker keypoint to within this many pixels of its frame
# keypoint. One limit for every technique, so their counts compare.
BENCHMARK_MAX_ERROR = 3.0

# Pipeline settings
MARKER_MAX_SIDE = 1000
CAMERA_INDEX = 1
# Camera used when CAMERA_INDEX can't be opened (0 is the built-in one)
FALLBACK_CAMERA_INDEX = 0
CAPTURE_NEW_FRAME = False
# How much of the frame shows through the substitute: 0 replaces the
# marker completely, 1 leaves the frame unchanged
FRAME_ALPHA = 0.2

# GUI settings
# Seconds a technique selector has to rest before the GUI switches to
# the technique, so dragging past the others doesn't switch to each
SELECTOR_DELAY = 0.5
# Longest side in pixels of the shown view: larger views are shrunk
DISPLAY_MAX_SIDE = 1920
# Size of the text in the controls panel and in the status strip under
# the view: 1 is the standard size of utils.draw_text_rows
PANEL_TEXT_SCALE = 0.85
STATUS_TEXT_SCALE = 1.4
# Width in pixels of the panel under the sliders, which also sets the
# width of the controls window
PANEL_WIDTH = 420
# Number of keys listed in one row of the panel
PANEL_KEYS_PER_ROW = 3

# Label of the slider that selects the technique of each stage
SELECTOR_LABELS = {
    "detector": "Detector",
    "matcher": "Matcher",
    "filter": "Filter",
    "estimator": "Estimator",
    "view": "View",
}

# Key that selects the next technique of each stage
SELECTOR_KEYS = {
    "detector": "d",
    "matcher": "m",
    "filter": "f",
    "estimator": "e",
    "view": "v",
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
        "filters": ("LOCAL", "GLOBAL"),
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
        "filters": ("LOCAL", "GLOBAL"),
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
FILTER_NAME = "LOCAL"

FILTERS = {
    "LOCAL": {
        "create": filters.LocalRatioFilter,
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

# Views: what the app shows of a processed frame
VIEW_NAME = "AUGMENTED"

VIEWS = {
    "AUGMENTED": {
        "create": views.AugmentedView,
        "settings": {"frame_alpha": FRAME_ALPHA},
        "controls": {
            "frame_alpha": {
                "label": "Alpha",
                "minimum": 0.0,
                "maximum": 1.0,
                "step": 0.01,
            },
        },
    },
    "KEYPOINTS": {"create": views.KeypointsView, "settings": {}},
    "UNFILTERED": {
        "create": views.UnfilteredMatchesView,
        "settings": {"shown_match_count": 100},
        "controls": {
            "shown_match_count": {
                "label": "Shown",
                "minimum": 10,
                "maximum": 1000,
                "step": 1,
            },
        },
    },
    "FILTERED": {"create": views.FilteredMatchesView, "settings": {}},
    "INLIERS": {"create": views.InlierMatchesView, "settings": {}},
}

# The registry of each pipeline stage, in pipeline order
REGISTRIES = {
    "detector": DETECTORS,
    "matcher": MATCHERS,
    "filter": FILTERS,
    "estimator": ESTIMATORS,
    "view": VIEWS,
}


def get_valid_filter_names(matcher_name):
    """
    Return the names of the filters that can use the matches of the
    matcher registered as `matcher_name`, as a tuple.
    """
    return MATCHERS[matcher_name]["filters"]


def check_selected_filter(matcher_name, filter_name):
    """
    Check that the filter registered as `filter_name` can use the
    matches of the matcher registered as `matcher_name`.
    Raises ValueError if it can't.
    """
    valid_filter_names = get_valid_filter_names(matcher_name)
    if filter_name not in valid_filter_names:
        raise ValueError(
            f"Filter {filter_name} can't be used with matcher {matcher_name}. "
            f"Its valid filters are {valid_filter_names}"
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


# Fail early if the selected techniques can't be used together, a
# control doesn't fit its setting or a label is reused
check_selected_filter(MATCHER_NAME, FILTER_NAME)
for stage_registry in REGISTRIES.values():
    check_controls(stage_registry)
check_control_labels(REGISTRIES)
