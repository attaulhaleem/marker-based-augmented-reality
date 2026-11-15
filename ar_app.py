import cv2 as cv

import config
from estimators import HomographyNotFoundError
from pipeline import (
    apply_detector,
    apply_estimator,
    apply_filter,
    apply_matcher,
    build_detector,
    build_estimator,
    build_filter,
    build_matcher,
)
from utils import (
    center_crop_to_size,
    load_image,
    overlay_image,
    shrink_to_max_side,
    warp_image,
)

WINDOW_NAME = "Marker AR"
CONTROLS_WINDOW_NAME = "Controls"

REGISTRIES = {
    "detector": config.DETECTORS,
    "matcher": config.MATCHERS,
    "filter": config.FILTERS,
    "estimator": config.ESTIMATORS,
}


def create_setting_trackbar(
    window_name, stage_name, setting_name, control, value, on_change
):
    """
    Add a trackbar for the setting `setting_name` of the stage
    `stage_name` to the window `window_name`. `control` is the setting's
    entry in a registry "controls" (label, minimum, maximum, step) and
    `value` is where the slider starts. A trackbar only holds whole
    numbers, so its position counts steps. When the slider moves,
    `on_change` is called with `stage_name`, `setting_name` and the new
    setting value.
    """
    step = control["step"]

    def on_position_change(position):
        """
        Trackbar callback: turn the slider `position` into the setting
        value and pass it on.
        """
        on_change(stage_name, setting_name, position * step)

    cv.createTrackbar(
        control["label"],
        window_name,
        round(value / step),
        round(control["maximum"] / step),
        on_position_change,
    )
    cv.setTrackbarMin(control["label"], window_name, round(control["minimum"] / step))


class MarkerArApp:
    """
    The marker AR pipeline with its current settings: finds the marker
    in a frame and replaces it by the substitute. The settings start at
    the config defaults and are changed with the sliders of the
    controls window.
    """

    def __init__(self, marker, substitute):
        """
        Create the app for the image `marker` and the image `substitute`
        (same size as the marker), with the techniques selected in
        config and their default settings.
        """
        self.marker = marker
        self.substitute = substitute

        self.names = {
            "detector": config.DETECTOR_NAME,
            "matcher": config.MATCHER_NAME,
            "filter": config.FILTER_NAME,
            "estimator": config.ESTIMATOR_NAME,
        }

        self.settings = {}
        for stage_name, technique_name in self.names.items():
            entry = REGISTRIES[stage_name][technique_name]
            self.settings[stage_name] = dict(entry["settings"])

        self.rebuild_stage = {
            "detector": self.rebuild_detector,
            "matcher": self.rebuild_matcher,
            "filter": self.rebuild_filter,
            "estimator": self.rebuild_estimator,
        }
        for rebuild in self.rebuild_stage.values():
            rebuild()

    def rebuild_detector(self):
        """
        Build the detector with its current settings and detect the
        marker features again, since they depend on the detector.
        """
        self.detector = build_detector(
            self.names["detector"], **self.settings["detector"]
        )
        self.marker_keypoints, self.marker_descriptors = apply_detector(
            self.detector, self.marker
        )

    def rebuild_matcher(self):
        """
        Build the matcher with its current settings.
        """
        self.matcher = build_matcher(
            self.names["matcher"], self.names["detector"], **self.settings["matcher"]
        )

    def rebuild_filter(self):
        """
        Build the match filter with its current settings.
        """
        self.match_filter = build_filter(
            self.names["filter"], **self.settings["filter"]
        )

    def rebuild_estimator(self):
        """
        Build the homography estimator with its current settings.
        """
        self.estimator = build_estimator(
            self.names["estimator"], **self.settings["estimator"]
        )

    def change_setting(self, stage_name, setting_name, value):
        """
        Store the new `value` of the setting `setting_name` of the stage
        `stage_name` and rebuild that stage only.
        """
        self.settings[stage_name][setting_name] = value
        self.rebuild_stage[stage_name]()

    def create_controls(self):
        """
        Open the controls window with one trackbar for every control of
        the selected techniques, in the order of the stages.
        """
        cv.namedWindow(CONTROLS_WINDOW_NAME)
        for stage_name, technique_name in self.names.items():
            controls = REGISTRIES[stage_name][technique_name].get("controls", {})
            for setting_name, control in controls.items():
                create_setting_trackbar(
                    CONTROLS_WINDOW_NAME,
                    stage_name,
                    setting_name,
                    control,
                    self.settings[stage_name][setting_name],
                    self.change_setting,
                )

    def process_frame(self, frame):
        """
        Run the pipeline on `frame`. Returns the frame with the marker
        replaced by the substitute, or the plain frame if the marker is
        not found.
        """
        frame_keypoints, frame_descriptors = apply_detector(self.detector, frame)
        knn_matches = apply_matcher(
            self.matcher, self.marker_descriptors, frame_descriptors
        )
        good_matches = apply_filter(self.match_filter, knn_matches)

        try:
            homography, inlier_mask = apply_estimator(
                self.estimator, self.marker_keypoints, frame_keypoints, good_matches
            )
        except HomographyNotFoundError:
            return frame

        frame_height, frame_width = frame.shape[:2]
        warped_substitute, substitute_mask = warp_image(
            self.substitute, homography, frame_width, frame_height
        )
        return overlay_image(
            frame, warped_substitute, substitute_mask, config.FRAME_ALPHA
        )


def application():
    """
    Run the live app: find the marker in every camera frame and show the
    frame with the marker replaced by the substitute, until q is pressed.
    Raises RuntimeError if the camera can't be opened.
    """
    capture = cv.VideoCapture(config.CAMERA_INDEX)
    if not capture.isOpened():
        raise RuntimeError(f"Can't open camera {config.CAMERA_INDEX}")

    marker = load_image(config.MARKER_PATH)
    marker = shrink_to_max_side(marker, config.MARKER_MAX_SIDE)

    substitute = load_image(config.SUBSTITUTE_PATH)
    substitute = center_crop_to_size(substitute, marker.shape[1], marker.shape[0])

    app = MarkerArApp(marker, substitute)
    app.create_controls()

    while True:
        frame_was_read, frame = capture.read()
        if not frame_was_read:
            break

        cv.imshow(WINDOW_NAME, app.process_frame(frame))

        if cv.waitKey(5) == ord("q"):
            break

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    application()
