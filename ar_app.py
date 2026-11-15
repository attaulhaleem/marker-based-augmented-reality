import argparse
import time

import cv2 as cv
import numpy as np

import config
from estimators import HomographyNotFoundError
from pipeline import (apply_detector, apply_estimator, apply_filter,
                      apply_matcher, apply_view, build_detector,
                      build_estimator, build_filter, build_matcher, build_view)
from utils import (center_crop_to_size, count_best_matches, draw_text_rows,
                   load_image, open_camera, shrink_to_max_side)
from views import FrameResult

VIEW_WINDOW_NAME = "Marker-based Augmented Reality"
CONTROLS_WINDOW_NAME = "Controls"

QUIT_KEY = "q"
RESET_KEY = "r"

# Decimals kept in a setting value read from a trackbar, to drop the
# rounding noise of position times step (70 * 0.01 is not exactly 0.7)
VALUE_DECIMALS = 10


def value_to_position(value, step):
    """
    Return the trackbar position that stands for the setting `value`:
    the number of `step`s in it, as a whole number.
    """
    return round(value / step)


def position_to_value(position, step):
    """
    Return the setting value that the trackbar `position` stands for:
    `position` times `step`.
    """
    return round(position * step, VALUE_DECIMALS)


def create_selector_trackbar(
    window_name, stage_name, label, option_count, position, on_change
):
    """
    Add a trackbar named `label` that selects the technique of the
    stage `stage_name` to the window `window_name`. It has
    `option_count` positions, one per technique that can be selected,
    and starts at `position`. When the slider moves, `on_change` is
    called with `stage_name` and the new position.
    """

    def on_position_change(new_position):
        """
        Trackbar callback: pass the slider position on with the stage.
        """
        on_change(stage_name, new_position)

    cv.createTrackbar(
        label,
        window_name,
        position,
        option_count - 1,
        on_position_change,
    )


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
        on_change(stage_name, setting_name, position_to_value(position, step))

    cv.createTrackbar(
        control["label"],
        window_name,
        value_to_position(value, step),
        value_to_position(control["maximum"], step),
        on_position_change,
    )
    cv.setTrackbarMin(
        control["label"], window_name, value_to_position(control["minimum"], step)
    )


class Stage:
    """
    One pipeline stage of the app: which technique is selected for it
    and the current settings of each of its techniques.
    """

    def __init__(self, stage_name, technique_name, rebuild):
        """
        Create the stage `stage_name` (a key of config.REGISTRIES) with
        the technique `technique_name` selected and every technique at
        its default settings. `rebuild` is the function that builds the
        selected technique again after a change.
        """
        self.label = config.SELECTOR_LABELS[stage_name]
        self.key = config.SELECTOR_KEYS[stage_name]
        self.registry = config.REGISTRIES[stage_name]
        self.technique_name = technique_name
        self.rebuild = rebuild

        self.settings = {}
        for registered_name, entry in self.registry.items():
            self.settings[registered_name] = dict(entry["settings"])

    def get_settings(self):
        """
        Return the current settings of the selected technique, as a
        dict.
        """
        return self.settings[self.technique_name]

    def get_controls(self):
        """
        Return the "controls" of the selected technique, as a dict that
        is empty if it has none.
        """
        return self.registry[self.technique_name].get("controls", {})


class StageTimer:
    """
    Stopwatch of the pipeline: how long every stage of the last frame
    took, and how long the whole frame took.
    """

    def __init__(self):
        """
        Create the timer, with no times yet.
        """
        self.stage_seconds = {}
        self.frame_seconds = None
        self.frame_start_time = None
        self.stage_start_time = None

    def start_frame(self):
        """
        Note that a new frame starts now. From the second frame on,
        `frame_seconds` is the time since the previous frame started.
        """
        start_time = time.perf_counter()
        if self.frame_start_time is not None:
            self.frame_seconds = start_time - self.frame_start_time
        self.frame_start_time = start_time
        self.stage_start_time = start_time

    def end_stage(self, stage_name):
        """
        Note that the stage `stage_name` ends now. Its entry in
        `stage_seconds` is the time since the previous stage ended, or
        since the frame started for the first stage.
        """
        end_time = time.perf_counter()
        self.stage_seconds[stage_name] = end_time - self.stage_start_time
        self.stage_start_time = end_time


class MarkerArApp:
    """
    The marker AR pipeline with its current settings: finds the marker
    in a frame and replaces it by the substitute. The settings start at
    the config defaults. The techniques and their settings are changed
    with the sliders of the controls window.
    """

    def __init__(self, marker, substitute):
        """
        Create the app for the image `marker` and the image `substitute`
        (same size as the marker), with the techniques selected in
        config and their default settings.
        """
        self.marker = marker
        self.substitute = substitute
        self.reset()

    def reset(self):
        """
        Go back to the techniques selected in config and the default
        settings of every technique, and rebuild every stage.
        """
        self.stages = {
            "detector": Stage("detector", config.DETECTOR_NAME, self.rebuild_detector),
            "matcher": Stage("matcher", config.MATCHER_NAME, self.rebuild_matcher),
            "filter": Stage("filter", config.FILTER_NAME, self.rebuild_filter),
            "estimator": Stage(
                "estimator", config.ESTIMATOR_NAME, self.rebuild_estimator
            ),
            "view": Stage("view", config.VIEW_NAME, self.rebuild_view),
        }

        self.requested_names = {}
        self.request_time = 0.0

        self.result = None
        self.timer = StageTimer()

        for stage in self.stages.values():
            stage.rebuild()

    def get_option_names(self, stage_name):
        """
        Return the names of the techniques that can be selected for the
        stage `stage_name`, as a list: for the filter only those the
        selected matcher allows, for the other stages every registered
        technique.
        """
        if stage_name == "filter":
            matcher_name = self.stages["matcher"].technique_name
            return list(config.get_valid_filter_names(matcher_name))
        return list(self.stages[stage_name].registry)

    def rebuild_detector(self):
        """
        Build the detector with its current settings and detect the
        marker features again, since they depend on the detector.
        """
        stage = self.stages["detector"]
        self.detector = build_detector(stage.technique_name, **stage.get_settings())
        self.marker_keypoints, self.marker_descriptors = apply_detector(
            self.detector, self.marker
        )

    def rebuild_matcher(self):
        """
        Build the matcher with its current settings.
        """
        stage = self.stages["matcher"]
        detector_name = self.stages["detector"].technique_name
        self.matcher = build_matcher(
            stage.technique_name, detector_name, **stage.get_settings()
        )

    def rebuild_filter(self):
        """
        Build the match filter with its current settings.
        """
        stage = self.stages["filter"]
        self.match_filter = build_filter(stage.technique_name, **stage.get_settings())

    def rebuild_estimator(self):
        """
        Build the homography estimator with its current settings.
        """
        stage = self.stages["estimator"]
        self.estimator = build_estimator(stage.technique_name, **stage.get_settings())

    def rebuild_view(self):
        """
        Build the view with its current settings.
        """
        stage = self.stages["view"]
        self.view = build_view(stage.technique_name, **stage.get_settings())

    def change_setting(self, stage_name, setting_name, value):
        """
        Store the new `value` of the setting `setting_name` of the stage
        `stage_name` and rebuild that stage only. Does nothing if the
        setting already has that value.
        """
        stage = self.stages[stage_name]
        settings = stage.get_settings()
        if settings[setting_name] == value:
            return
        settings[setting_name] = value
        stage.rebuild()

    def request_technique(self, stage_name, position):
        """
        Note that the selector of the stage `stage_name` moved to
        `position`. The technique is not switched here but by
        apply_requested_techniques, once the selectors have rested.
        """
        option_names = self.get_option_names(stage_name)
        self.requested_names[stage_name] = option_names[position]
        self.request_time = time.perf_counter()

    def apply_requested_techniques(self):
        """
        Switch to the techniques requested with the selectors, once
        they have rested for config.SELECTOR_DELAY seconds.
        """
        if not self.requested_names:
            return
        if time.perf_counter() - self.request_time < config.SELECTOR_DELAY:
            return
        self.switch_to_requested_techniques()

    def select_next_technique(self, stage_name):
        """
        Switch the stage `stage_name` to the technique after the
        selected one, in the order of its selector, and from the last
        back to the first.
        """
        option_names = self.get_option_names(stage_name)
        selected_index = option_names.index(self.stages[stage_name].technique_name)
        next_index = (selected_index + 1) % len(option_names)

        self.requested_names[stage_name] = option_names[next_index]
        self.switch_to_requested_techniques()

    def switch_to_requested_techniques(self):
        """
        Switch to the requested techniques now. Then rebuild the changed
        stages and the controls window, so it shows the sliders of the
        selected techniques only.
        """
        changed_stage_names = self.select_requested_names()
        if not changed_stage_names:
            return

        self.replace_invalid_filter()
        self.rebuild_from_first_changed_stage(changed_stage_names)

        self.recreate_controls()

    def select_requested_names(self):
        """
        Make the requested techniques the selected ones and forget the
        requests. Returns the names of the stages whose technique
        changed, as a list.
        """
        changed_stage_names = []
        for stage_name, technique_name in self.requested_names.items():
            stage = self.stages[stage_name]
            if technique_name != stage.technique_name:
                stage.technique_name = technique_name
                changed_stage_names.append(stage_name)

        self.requested_names = {}
        return changed_stage_names

    def replace_invalid_filter(self):
        """
        If the selected matcher can't use the selected filter, select
        the first filter it can use.
        """
        filter_stage = self.stages["filter"]
        valid_filter_names = self.get_option_names("filter")
        if filter_stage.technique_name not in valid_filter_names:
            filter_stage.technique_name = valid_filter_names[0]

    def rebuild_from_first_changed_stage(self, changed_stage_names):
        """
        Rebuild the first stage in `changed_stage_names` and every stage
        after it, in pipeline order, since a stage can depend on the
        ones before it.
        """
        changed_stage_was_reached = False
        for stage_name, stage in self.stages.items():
            if stage_name in changed_stage_names:
                changed_stage_was_reached = True
            if changed_stage_was_reached:
                stage.rebuild()

    def handle_key(self, key):
        """
        React to the pressed `key` (as returned by cv.waitKey): the
        reset key resets the app, and the key of a stage selects its
        next technique. Other keys do nothing.
        """
        if key == ord(RESET_KEY):
            self.reset()
            self.recreate_controls()

        for stage_name, stage in self.stages.items():
            if key == ord(stage.key):
                self.select_next_technique(stage_name)

    def get_key_texts(self, stage_action=""):
        """
        Return one text per key of the app, as a list: the key and what
        it does. For the key of a stage that is `stage_action` followed
        by the stage label, such as "d: Next Detector".
        """
        key_texts = []
        for stage in self.stages.values():
            key_texts.append(f"{stage.key}: {stage_action}{stage.label}")
        key_texts.append(f"{RESET_KEY}: Reset")
        key_texts.append(f"{QUIT_KEY}: Quit")
        return key_texts

    def create_controls(self):
        """
        Open the controls window. Every stage gets a trackbar that
        selects its technique, unless only one can be selected, followed
        by one trackbar for every control of the selected technique.
        """
        cv.namedWindow(CONTROLS_WINDOW_NAME)
        for stage_name, stage in self.stages.items():
            option_names = self.get_option_names(stage_name)
            if len(option_names) > 1:
                create_selector_trackbar(
                    CONTROLS_WINDOW_NAME,
                    stage_name,
                    stage.label,
                    len(option_names),
                    option_names.index(stage.technique_name),
                    self.request_technique,
                )

            settings = stage.get_settings()
            for setting_name, control in stage.get_controls().items():
                create_setting_trackbar(
                    CONTROLS_WINDOW_NAME,
                    stage_name,
                    setting_name,
                    control,
                    settings[setting_name],
                    self.change_setting,
                )

    def recreate_controls(self):
        """
        Close the controls window and open it again, so it has the
        sliders of the techniques that are selected now, at their
        current values.
        """
        cv.destroyWindow(CONTROLS_WINDOW_NAME)
        self.create_controls()

    def show_panel(self):
        """
        Show the panel under the sliders of the controls window: the
        selected techniques by name and the real values of their
        settings, which the sliders can only show as whole numbers.
        """
        panel = draw_text_rows(
            self.get_panel_rows(), config.PANEL_WIDTH, config.PANEL_TEXT_SCALE
        )
        cv.imshow(CONTROLS_WINDOW_NAME, panel)

    def get_panel_rows(self):
        """
        Return what the controls panel shows, as a list of (heading,
        text) pairs: for every stage its label with the selected
        technique, then one row per control with its label and the
        current value of its setting, and last the keys of the app,
        config.PANEL_KEYS_PER_ROW to a row.
        """
        rows = []
        for stage in self.stages.values():
            rows.append((stage.label, stage.technique_name))
            settings = stage.get_settings()
            for setting_name, control in stage.get_controls().items():
                value = settings[setting_name]
                rows.append(("", f"{control['label']}: {value:g}"))

        key_texts = self.get_key_texts()
        heading = "Keys"
        for row_start in range(0, len(key_texts), config.PANEL_KEYS_PER_ROW):
            row_texts = key_texts[row_start : row_start + config.PANEL_KEYS_PER_ROW]
            rows.append((heading, ", ".join(row_texts)))
            heading = ""
        return rows

    def process_frame(self, frame):
        """
        Run the pipeline on `frame`. Returns what the selected view
        draws of the result, as an image. Also keeps the result and
        times every stage, for get_status_rows.
        """
        self.timer.start_frame()

        frame_keypoints, frame_descriptors = apply_detector(self.detector, frame)
        self.timer.end_stage("detector")

        knn_matches = apply_matcher(
            self.matcher, self.marker_descriptors, frame_descriptors
        )
        self.timer.end_stage("matcher")

        filtered_matches = apply_filter(self.match_filter, knn_matches)
        self.timer.end_stage("filter")

        try:
            homography, inlier_mask = apply_estimator(
                self.estimator, self.marker_keypoints, frame_keypoints, filtered_matches
            )
        except HomographyNotFoundError:
            homography = None
            inlier_mask = None
        self.timer.end_stage("estimator")

        self.result = FrameResult(
            self.marker,
            self.marker_keypoints,
            self.substitute,
            frame,
            frame_keypoints,
            knn_matches,
            filtered_matches,
            homography,
            inlier_mask,
        )
        view_image = apply_view(self.view, self.result)
        self.timer.end_stage("view")

        return view_image

    def get_status_rows(self):
        """
        Return what the status strip shows about the last processed
        frame, as a list of (heading, text) pairs: whether the marker
        was found, the keypoint counts, the match counts after every
        stage, the time every stage took, and the keys of the app. The
        list is empty before the first frame.
        """
        if self.result is None:
            return []
        result = self.result

        if result.homography is None:
            status = "Marker Not Found"
            inlier_count = 0
        else:
            status = "Marker Found"
            inlier_count = int(result.inlier_mask.sum())

        keypoint_text = (
            f"Marker: {len(result.marker_keypoints)}, "
            f"Frame: {len(result.frame_keypoints)}"
        )
        match_text = (
            f"Unfiltered: {count_best_matches(result.knn_matches)}, "
            f"Filtered: {len(result.filtered_matches)}, "
            f"Inliers: {inlier_count}"
        )

        time_texts = []
        for stage_name, seconds in self.timer.stage_seconds.items():
            label = self.stages[stage_name].label
            time_texts.append(f"{label}: {seconds * 1000:.0f} ms")
        frame_seconds = self.timer.frame_seconds
        if frame_seconds is not None:
            frame_rate = 1 / frame_seconds
            time_texts.append(
                f"Frame: {frame_seconds * 1000:.0f} ms ({frame_rate:.1f} FPS)"
            )

        return [
            ("Status", status),
            ("Keypoints", keypoint_text),
            ("Matches", match_text),
            ("Time", ", ".join(time_texts)),
            ("Keys", ", ".join(self.get_key_texts("Next "))),
        ]


def parse_arguments():
    """
    Read the command line of the app. Returns the parsed arguments:
    `marker` and `substitute`, the paths of the two images, which
    default to the ones in config.
    """
    parser = argparse.ArgumentParser(description="Marker-based augmented reality")
    parser.add_argument(
        "--marker",
        default=config.MARKER_PATH,
        help=f"image of the marker to find (default: {config.MARKER_PATH})",
    )
    parser.add_argument(
        "--substitute",
        default=config.SUBSTITUTE_PATH,
        help=f"image shown in place of the marker (default: {config.SUBSTITUTE_PATH})",
    )
    return parser.parse_args()


def run_app(marker_path=config.MARKER_PATH, substitute_path=config.SUBSTITUTE_PATH):
    """
    Run the live app: find the marker (the image at `marker_path`) in
    every camera frame and show the frame with the marker replaced by
    the substitute (the image at `substitute_path`), above a status
    strip, until q is pressed.
    Pressing r resets every technique and setting to its config default,
    and d, m, f, e or v selects the next detector, matcher, filter,
    estimator or view.
    Raises FileNotFoundError if an image can't be read and RuntimeError
    if no camera can be opened.
    """
    marker = load_image(marker_path)
    marker = shrink_to_max_side(marker, config.MARKER_MAX_SIDE)

    substitute = load_image(substitute_path)
    substitute = center_crop_to_size(substitute, marker.shape[1], marker.shape[0])

    capture = open_camera(config.CAMERA_INDEX, config.FALLBACK_CAMERA_INDEX)

    app = MarkerArApp(marker, substitute)
    app.create_controls()

    while True:
        frame_was_read, frame = capture.read()
        if not frame_was_read:
            break

        app.apply_requested_techniques()
        app.show_panel()
        shown_view = app.process_frame(frame)
        shown_view = shrink_to_max_side(shown_view, config.DISPLAY_MAX_SIDE)
        status_strip = draw_text_rows(
            app.get_status_rows(), shown_view.shape[1], config.STATUS_TEXT_SCALE
        )
        cv.imshow(VIEW_WINDOW_NAME, np.vstack([shown_view, status_strip]))

        key = cv.waitKey(5)
        if key == ord(QUIT_KEY):
            break
        app.handle_key(key)

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    arguments = parse_arguments()
    run_app(arguments.marker, arguments.substitute)
