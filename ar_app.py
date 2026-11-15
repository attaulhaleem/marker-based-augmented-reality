import time

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
    draw_text_rows,
    load_image,
    overlay_image,
    shrink_to_max_side,
    warp_image,
)

WINDOW_NAME = "Marker-based Augmented Reality"
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
        self.name = technique_name
        self.rebuild = rebuild

        self.settings = {}
        for registered_name, entry in self.registry.items():
            self.settings[registered_name] = dict(entry["settings"])

    def get_settings(self):
        """
        Return the current settings of the selected technique, as a
        dict.
        """
        return self.settings[self.name]

    def get_controls(self):
        """
        Return the "controls" of the selected technique, as a dict that
        is empty if it has none.
        """
        return self.registry[self.name].get("controls", {})


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
        }

        self.requested_names = {}
        self.request_time = 0.0

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
            matcher_name = self.stages["matcher"].name
            return list(config.get_valid_filter_names(matcher_name))
        return list(self.stages[stage_name].registry)

    def rebuild_detector(self):
        """
        Build the detector with its current settings and detect the
        marker features again, since they depend on the detector.
        """
        stage = self.stages["detector"]
        self.detector = build_detector(stage.name, **stage.get_settings())
        self.marker_keypoints, self.marker_descriptors = apply_detector(
            self.detector, self.marker
        )

    def rebuild_matcher(self):
        """
        Build the matcher with its current settings.
        """
        stage = self.stages["matcher"]
        detector_name = self.stages["detector"].name
        self.matcher = build_matcher(stage.name, detector_name, **stage.get_settings())

    def rebuild_filter(self):
        """
        Build the match filter with its current settings.
        """
        stage = self.stages["filter"]
        self.match_filter = build_filter(stage.name, **stage.get_settings())

    def rebuild_estimator(self):
        """
        Build the homography estimator with its current settings.
        """
        stage = self.stages["estimator"]
        self.estimator = build_estimator(stage.name, **stage.get_settings())

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

    def select_next_technique(self, stage_name):
        """
        Switch the stage `stage_name` to the technique after the
        selected one, in the order of its selector, and from the last
        back to the first.
        """
        option_names = self.get_option_names(stage_name)
        selected_index = option_names.index(self.stages[stage_name].name)
        next_index = (selected_index + 1) % len(option_names)

        self.requested_names[stage_name] = option_names[next_index]
        self.switch_to_requested_techniques()

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

    def select_requested_names(self):
        """
        Make the requested techniques the selected ones and forget the
        requests. Returns the names of the stages whose technique
        changed, as a list.
        """
        changed_stage_names = []
        for stage_name, technique_name in self.requested_names.items():
            stage = self.stages[stage_name]
            if technique_name != stage.name:
                stage.name = technique_name
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
        if filter_stage.name not in valid_filter_names:
            filter_stage.name = valid_filter_names[0]

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
                    option_names.index(stage.name),
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

    def get_panel_rows(self):
        """
        Return what the controls panel shows, as a list of (heading,
        text) pairs: for every stage its label with the selected
        technique, then one row per control with its label and the
        current value of its setting, and last one row per key of the
        app.
        """
        rows = []
        for stage in self.stages.values():
            rows.append((stage.label, stage.name))
            settings = stage.get_settings()
            for setting_name, control in stage.get_controls().items():
                value = settings[setting_name]
                rows.append(("", f"{control['label']}: {value:g}"))

        key_texts = []
        for stage in self.stages.values():
            key_texts.append(f"{stage.key}: Next {stage.label}")
        key_texts.append(f"{RESET_KEY}: Reset")
        key_texts.append(f"{QUIT_KEY}: Quit")

        heading = "Keys"
        for key_text in key_texts:
            rows.append((heading, key_text))
            heading = ""
        return rows

    def show_panel(self):
        """
        Show the panel under the sliders of the controls window: the
        selected techniques by name and the real values of their
        settings, which the sliders can only show as whole numbers.
        """
        panel = draw_text_rows(self.get_panel_rows(), config.PANEL_WIDTH)
        cv.imshow(CONTROLS_WINDOW_NAME, panel)

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
    Pressing r resets every technique and setting to its config default,
    and d, m, f or e selects the next detector, matcher, filter or
    estimator.
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

        app.apply_requested_techniques()
        app.show_panel()
        cv.imshow(WINDOW_NAME, app.process_frame(frame))

        key = cv.waitKey(5)
        if key == ord(QUIT_KEY):
            break
        app.handle_key(key)

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    application()
