I'm building a marker-based augmented reality app in Python with OpenCV (course project).

Files:

- report.ipynb: the report. Runs the pipeline step by step on one saved frame, reads config, does all plotting. It compares every registered detector and every registered estimator side by side; the other cells run on the techniques selected in config.
- ar_app.py: the live webcam app and its GUI (see "GUI" below). Runs the pipeline on every camera frame and shows the selected view in an OpenCV window. Run it with `python ar_app.py`, not from the notebook.
- config.py: paths, settings, GUI settings and the technique registries (DETECTORS, MATCHERS, FILTERS, ESTIMATORS, VIEWS). The selected names (DETECTOR_NAME and so on) are what the app starts with and resets to.
- pipeline.py: the build_x factories and the pipeline stage functions.
- estimators.py: the homography estimator class (HomographyEstimator) and HomographyNotFoundError. Imports nothing from the project, so config.py can import it without a cycle.
- filters.py: the match filter classes (LocalRatioFilter, GlobalThresholdFilter). Imports nothing from the project, so config.py can import it without a cycle.
- matchers.py: the matcher classes (BruteForceMatcher, FlannMatcher). Imports nothing from the project, so config.py can import it without a cycle.
- views.py: the view classes of the live app (AugmentedView, KeypointsView, UnfilteredMatchesView, FilteredMatchesView, InlierMatchesView) and FrameResult, what a view draws. Imports only utils.py, so config.py can import it without a cycle.
- utils.py: general helpers (plotting, image, geometry, draw_text_rows for the GUI text).
- VAR-BSC-Final-Project-2026.pdf: the assignment. Check it for requirements.
- Marker AR Project Guide.html: the project guide (pipeline stages, GUI, study, report, pitfalls). Check it for the expected approach.
- .venv: the project environment, with opencv-contrib-python 5.0.0. In this build AKAZE and BRISK are under cv.xfeatures2d, and SURF can't run (patented, compiled out).
- data/: marker1.png (the marker; marker2.png to marker4.png are weaker markers for the study), substitute1.png and substitute2.png (images to warp onto the marker), frames/ (saved camera frames: test_frame.png for the notebook, the others one per study condition).

Design rules to keep:

- Pipeline stages are pure functions in pipeline.py (parameters in, data out, no plotting). They may look up the registries by name, but never read settings such as DETECTOR_NAME; the notebook passes those in.
- Every swappable technique is registered in config.py (see DETECTORS, MATCHERS, FILTERS, ESTIMATORS and VIEWS) as {"create": ..., "settings": {...}}. Each has a build_x(name) function that creates the technique and an apply_x stage function that takes the built object and runs it (build_detector and apply_detector, and so on).
- config.py decides which techniques can be combined. A registry entry may carry extra keys for that: "norm" in DETECTORS, "filters" in MATCHERS (the filters that can use its matches). config.py checks the selected names itself when it is imported. The notebook never checks combinations.
- A stage that can't produce its result raises the error itself (apply_estimator raises HomographyNotFoundError). The notebook never checks results or raises. The one exception: a loop that compares techniques catches HomographyNotFoundError and prints it, so one failing technique doesn't stop the comparison.
- The notebook only calls build_x and the stage functions. Anything else a technique needs (such as how many candidates a matcher returns) is internal to its class.
- Settings are never hard-coded in the code. They live in the registry "settings" so the GUI can change them. The only exception is a value that stays constant in every case.
- Every build_x goes through build_from_registry(registry, name, **overrides) in pipeline.py. Don't repeat the lookup and settings merge.
- Techniques I write myself are classes, so they are built like OpenCV objects: settings go into __init__, and one method does the work. A matcher wraps an OpenCV matcher and has knn_match(query_descriptors, train_descriptors), which returns one list of DMatch candidates per query descriptor. A filter has match_filter(knn_matches), which returns a flat list of DMatch. An estimator wraps cv.findHomography and has estimate(query_points, train_points), which returns (homography, inlier_mask). A view has draw(result), which takes a FrameResult and returns the drawn image. Adding a filter is one class in filters.py plus one line in FILTERS.
- General helpers (plotting, image, geometry) live in utils.py. pipeline.py holds only the build_x factories and the stage functions.
- Maintainability, readability and being generic matter most.

GUI (ar_app.py, OpenCV HighGUI on macOS):

- Two windows. The video window shows the selected view above a status strip (marker found or not, keypoint and match counts, time per stage and FPS, the keys). The Controls window has the trackbars above a panel that shows the selected techniques by name, the real values of their settings and the keys.
- The app has five stages: detector, matcher, filter, estimator and view. Each is a Stage in MarkerArApp.stages, built from config.REGISTRIES, config.SELECTOR_LABELS and config.SELECTOR_KEYS. Adding a stage is one line there plus its registry, label and key in config.
- Each stage has a selector trackbar for its technique and one trackbar per entry of the selected technique's "controls" in its registry: {setting_name: {"label", "minimum", "maximum", "step"}}. A setting without a control has no slider. Adding a slider is one "controls" entry, no app code.
- Keys: d, m, f, e and v select the next detector, matcher, filter, estimator and view; s saves the camera frame (nothing drawn on it) as the next unused frame_NN.png in config.FRAMES_FOLDER, or the folder given with --frames; r resets every technique and setting to the config defaults; q quits.
- The GUI never offers a setting that conflicts with another or has no effect. Only the selected techniques have sliders (the Controls window is closed and rebuilt when a technique changes). The filter selector and the f key only cover config.get_valid_filter_names(matcher_name); with one valid filter there is no filter selector. A matcher change that makes the filter invalid selects the first valid one.
- config.py checks the controls on import: each must name an existing setting, have its minimum below its maximum and contain the default (check_controls), and a label can't be reused across stages or be a selector label (check_control_labels), because OpenCV finds a trackbar by its label.
- A change rebuilds only the stage it belongs to. A technique change rebuilds that stage and the ones after it. A detector change also detects the marker features again.
- Limits of trackbars on this backend, which the design works around: a trackbar only shows a whole number (so the position counts steps, and the panel shows the real value); label and value share about 15 characters (so labels are short, Title Case, at most about 11 characters); a trackbar can't be removed or disabled (hence the window rebuild); the slider doesn't snap to positions. A selector trackbar must rest for config.SELECTOR_DELAY seconds before the technique switches; the keys switch at once.
- All display text follows one casing: labels in Title Case, technique names as registered (SIFT, BF_CROSS, LOCAL).
- Tkinter is not usable in this .venv: its Python is linked to Apple's Tk 8.5.9, which hangs when a window updates.
- A GUI step can be checked without the camera: build MarkerArApp with the marker and substitute, call process_frame on data/frames/test_frame.png, move sliders with cv.setTrackbarPos and press keys with handle_key.

Code style:

- Every function has a docstring in the existing style (what it does, what it returns, what it raises).
- Consistent names: plot_x for plotting helpers, draw_x for helpers that return the drawn image without showing it, build_x for factories, descriptive variable names (shown_match_count, not shown).
- Formatting and import order are done by Black and isort on save. Don't dictate wrapping by hand.
- Prefer plain, readable code over compact idioms.
- A comment stays only where it makes the code clearer or easier to understand (why a value or a step is needed). Don't add comments that repeat what the code says. Put the rest of the explanation in chat.

How to work with me:

- Work one small step at a time. I set the pace. Don't jump ahead or write the whole thing at once.
- Apply each step to the files yourself. If I ask you not to, give the code in chat instead.
- Explain everything. For a new stage, first show the simplest way to do it, then the generic version that fits the pipeline.
- Be concise. Even when explaining everything, use short sentences and short paragraphs or bullets.
- After a step is applied, check it by running it before moving on.
