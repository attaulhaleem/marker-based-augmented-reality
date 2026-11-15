I'm building a marker-based augmented reality app in Python with OpenCV (course project).

Files:

- ar_app.ipynb: runs the pipeline step by step, reads config, does all plotting.
- config.py: paths, settings and the technique registries (DETECTORS, MATCHERS, FILTERS).
- pipeline.py: the build_x factories and the pipeline stage functions.
- filters.py: the match filter classes (RatioFilter). Imports nothing from the project, so config.py can import it without a cycle.
- matchers.py: the matcher classes (BruteForceMatcher, FlannMatcher). Imports nothing from the project, so config.py can import it without a cycle.
- utils.py: plotting and image helpers.
- VAR-BSC-Final-Project-2026.pdf: the assignment. Check it for requirements.
- Marker AR Project Guide.html: the project guide (pipeline stages, GUI, study, report, pitfalls). Check it for the expected approach.
- data/: marker.png (the marker), frame.png (saved camera frame), substitute.png (image to warp onto the marker).

Design rules to keep:

- Pipeline stages are pure functions in pipeline.py (parameters in, data out, no plotting). They may look up the registries by name, but never read settings such as DETECTOR_NAME; the notebook passes those in.
- Every swappable technique is registered in config.py (see DETECTORS, MATCHERS and FILTERS) as {"create": ..., "settings": {...}}. Each has a build_x(name) function, and the stage function takes the built object.
- config.py decides which techniques can be combined. A registry entry may carry extra keys for that: "norm" in DETECTORS, "filters" in MATCHERS (the filters that can use its matches). config.py checks the selected names itself when it is imported. The notebook never checks combinations.
- The notebook only calls build_x and the stage functions. Anything else a technique needs (such as how many candidates a matcher returns) is internal to its class.
- Settings are never hard-coded in the code. They live in the registry "settings" so the GUI can change them. The only exception is a value that stays constant in every case.
- Every build_x goes through build_from_registry(registry, name, **overrides) in pipeline.py. Don't repeat the lookup and settings merge.
- Techniques I write myself are classes, so they are built like OpenCV objects: settings go into __init__, and one method does the work. A matcher wraps an OpenCV matcher and has knn_match(query_descriptors, train_descriptors), which returns one list of DMatch candidates per query descriptor. A filter has match_filter(knn_matches), which returns a flat list of DMatch. Adding a filter is one class in filters.py plus one line in FILTERS.
- Plotting and image helpers live in utils.py.
- Maintainability, readability and being generic matter most.

Code style:

- Every function has a docstring in the existing style (what it does, what it returns, what it raises).
- Consistent names: plot_x for plotting helpers, build_x for factories, descriptive variable names (shown_match_count, not shown).
- Formatting and import order are done by Black and isort on save. Don't dictate wrapping by hand.
- Prefer plain, readable code over compact idioms.

How to work with me:

- Guide me in chat, one small step at a time. I set the pace. Don't jump ahead or write the whole thing at once.
- Only edit files when I explicitly tell you to. Otherwise give the code in chat.
- Explain everything. For a new stage, first show the simplest way to do it, then the generic version that fits the pipeline.
- Be concise. Even when explaining everything, use short sentences and short paragraphs or bullets.
- After I've typed a step, check it by running it before moving on.
