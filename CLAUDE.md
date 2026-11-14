I'm building a marker-based augmented reality app in Python with OpenCV (course project).

Files:

- ar_app.ipynb: runs the pipeline step by step, reads config, does all plotting.
- VAR-BSC-Final-Project-2026.pdf: the assignment. Check it for requirements.
- Marker AR Project Guide.html: the project guide (pipeline stages, GUI, study, report, pitfalls). Check it for the expected approach.
- data/: marker.png (the marker), frame.png (saved camera frame), substitute.png (image to warp onto the marker).

Current state: SIFT feature extraction, descriptor matching (brute force, knnMatch with k=2) and match visualization work on a marker image and a camera frame.

Next steps: match filtering (ratio test), RANSAC homography, then warping the substitute image onto the frame.

Design rules to keep:

- Pipeline stages are pure functions in pipeline.py (parameters in, data out, no plotting). They may look up the registries by name, but never read settings such as DETECTOR_NAME; the notebook passes those in.
- Every swappable technique is registered in config.py (see DETECTORS and MATCHERS) as {"create": ..., "settings": {...}}. Each has a build_x(name) function, and the stage function takes the built object.
- Plotting and image helpers live in utils.py.
- Maintainability, readability and being generic matter most.

Code style:

- Every function has a docstring in the existing style (what it does, what it returns, what it raises).
- Consistent names: plot_x for plotting helpers, build_x for factories, descriptive variable names (shown_matches, not shown).
- Calls that don't fit on one line are wrapped one argument per line with a trailing comma. Imports are alphabetical.
- Prefer plain, readable code over compact idioms.

How to work with me:

- Guide me in chat, one small step at a time. I set the pace. Don't jump ahead or write the whole thing at once.
- Only edit files when I explicitly tell you to. Otherwise give the code in chat.
- Explain everything. For a new stage, first show the simplest way to do it, then the generic version that fits the pipeline.
- Be concise. Even when explaining everything, use short sentences and short paragraphs or bullets.
- After I've typed a step, check it by running it before moving on.
