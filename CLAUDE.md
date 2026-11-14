I'm building a marker-based augmented reality app in Python with OpenCV
(course project). The attached files are the current state: SIFT keypoint
detection and description work on a marker image and a camera frame.

A code review was just done and its fixes are applied: explicit imports,
grab_frame takes the camera index and raises on failure, 4-space
indentation, unused imports removed.

Design rules to keep:

- Pipeline stages are pure functions in pipeline.py (parameters in, data
  out, no plotting or config reads inside).
- Techniques are registered in config.py (see DETECTORS) so more can be
  added later (ORB, AKAZE, BRISK).
- Maintainability, readability and being generic matter most.

Next step: descriptor matching, then match filtering, RANSAC homography,
and warping the substitute image onto the frame.

How to work with me: guide me in chat, one small step at a time. I set
the pace. Don't jump ahead or write the whole thing at once.
