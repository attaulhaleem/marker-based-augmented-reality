# Marker-Based Augmented Reality

Python app powered by OpenCV that finds a planar marker in the webcam stream with keypoint descriptor matching and replaces it with a substitute image.

## Setup

The app was run with Python 3.10.13.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Change `CAMERA_INDEX` in `config.py` to select camera. By default, it uses camera `1` with camera `0` as fallback.

Run the live app:

```
python ar_app.py
```

Optionally, specify your own marker and substitute images.

```
python ar_app.py --marker my_marker.png --substitute my_substitute.png
```

Hold up the marker object in front of the webcam to see the augmentation.

For a step-by-step walkthrough of the pipeline on one saved frame: open `walkthrough.ipynb` and run all cells.

## Controls

The app opens two windows: the video, and a Controls window with sliders for selecting techniques and tuning settings.

| Key | Action |
| --- | --- |
| `d` | Next detector (SIFT, ORB, AKAZE, BRISK) |
| `m` | Next matcher (BF, BF_CROSS, FLANN) |
| `f` | Next filter (LOCAL, GLOBAL) |
| `e` | Next estimator (RANSAC, LEAST_SQUARES, LMEDS, PROSAC) |
| `v` | Next view (AUGMENTED, KEYPOINTS, UNFILTERED, FILTERED, INLIERS) |
| `s` | Save the camera frame to `data/frames/` (change the folder with `--frames`) |
| `r` | Reset every technique and setting to the defaults |
| `q` | Quit |

## Folder structure

```
ar_app.py            Live webcam app and its GUI
walkthrough.ipynb    The pipeline step by step on one saved frame
config.py            Paths, settings and the technique registries
pipeline.py          Factories and stage functions of the pipeline
matchers.py          Brute-force and FLANN matchers
filters.py           Local (ratio test) and global (distance threshold) filters
estimators.py        Homography estimator
views.py             Views of the live app
utils.py             Image, geometry, drawing and plotting helpers
requirements.txt     Python packages
data/
  marker1.png        Marker (marker2.png to marker4.png are other markers)
  substitute1.png    Image shown in place of the marker (substitute2.png is another one)
  frames/            Saved camera frames: test_frame.png is used by the notebook,
                     the others show the marker in different conditions
```
