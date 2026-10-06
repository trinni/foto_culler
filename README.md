# Photo Culler

Interactive, multithreaded photo culling for Linux with brightness and blur detection, keyboard-based manual rating, learned thresholds, OpenCV YuNet face detection, and automatic sorting into sharp-face, blurry/no-face, dark, bright, and blurry categories.

## Features

- Interactive GUI-based photo review.
- Keyboard-driven manual culling.
- Brightness, contrast, and sharpness analysis.
- Learn mode for deriving thresholds from manual ratings.
- Automatic resume after an interrupted learn run.
- Previously rated images are skipped by default.
- Image analysis is performed only for images that still need rating.
- Multithreaded batch processing in learn and execution modes.
- Face detection using OpenCV YuNet.
- Face-region sharpness classification.
- CSV reports containing calculated metrics.
- Local processing without uploading photos.
- Collision-safe file moving.

## Installation

Linux Mint / Debian-based systems:

```bash
make install-system
```

For an isolated Python virtual environment:

```bash
make install
```

The virtual environment installs NumPy, Pillow, and `opencv-contrib-python`. Tkinter is normally provided by the system package `python3-tk`.

Check the installation:

```bash
make check
```

## YuNet model

```bash
mkdir -p "$HOME/.local/share/photo-culler"
wget -O "$HOME/.local/share/photo-culler/face_detection_yunet_2023mar.onnx" \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
```

## Learn mode and resume behavior

Start the learn mode:

```bash
./foto_culler.py learn \
  "$HOME/Fotos/00_original" \
  "$HOME/Fotos/learn" \
  --workers 8
```

The program first loads the existing state file:

```text
$HOME/Fotos/learn/labels.json
```

If it exists, every image already present in `labels.json` is skipped automatically. Only images without a stored rating are passed to the batch analyzer and then shown in the review GUI.

This means that restarting the same command continues the previous review by default. You do not need a special resume option.

The workflow is:

1. Load `labels.json` if available.
2. Find all images recursively.
3. Remove already rated images from the pending list.
4. If no pending images remain, finish without opening the review window.
5. Run the keyboard test.
6. Analyze only pending images in parallel.
7. Review only pending images.
8. Save the updated ratings.
9. Analyze the complete collection once to calculate final thresholds.

If you want to start over, delete the state file manually:

```bash
rm "$HOME/Fotos/learn/labels.json"
```

You may also remove the previous generated files before starting over:

```bash
rm -f \
  "$HOME/Fotos/learn/labels.json" \
  "$HOME/Fotos/learn/thresholds.json" \
  "$HOME/Fotos/learn/learn-report.csv"
```

The image previews are not loaded or resized before the program knows which images still need rating. Previously rated images therefore do not trigger unnecessary analysis or review work.

## Keyboard controls

The keyboard test requests:

- Space
- Right Shift
- Right Control

During review:

| Key | Action |
|---|---|
| Left arrow | Go back |
| Right arrow | Go forward; unrated image becomes `ok` |
| Up arrow | Rate `too_bright` and continue |
| Down arrow | Rate `too_dark` and continue |
| Space | Rate `blurry` and continue |
| Right Shift | Rate `blurry` and continue |
| Right Control | Rate `blurry` and continue |
| Escape | Abort and save the current state |

Existing ratings are preserved when navigating backward. A new rating overwrites the previous rating.

## Generated files

```text
learn/
├── labels.json
├── thresholds.json
└── learn-report.csv
```

`labels.json` is the resume state. It is intentionally not deleted or replaced automatically. Delete it when a completely new review is required.

## Execution mode

```bash
./foto_culler.py actually-do-it \
  "$HOME/Fotos/00_original" \
  "$HOME/Fotos/auswertung" \
  --thresholds "$HOME/Fotos/learn/thresholds.json" \
  --workers 8
```

The program first classifies images technically into `too_dark`, `too_bright`, `blurry`, and `ok`. Face detection is then performed only for technically acceptable images.

The output structure is:

```text
output/
├── technical/
│   ├── too_dark/
│   ├── too_bright/
│   └── blurry/
├── ok/
│   ├── sharp_face/
│   └── no_face_or_blurry_face/
└── actual-report.csv
```

The execution mode moves files rather than copying them. A backup is strongly recommended. Use `--yes` to suppress the confirmation prompt.

## Threshold override

```bash
./foto_culler.py actually-do-it \
  "$HOME/Fotos/00_original" \
  "$HOME/Fotos/auswertung" \
  --dark-below 32 \
  --bright-above 232 \
  --blur-below 50 \
  --workers 8
```

## Face detection

The face detector uses OpenCV YuNet. A photo is placed in `sharp_face` if at least one detected face has a face-region sharpness value at or above `--face-blur-below`. Images without a detected face, or with only blurry detected faces, are placed in `no_face_or_blurry_face`.

Relevant options:

```text
--face-model PATH
--face-confidence FLOAT
--face-nms-threshold FLOAT
--face-top-k INTEGER
--face-blur-below FLOAT
```

Example:

```bash
./foto_culler.py actually-do-it \
  "$HOME/Fotos/00_original" \
  "$HOME/Fotos/auswertung" \
  --thresholds "$HOME/Fotos/learn/thresholds.json" \
  --face-blur-below 70 \
  --workers 8
```

## Reports

`learn-report.csv` contains the manual ratings and calculated global metrics.

`actual-report.csv` contains global metrics, face metrics, and final classifications, including:

- source path;
- image dimensions;
- brightness;
- global sharpness;
- contrast;
- detected face count;
- sharp face count;
- blurry face count;
- maximum face sharpness;
- face confidence;
- final classification.

## Supported formats

```text
.jpg .jpeg .png .tif .tiff .webp .bmp
```

RAW files are not decoded directly. Export previews first or add a RAW decoder such as `rawpy`.

## Safety and limitations

- Keep a backup of the original collection.
- Use an output directory outside the source directory.
- Review `actual-report.csv` before deleting anything.
- Brightness is a global average and can misclassify backlit or night images.
- Laplacian variance is an empirical sharpness measure.
- Face detection can miss small, occluded, profile, or rotated faces.
- The learn state is collection-specific and is not a general machine-learning model.

## Attribution

Based on the original project:

**Photo Culler** by **Duy Trinh**

Original repository: `https://github.com/trinni/foto_culler`

Modified versions must preserve this attribution and clearly identify
their changes.

## License

Copyright (c) 2026 Duy Trinh

This project is provided free of charge for civilian use.

Permission is granted to:

- use this software for civilian purposes;
- inspect and modify the source code;
- create and distribute modified versions for civilian purposes.

The following conditions apply:

1. The original author and the original project must be clearly credited.
2. Modified versions must retain this license notice.
3. Modified versions must clearly state that they have been modified.
4. The original project URL must be preserved in the attribution.
5. Military use, military deployment, military research, military training, and use by or for armed forces are prohibited.
6. Use for weapons development, targeting, combat systems, surveillance or reconnaissance in a military context is prohibited.

This software is provided “as is”, without warranty of any kind. The author shall not be liable for any claim, damages or other liability arising from the use of this software.

By using, modifying, or distributing this software, you agree to the terms of this license.
