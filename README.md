# Photo Culler

Interactive, multithreaded photo culling for Linux with brightness and blur detection, keyboard-based manual rating, learned thresholds, OpenCV YuNet face detection, and automatic sorting into sharp-face, blurry/no-face, dark, bright, and blurry categories.

## Features

- Interactive GUI-based photo review.
- Normal maximized window on startup.
- Optional fullscreen mode using `F11` or `F`.
- EXIF-aware image rotation during preview.
- Keyboard-driven culling.
- Manual “interesting, keep” override for technically imperfect images.
- Automatic brightness, contrast, and sharpness analysis.
- Learn mode for deriving thresholds from manual ratings.
- Configurable random learning sample.
- Multithreaded batch processing.
- Face detection using OpenCV YuNet.
- Face-region sharpness classification.
- CSV reports containing calculated metrics.
- Default symbolic-link output mode.
- Optional physical file moving with `--move`.
- Local processing without uploading photos to an external service.

## Manual ratings

The learn mode uses the following controls:

| Key | Action |
|---|---|
| Left arrow | Go back one image |
| Right arrow | Go forward; unrated images become `ok` |
| Up arrow | Mark as `zu_hell` and continue |
| Down arrow | Mark as `zu_dunkel` and continue |
| Space | Mark as `unscharf` and continue |
| Right Shift | Mark as `unscharf` and continue |
| Right Control | Mark as `unscharf` and continue |
| `I` | Mark as technically imperfect but interesting |
| `F11` or `F` | Toggle fullscreen |
| Escape | Abort and save progress |

The `interessant` rating is an explicit manual override. It is intended for images that are technically imperfect but worth keeping because of their composition, moment, subject, emotion, or documentary value.

## Classification priority

Manual interest overrides technical rejection:

```text
interessant
    > zu_dunkel
    > zu_hell
    > unscharf
    > ok
```

An image marked as `interessant` is classified as:

```text
mangelhaft_aber_interessant
```

It is not passed to the normal face-based classification because it has already been explicitly selected for retention.

## Output structure

```text
output/
├── technisch/
│   ├── zu_dunkel/
│   ├── zu_hell/
│   ├── unscharf/
│   └── mangelhaft_aber_interessant/
├── ok/
│   ├── gesicht_scharf/
│   └── kein_gesicht_oder_gesicht_unscharf/
└── actual-report.csv
```

## Learn mode

Start learning in the current directory:

```bash
foto_culler learn --workers 8
```

By default, the tool evaluates up to 100 randomly selected images. If fewer than 100 supported images exist, all images are evaluated.

Learning data is stored in:

```text
.foto-culler-learn/
├── labels.json
├── sample.json
├── thresholds.json
└── learn-report.csv
```

The selected learning sample is persistent. A rerun continues with the same sample and skips already rated images.

Increase the sample size:

```bash
foto_culler learn \
    --sample-size 250 \
    --workers 8
```

The existing sample is retained and additional images are added.

Evaluate all images:

```bash
foto_culler learn \
    --sample-size 0 \
    --workers 8
```

Create a new random sample:

```bash
foto_culler learn \
    --sample-size 100 \
    --new-sample \
    --workers 8
```

Resume an interrupted review:

```bash
foto_culler learn \
    --resume \
    --workers 8
```

## Execution mode

Run classification using the learned thresholds:

```bash
foto_culler actually-do-it \
    --workers 8
```

The program searches for thresholds automatically:

```text
./.foto-culler-learn/thresholds.json
./thresholds.json
```

A specific threshold file can be supplied:

```bash
foto_culler actually-do-it \
    --thresholds "$HOME/Fotos/learn/thresholds.json" \
    --workers 8
```

Manual labels are loaded automatically from:

```text
<source>/.foto-culler-learn/labels.json
```

A different labels file can be supplied:

```bash
foto_culler actually-do-it \
    --labels "$HOME/Fotos/learn/labels.json" \
    --workers 8
```

## Output mode

Symbolic links are the default:

```bash
foto_culler actually-do-it \
    --workers 8
```

The original files remain unchanged.

To move the files physically instead:

```bash
foto_culler actually-do-it \
    --move \
    --workers 8
```

Use `--yes` to skip the confirmation prompt:

```bash
foto_culler actually-do-it \
    --move \
    --yes \
    --workers 8
```

Always keep a backup before using `--move`.

## Technical thresholds

The learn mode derives thresholds for:

- brightness below which an image is considered too dark;
- brightness above which an image is considered too bright;
- global sharpness below which an image is considered blurry.

The manually selected `interessant` images are excluded from the technical `ok` sample and therefore do not distort the learned thresholds.

## Face classification

Technically acceptable images are analyzed with OpenCV YuNet.

A face is considered sharp when the face-region sharpness is at least the configured `--face-blur-below` value.

The default categories are:

```text
gesicht_scharf
kein_gesicht_oder_gesicht_unscharf
```

Face classification is applied only to images that were not technically rejected and were not manually marked as `interessant`.

## Fullscreen and image orientation

The GUI starts as a normal maximized window. It does not start in fullscreen mode.

Use:

```text
F11
```

or:

```text
F
```

to toggle fullscreen.

Preview images are passed through Pillow’s EXIF orientation handling before display, so images captured in portrait orientation are rotated according to their metadata.

## Safety

- Keep the original photo collection unchanged until the result has been checked.
- Symbolic links are the default output mode.
- Use `--move` only after validating the classification.
- Review `actual-report.csv`.
- Use a separate output directory outside the source directory.
- The tool does not delete source files automatically.
- Manually marked interesting images are preserved separately from technical failures.

## License

Copyright (c) 2026 Duy Trinh

This project is provided free of charge for civilian use.

Permission is granted to:

- use this software for civilian purposes;
- inspect and modify the source code;
- create and distribute modified versions for civilian purposes.

The original author and the original project must be clearly credited.

Military use, military deployment, military research, military training, military intelligence, weapons development, targeting, combat systems, surveillance or reconnaissance in a military context are prohibited.

Modified versions must retain this license notice and clearly state that they have been modified.

The software is provided “as is”, without warranty of any kind.
