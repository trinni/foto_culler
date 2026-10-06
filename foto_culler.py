#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageOps, ImageTk


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".bmp",
}

DEFAULT_FACE_MODEL = (
    Path.home()
    / ".local"
    / "share"
    / "foto-culler"
    / "face_detection_yunet_2023mar.onnx"
)

DEFAULT_LEARN_SAMPLE_SIZE = 100
DEFAULT_SAMPLE_SEED = 20261006


@dataclass
class FaceMetrics:
    count: int
    sharp_faces: int
    blurry_faces: int
    max_sharpness: float
    max_blurry_face_sharpness: float
    face_confidence: float


@dataclass
class PhotoMetrics:
    path: str
    brightness: float
    sharpness: float
    contrast: float
    width: int
    height: int
    face_count: int = 0
    sharp_faces: int = 0
    blurry_faces: int = 0
    max_face_sharpness: float = 0.0
    max_blurry_face_sharpness: float = 0.0
    face_confidence: float = 0.0


def default_worker_count() -> int:
    return min(32, max(1, os.cpu_count() or 4))


def collect_images(source: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in source.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path).lower(),
    )


def calculate_metrics(
    path: Path,
) -> Optional[PhotoMetrics]:
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        return None

    height, width = image.shape
    max_dimension = max(width, height)

    if max_dimension > 1600:
        scale = 1600 / max_dimension

        image = cv2.resize(
            image,
            (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    brightness = float(np.mean(image))
    sharpness = float(
        cv2.Laplacian(
            image,
            cv2.CV_64F,
        ).var()
    )
    contrast = float(np.std(image))

    return PhotoMetrics(
        path=str(path),
        brightness=brightness,
        sharpness=sharpness,
        contrast=contrast,
        width=width,
        height=height,
    )


def load_metrics(
    paths: list[Path],
    workers: Optional[int] = None,
) -> list[PhotoMetrics]:
    if not paths:
        return []

    workers = workers or default_worker_count()
    workers = max(1, workers)

    results: list[Optional[PhotoMetrics]] = [
        None
    ] * len(paths)

    completed = 0

    print(
        f"Analyzing {len(paths)} images with "
        f"{workers} worker threads..."
    )

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="photo-analysis",
    ) as executor:
        futures = {
            executor.submit(
                calculate_metrics,
                path,
            ): index
            for index, path in enumerate(paths)
        }

        for future in as_completed(futures):
            index = futures[future]

            try:
                results[index] = future.result()
            except Exception as error:
                print(
                    f"\nError analyzing {paths[index]}: {error}",
                    file=sys.stderr,
                )

            completed += 1

            print(
                f"\rAnalyzing {completed}/{len(paths)}",
                end="",
                flush=True,
            )

    print()

    return [
        metric
        for metric in results
        if metric is not None
    ]


def crop_with_padding(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    padding: float = 0.15,
) -> Optional[np.ndarray]:
    image_height, image_width = image.shape[:2]

    pad_x = int(width * padding)
    pad_y = int(height * padding)

    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(
        image_width,
        x + width + pad_x,
    )
    bottom = min(
        image_height,
        y + height + pad_y,
    )

    if right <= left or bottom <= top:
        return None

    return image[top:bottom, left:right]


def calculate_face_sharpness(
    face_image: Optional[np.ndarray],
) -> float:
    if face_image is None or face_image.size == 0:
        return 0.0

    if len(face_image.shape) == 3:
        gray = cv2.cvtColor(
            face_image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = face_image

    max_dimension = max(gray.shape[:2])

    if max_dimension > 500:
        scale = 500 / max_dimension

        gray = cv2.resize(
            gray,
            (
                max(1, int(gray.shape[1] * scale)),
                max(1, int(gray.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )


class YuNetDetector:
    def __init__(
        self,
        model_path: Path,
        confidence: float,
        nms_threshold: float,
        top_k: int,
    ):
        if not model_path.exists():
            raise FileNotFoundError(
                "YuNet model not found:\n"
                f"{model_path}\n\n"
                "Download it with:\n"
                "make model"
            )

        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError(
                "This OpenCV build does not provide "
                "FaceDetectorYN."
            )

        self.model_path = model_path
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.top_k = top_k

    def detect(
        self,
        image: np.ndarray,
        face_blur_below: float,
    ) -> FaceMetrics:
        height, width = image.shape[:2]

        detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            (width, height),
            self.confidence,
            self.nms_threshold,
            self.top_k,
        )

        _, faces = detector.detect(image)

        if faces is None:
            return FaceMetrics(
                count=0,
                sharp_faces=0,
                blurry_faces=0,
                max_sharpness=0.0,
                max_blurry_face_sharpness=0.0,
                face_confidence=0.0,
            )

        sharp_faces = 0
        blurry_faces = 0
        max_sharpness = 0.0
        max_blurry_face_sharpness = 0.0
        max_confidence = 0.0

        for face in faces:
            x = max(0, int(round(face[0])))
            y = max(0, int(round(face[1])))

            face_width = max(
                1,
                int(round(face[2])),
            )

            face_height = max(
                1,
                int(round(face[3])),
            )

            confidence = float(face[14])
            max_confidence = max(
                max_confidence,
                confidence,
            )

            face_image = crop_with_padding(
                image,
                x,
                y,
                face_width,
                face_height,
            )

            face_sharpness = calculate_face_sharpness(
                face_image,
            )

            max_sharpness = max(
                max_sharpness,
                face_sharpness,
            )

            if face_sharpness >= face_blur_below:
                sharp_faces += 1
            else:
                blurry_faces += 1

                max_blurry_face_sharpness = max(
                    max_blurry_face_sharpness,
                    face_sharpness,
                )

        return FaceMetrics(
            count=len(faces),
            sharp_faces=sharp_faces,
            blurry_faces=blurry_faces,
            max_sharpness=max_sharpness,
            max_blurry_face_sharpness=(
                max_blurry_face_sharpness
            ),
            face_confidence=max_confidence,
        )


def detect_faces_for_metric(
    metric: PhotoMetrics,
    detector: YuNetDetector,
    face_blur_below: float,
) -> PhotoMetrics:
    image = cv2.imread(
        metric.path,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        return metric

    max_dimension = max(image.shape[:2])

    if max_dimension > 1600:
        scale = 1600 / max_dimension

        image = cv2.resize(
            image,
            (
                max(1, int(image.shape[1] * scale)),
                max(1, int(image.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    face_metrics = detector.detect(
        image,
        face_blur_below,
    )

    metric.face_count = face_metrics.count
    metric.sharp_faces = face_metrics.sharp_faces
    metric.blurry_faces = face_metrics.blurry_faces
    metric.max_face_sharpness = (
        face_metrics.max_sharpness
    )
    metric.max_blurry_face_sharpness = (
        face_metrics.max_blurry_face_sharpness
    )
    metric.face_confidence = (
        face_metrics.face_confidence
    )

    return metric


def load_face_metrics(
    metrics: list[PhotoMetrics],
    model_path: Path,
    face_confidence: float,
    face_nms_threshold: float,
    face_top_k: int,
    face_blur_below: float,
    workers: Optional[int] = None,
) -> list[PhotoMetrics]:
    if not metrics:
        return metrics

    workers = workers or default_worker_count()
    workers = max(1, workers)

    print(
        f"Detecting faces in {len(metrics)} images "
        f"with {workers} worker threads..."
    )

    results: list[Optional[PhotoMetrics]] = [
        None
    ] * len(metrics)

    completed = 0

    def process(
        metric: PhotoMetrics,
    ) -> PhotoMetrics:
        detector = YuNetDetector(
            model_path=model_path,
            confidence=face_confidence,
            nms_threshold=face_nms_threshold,
            top_k=face_top_k,
        )

        return detect_faces_for_metric(
            metric,
            detector,
            face_blur_below,
        )

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="face-analysis",
    ) as executor:
        futures = {
            executor.submit(
                process,
                metric,
            ): index
            for index, metric in enumerate(metrics)
        }

        for future in as_completed(futures):
            index = futures[future]

            try:
                results[index] = future.result()
            except Exception as error:
                print(
                    f"\nError detecting faces in "
                    f"{metrics[index].path}: {error}",
                    file=sys.stderr,
                )
                results[index] = metrics[index]

            completed += 1

            print(
                f"\rFace analysis "
                f"{completed}/{len(metrics)}",
                end="",
                flush=True,
            )

    print()

    return [
        metric
        for metric in results
        if metric is not None
    ]


def percentile(
    values: list[float],
    percentage: float,
) -> float:
    if not values:
        raise ValueError(
            "Cannot calculate percentile "
            "without values."
        )

    return float(
        np.percentile(
            np.asarray(values, dtype=float),
            percentage,
        )
    )


def calculate_thresholds(
    metrics: list[PhotoMetrics],
    labels: dict[str, str],
) -> dict[str, float]:
    dark_values = [
        metric.brightness
        for metric in metrics
        if labels.get(metric.path) == "zu_dunkel"
    ]

    bright_values = [
        metric.brightness
        for metric in metrics
        if labels.get(metric.path) == "zu_hell"
    ]

    blurry_values = [
        metric.sharpness
        for metric in metrics
        if labels.get(metric.path) == "unscharf"
    ]

    ok_metrics = [
        metric
        for metric in metrics
        if labels.get(metric.path) == "ok"
    ]

    thresholds: dict[str, float] = {}

    if dark_values:
        thresholds["dark_below"] = percentile(
            dark_values,
            95,
        )

    if bright_values:
        thresholds["bright_above"] = percentile(
            bright_values,
            5,
        )

    if blurry_values:
        thresholds["blur_below"] = percentile(
            blurry_values,
            95,
        )

    if ok_metrics:
        ok_brightness = [
            metric.brightness
            for metric in ok_metrics
        ]

        ok_sharpness = [
            metric.sharpness
            for metric in ok_metrics
        ]

        if "dark_below" in thresholds:
            thresholds["dark_below"] = min(
                thresholds["dark_below"],
                percentile(ok_brightness, 5),
            )

        if "bright_above" in thresholds:
            thresholds["bright_above"] = max(
                thresholds["bright_above"],
                percentile(ok_brightness, 95),
            )

        if "blur_below" in thresholds:
            thresholds["blur_below"] = min(
                thresholds["blur_below"],
                percentile(ok_sharpness, 5),
            )

    return thresholds


def load_json(
    path: Path,
    default=None,
):
    if not path.exists():
        return default

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Could not parse JSON file {path}: {error}"
        ) from error


def write_json(
    path: Path,
    value,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
        )


def write_labels(
    path: Path,
    labels: dict[str, str],
):
    write_json(path, labels)


def write_thresholds(
    path: Path,
    thresholds: dict[str, float],
):
    write_json(path, thresholds)


def write_report(
    path: Path,
    metrics: list[PhotoMetrics],
    labels: Optional[dict[str, str]] = None,
    classifications: Optional[dict[str, str]] = None,
):
    labels = labels or {}
    classifications = classifications or {}

    rows = []

    for metric in metrics:
        row = asdict(metric)

        row["learn_label"] = labels.get(
            metric.path,
            "",
        )

        row["classification"] = classifications.get(
            metric.path,
            "",
        )

        rows.append(row)

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


class KeyTestWindow:
    REQUIRED_KEYS = {
        "space": "Space",
        "shift_r": "Right Shift",
        "control_r": "Right Control",
    }

    def __init__(
        self,
        root: tk.Tk,
    ):
        self.root = root
        self.detected: set[str] = set()
        self.result = False

        self.root.title(
            "Keyboard test"
        )

        self.root.geometry(
            "720x300"
        )

        self.root.resizable(
            False,
            False,
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.abort,
        )

        title = tk.Label(
            root,
            text=(
                "Press the following keys:\n"
                "Space, Right Shift and Right Control"
            ),
            font=("Sans", 16),
        )

        title.pack(
            pady=(20, 10),
        )

        self.status_label = tk.Label(
            root,
            text="",
            font=("Sans", 12),
            justify=tk.LEFT,
        )

        self.status_label.pack(
            pady=10,
        )

        hint = tk.Label(
            root,
            text="Escape cancels the operation",
            fg="#555555",
        )

        hint.pack(
            pady=10,
        )

        self.root.bind(
            "<KeyPress>",
            self.on_key,
        )

        self.root.focus_force()
        self.update_status()

    def normalize_event(
        self,
        event,
    ) -> Optional[str]:
        keysym = event.keysym
        keysym_num = getattr(
            event,
            "keysym_num",
            None,
        )

        if keysym == "space":
            return "space"

        if keysym == "Shift_R":
            return "shift_r"

        if keysym == "Control_R":
            return "control_r"

        if keysym_num == 65506:
            return "shift_r"

        if keysym_num == 65508:
            return "control_r"

        return None

    def on_key(
        self,
        event,
    ):
        if event.keysym == "Escape":
            self.abort()
            return "break"

        detected = self.normalize_event(
            event,
        )

        if detected is not None:
            self.detected.add(
                detected,
            )
            self.update_status()

        if self.detected == set(
            self.REQUIRED_KEYS,
        ):
            self.result = True

            self.root.after(
                350,
                self.finish,
            )

        return "break"

    def update_status(self):
        lines = []

        for key_id, label in self.REQUIRED_KEYS.items():
            marker = (
                "✓"
                if key_id in self.detected
                else "○"
            )

            lines.append(
                f"{marker} {label}"
            )

        self.status_label.configure(
            text="\n".join(lines),
        )

    def finish(self):
        self.root.destroy()

    def abort(self):
        self.result = False
        self.root.destroy()


def run_key_test() -> bool:
    root = tk.Tk()
    test_window = KeyTestWindow(root)
    root.mainloop()
    return test_window.result


class LearnWindow:
    def __init__(
        self,
        root: tk.Tk,
        metrics: list[PhotoMetrics],
        labels: dict[str, str],
        preview_width: int = 1100,
        preview_height: int = 760,
    ):
        self.root = root
        self.metrics = metrics
        self.labels = labels
        self.preview_width = preview_width
        self.preview_height = preview_height

        self.index = 0
        self.photo_image = None
        self.finished = False
        self.fullscreen = False
        self.normal_geometry = None
        self.resize_job = None

        self.root.title(
            "Photo Culler - Learn mode"
        )

        self.root.geometry(
            f"{preview_width}x{preview_height}"
        )

        self.root.minsize(
            800,
            600,
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.abort,
        )

        self.root.bind(
            "<KeyPress>",
            self.handle_key,
        )

        self.root.bind(
            "<Configure>",
            self.handle_configure,
        )

        self.build_widgets()

        self.root.after(
            100,
            self.finish_initial_window_setup,
        )

    def build_widgets(self):
        self.root.configure(
            background="#202020",
        )

        self.image_frame = tk.Frame(
            self.root,
            background="#202020",
        )

        self.image_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=(10, 4),
        )

        self.image_label = tk.Label(
            self.image_frame,
            background="#202020",
        )

        self.image_label.pack(
            fill=tk.BOTH,
            expand=True,
        )

        self.info_label = tk.Label(
            self.root,
            text="",
            anchor="w",
            justify=tk.LEFT,
            font=("Sans", 12),
            foreground="#ffffff",
            background="#202020",
        )

        self.info_label.pack(
            fill=tk.X,
            padx=10,
            pady=(4, 0),
        )

        self.help_label = tk.Label(
            self.root,
            text=(
                "← back   "
                "→ next/OK   "
                "↑ too bright   "
                "↓ too dark   "
                "Space/right Shift/right Control = blurry   "
                "I = interesting/keep   "
                "F11/F = fullscreen   "
                "Esc = cancel"
            ),
            anchor="w",
            foreground="#bbbbbb",
            background="#202020",
        )

        self.help_label.pack(
            fill=tk.X,
            padx=10,
            pady=(2, 8),
        )

    def finish_initial_window_setup(self):
        self.root.update_idletasks()

        try:
            self.root.state("zoomed")
        except tk.TclError:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            self.root.geometry(
                f"{screen_width}x{screen_height}+0+0"
            )

        self.root.update_idletasks()

        self.normal_geometry = (
            self.root.geometry()
        )

        self.root.focus_force()

        self.root.after_idle(
            self.show_current,
        )

    def current_metric(
        self,
    ) -> Optional[PhotoMetrics]:
        if not self.metrics:
            return None

        if self.index < 0:
            return None

        if self.index >= len(self.metrics):
            return None

        return self.metrics[self.index]

    def get_image_area_size(self) -> tuple[int, int]:
        self.root.update_idletasks()

        width = self.image_frame.winfo_width()
        height = self.image_frame.winfo_height()

        if width <= 1:
            width = self.root.winfo_width()

        if height <= 1:
            height = self.root.winfo_height()

        return (
            max(100, width - 20),
            max(100, height - 20),
        )

    def show_current(self):
        metric = self.current_metric()

        if metric is None:
            self.finish()
            return

        path = Path(metric.path)

        try:
            with Image.open(path) as source_image:
                image = ImageOps.exif_transpose(
                    source_image,
                )

                image = image.convert(
                    "RGB",
                )

            image.thumbnail(
                self.get_image_area_size(),
                Image.Resampling.LANCZOS,
            )

            self.photo_image = ImageTk.PhotoImage(
                image,
            )

            self.image_label.configure(
                image=self.photo_image,
                text="",
            )

        except Exception as error:
            self.photo_image = None

            self.image_label.configure(
                image="",
                text=(
                    "Could not load image:\n"
                    f"{error}"
                ),
                foreground="white",
                background="#202020",
            )

        label = self.labels.get(
            metric.path,
        )

        info = (
            f"{self.index + 1}/{len(self.metrics)}   "
            f"{path.name}\n"
            f"brightness: {metric.brightness:.2f}   "
            f"sharpness: {metric.sharpness:.2f}   "
            f"contrast: {metric.contrast:.2f}\n"
            f"Current rating: "
            f"{label or 'not rated'}"
        )

        self.info_label.configure(
            text=info,
        )

    def handle_configure(
        self,
        event,
    ):
        if event.widget is not self.root:
            return

        if self.resize_job is not None:
            self.root.after_cancel(
                self.resize_job,
            )

        self.resize_job = self.root.after(
            100,
            self.refresh_preview,
        )

    def refresh_preview(self):
        self.resize_job = None

        if self.current_metric() is not None:
            self.show_current()

    def toggle_fullscreen(self):
        if self.fullscreen:
            self.fullscreen = False

            self.root.attributes(
                "-fullscreen",
                False,
            )

            if self.normal_geometry:
                self.root.geometry(
                    self.normal_geometry,
                )

            try:
                self.root.state(
                    "zoomed",
                )
            except tk.TclError:
                pass

        else:
            self.normal_geometry = (
                self.root.geometry()
            )

            self.fullscreen = True

            self.root.attributes(
                "-fullscreen",
                True,
            )

        self.root.after(
            100,
            self.refresh_preview,
        )

    def handle_key(
        self,
        event,
    ):
        keysym = event.keysym
        keysym_num = getattr(
            event,
            "keysym_num",
            None,
        )

        if keysym == "Escape":
            self.abort()
            return "break"

        if keysym in {
            "F11",
            "f",
            "F",
        }:
            self.toggle_fullscreen()
            return "break"

        if keysym == "Left":
            self.go_back()
            return "break"

        if keysym == "Right":
            self.go_forward()
            return "break"

        if keysym == "Up":
            self.mark_and_advance(
                "zu_hell",
            )
            return "break"

        if keysym == "Down":
            self.mark_and_advance(
                "zu_dunkel",
            )
            return "break"

        if keysym == "space":
            self.mark_and_advance(
                "unscharf",
            )
            return "break"

        if keysym == "Shift_R":
            self.mark_and_advance(
                "unscharf",
            )
            return "break"

        if keysym == "Control_R":
            self.mark_and_advance(
                "unscharf",
            )
            return "break"

        if keysym_num == 65506:
            self.mark_and_advance(
                "unscharf",
            )
            return "break"

        if keysym_num == 65508:
            self.mark_and_advance(
                "unscharf",
            )
            return "break"

        if keysym.lower() == "i":
            self.mark_and_advance(
                "interessant",
            )
            return "break"

        return "break"

    def go_back(self):
        if self.index > 0:
            self.index -= 1

        self.show_current()

    def go_forward(self):
        metric = self.current_metric()

        if metric is None:
            return

        if metric.path not in self.labels:
            self.labels[metric.path] = "ok"

        self.index += 1
        self.show_current()

    def mark_and_advance(
        self,
        label: str,
    ):
        metric = self.current_metric()

        if metric is None:
            return

        self.labels[metric.path] = label
        self.index += 1
        self.show_current()

    def finish(self):
        self.finished = True

        if self.fullscreen:
            self.root.attributes(
                "-fullscreen",
                False,
            )

        self.root.destroy()

    def abort(self):
        self.finished = False

        if self.fullscreen:
            self.root.attributes(
                "-fullscreen",
                False,
            )

        self.root.destroy()


def classify(
    metric: PhotoMetrics,
    thresholds: dict[str, float],
) -> str:
    if (
        "dark_below" in thresholds
        and metric.brightness < thresholds["dark_below"]
    ):
        return "zu_dunkel"

    if (
        "bright_above" in thresholds
        and metric.brightness > thresholds["bright_above"]
    ):
        return "zu_hell"

    if (
        "blur_below" in thresholds
        and metric.sharpness < thresholds["blur_below"]
    ):
        return "unscharf"

    return "ok"


def classify_face(
    metric: PhotoMetrics,
) -> str:
    if (
        metric.face_count > 0
        and metric.sharp_faces > 0
    ):
        return "gesicht_scharf"

    return (
        "kein_gesicht_oder_"
        "gesicht_unscharf"
    )


def unique_target(path: Path) -> Path:
    if (
        not path.exists()
        and not path.is_symlink()
    ):
        return path

    counter = 1

    while True:
        candidate = path.with_name(
            f"{path.stem}_{counter}{path.suffix}",
        )

        if (
            not candidate.exists()
            and not candidate.is_symlink()
        ):
            return candidate

        counter += 1


def move_classified(
    metrics: list[PhotoMetrics],
    classifications: dict[str, str],
    output_dir: Path,
    move_files: bool,
) -> dict[str, int]:
    counters: dict[str, int] = {}

    for metric in metrics:
        classification = classifications[
            metric.path
        ]

        source = Path(metric.path)

        target_dir = output_dir / classification
        target_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        target = unique_target(
            target_dir / source.name,
        )

        if move_files:
            shutil.move(
                str(source),
                str(target),
            )
        else:
            relative_source = os.path.relpath(
                source,
                start=target_dir,
            )

            target.symlink_to(
                relative_source,
            )

        counters[classification] = (
            counters.get(classification, 0) + 1
        )

    return counters


def print_thresholds(
    thresholds: dict[str, float],
):
    print("\nThresholds:")

    if "dark_below" in thresholds:
        print(
            f"  --dark-below "
            f"{thresholds['dark_below']:.4f}"
        )

    if "bright_above" in thresholds:
        print(
            f"  --bright-above "
            f"{thresholds['bright_above']:.4f}"
        )

    if "blur_below" in thresholds:
        print(
            f"  --blur-below "
            f"{thresholds['blur_below']:.4f}"
        )


def load_sample(
    path: Path,
) -> dict:
    if not path.exists():
        return {
            "seed": DEFAULT_SAMPLE_SEED,
            "requested_size": DEFAULT_LEARN_SAMPLE_SIZE,
            "paths": [],
        }

    value = load_json(
        path,
        default={},
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"Sample file must contain an object: {path}"
        )

    paths = value.get(
        "paths",
        [],
    )

    if not isinstance(paths, list):
        raise ValueError(
            f"Sample file contains invalid paths: {path}"
        )

    return {
        "seed": int(
            value.get(
                "seed",
                DEFAULT_SAMPLE_SEED,
            )
        ),
        "requested_size": int(
            value.get(
                "requested_size",
                DEFAULT_LEARN_SAMPLE_SIZE,
            )
        ),
        "paths": [
            str(item)
            for item in paths
        ],
    }


def choose_sample_paths(
    all_paths: list[Path],
    sample_file: Path,
    requested_size: int,
    new_sample: bool = False,
) -> list[Path]:
    if requested_size < 0:
        raise ValueError(
            "sample size must be zero or greater"
        )

    all_paths_by_string = {
        str(path): path
        for path in all_paths
    }

    if requested_size == 0:
        selected_paths = list(all_paths)

        write_json(
            sample_file,
            {
                "seed": DEFAULT_SAMPLE_SEED,
                "requested_size": 0,
                "paths": [
                    str(path)
                    for path in selected_paths
                ],
            },
        )

        return selected_paths

    if (
        sample_file.exists()
        and not new_sample
    ):
        sample_data = load_sample(
            sample_file,
        )
    else:
        sample_data = {
            "seed": DEFAULT_SAMPLE_SEED,
            "requested_size": requested_size,
            "paths": [],
        }

    seed = int(
        sample_data.get(
            "seed",
            DEFAULT_SAMPLE_SEED,
        )
    )

    old_paths = [
        path
        for path in sample_data.get(
            "paths",
            [],
        )
        if path in all_paths_by_string
    ]

    effective_size = min(
        requested_size,
        len(all_paths),
    )

    if len(old_paths) > effective_size:
        rng = random.Random(seed)

        old_paths = rng.sample(
            old_paths,
            effective_size,
        )

    selected_strings = set(old_paths)

    remaining = [
        path
        for path in all_paths
        if str(path) not in selected_strings
    ]

    additional_count = (
        effective_size - len(old_paths)
    )

    if additional_count > 0:
        rng = random.Random(seed)
        rng.shuffle(remaining)

        additional_paths = remaining[
            :additional_count
        ]

        old_paths.extend(
            str(path)
            for path in additional_paths
        )

    selected_paths = [
        all_paths_by_string[path]
        for path in old_paths
        if path in all_paths_by_string
    ]

    selected_paths = selected_paths[
        :effective_size
    ]

    write_json(
        sample_file,
        {
            "seed": seed,
            "requested_size": requested_size,
            "paths": [
                str(path)
                for path in selected_paths
            ],
        },
    )

    return selected_paths


def run_learn(args) -> int:
    source = (
        args.source.expanduser().resolve()
        if args.source is not None
        else Path.cwd().resolve()
    )

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else source / ".foto-culler-learn"
    )

    print(
        f"Source directory: {source}"
    )

    print(
        f"Learning data directory: {output}"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_path = output / "labels.json"
    thresholds_path = output / "thresholds.json"
    report_path = output / "learn-report.csv"
    sample_path = output / "sample.json"

    all_paths = collect_images(source)

    if not all_paths:
        print(
            f"No supported image files found in: "
            f"{source}"
        )
        return 1

    requested_sample_size = (
        args.sample_size
        if args.sample_size is not None
        else DEFAULT_LEARN_SAMPLE_SIZE
    )

    try:
        sample_paths = choose_sample_paths(
            all_paths=all_paths,
            sample_file=sample_path,
            requested_size=requested_sample_size,
            new_sample=args.new_sample,
        )

    except ValueError as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        return 1

    print(
        f"{len(all_paths)} images found."
    )

    print(
        f"Evaluating {len(sample_paths)} "
        "images as the learning sample."
    )

    labels: dict[str, str] = {}

    if labels_path.exists():
        try:
            loaded_labels = load_json(
                labels_path,
                default={},
            )

            if not isinstance(
                loaded_labels,
                dict,
            ):
                print(
                    f"Invalid labels file: {labels_path}",
                    file=sys.stderr,
                )
                return 1

            labels = {
                str(path): str(label)
                for path, label in loaded_labels.items()
            }

        except ValueError as error:
            print(
                str(error),
                file=sys.stderr,
            )
            return 1

        print(
            f"Loaded {len(labels)} existing ratings."
        )

    sample_path_strings = {
        str(path)
        for path in sample_paths
    }

    labels = {
        path: label
        for path, label in labels.items()
        if path in sample_path_strings
    }

    pending_paths = [
        path
        for path in sample_paths
        if str(path) not in labels
    ]

    if not pending_paths:
        print(
            "All images in the learning sample "
            "are already rated."
        )

        if args.rebuild_report:
            sample_metrics = load_metrics(
                sample_paths,
                args.workers,
            )

            thresholds = calculate_thresholds(
                sample_metrics,
                labels,
            )

            write_thresholds(
                thresholds_path,
                thresholds,
            )

            write_report(
                report_path,
                sample_metrics,
                labels=labels,
            )

            print_thresholds(
                thresholds,
            )

        return 0

    print(
        f"{len(pending_paths)} images still need "
        "rating."
    )

    print(
        "\nKeyboard test: press Space, "
        "Right Shift and Right Control."
    )

    if not run_key_test():
        print(
            "Keyboard test cancelled."
        )
        return 0

    pending_metrics = load_metrics(
        pending_paths,
        args.workers,
    )

    if not pending_metrics:
        print(
            "No image metrics could be calculated.",
            file=sys.stderr,
        )
        return 1

    root = tk.Tk()

    window = LearnWindow(
        root,
        pending_metrics,
        labels,
    )

    root.mainloop()

    write_labels(
        labels_path,
        labels,
    )

    write_report(
        report_path,
        pending_metrics,
        labels=labels,
    )

    if not window.finished:
        print(
            f"Progress saved to {labels_path}"
        )
        return 0

    missing = [
        str(path)
        for path in pending_paths
        if str(path) not in labels
    ]

    if missing:
        print(
            f"{len(missing)} images are still unrated."
        )

        print(
            f"Progress saved to {labels_path}"
        )

        return 1

    sample_metrics = load_metrics(
        sample_paths,
        args.workers,
    )

    if not sample_metrics:
        print(
            "No image metrics could be calculated "
            "for the learning sample.",
            file=sys.stderr,
        )
        return 1

    thresholds = calculate_thresholds(
        sample_metrics,
        labels,
    )

    write_thresholds(
        thresholds_path,
        thresholds,
    )

    write_report(
        report_path,
        sample_metrics,
        labels=labels,
    )

    print_thresholds(
        thresholds,
    )

    print(
        f"Saved ratings: {labels_path}"
    )

    print(
        f"Saved sample: {sample_path}"
    )

    print(
        f"Saved thresholds: {thresholds_path}"
    )

    print(
        f"Saved report: {report_path}"
    )

    return 0


def load_execution_thresholds(
    args,
) -> dict[str, float]:
    thresholds: dict[str, float] = {}

    thresholds_path = (
        args.thresholds.expanduser().resolve()
        if args.thresholds is not None
        else None
    )

    if thresholds_path is None:
        source = (
            args.source.expanduser().resolve()
            if args.source is not None
            else Path.cwd().resolve()
        )

        candidates = [
            source
            / ".foto-culler-learn"
            / "thresholds.json",
            source / "thresholds.json",
            Path.cwd() / "thresholds.json",
        ]

        for candidate in candidates:
            if candidate.exists():
                thresholds_path = candidate
                break

    if thresholds_path is not None:
        loaded = load_json(
            thresholds_path,
            default={},
        )

        if not isinstance(
            loaded,
            dict,
        ):
            raise ValueError(
                f"Threshold file must contain an object: "
                f"{thresholds_path}"
            )

        thresholds = {
            str(key): float(value)
            for key, value in loaded.items()
            if key in {
                "dark_below",
                "bright_above",
                "blur_below",
            }
        }

        print(
            f"Loaded thresholds from: "
            f"{thresholds_path}"
        )

    if args.dark_below is not None:
        thresholds["dark_below"] = args.dark_below

    if args.bright_above is not None:
        thresholds["bright_above"] = (
            args.bright_above
        )

    if args.blur_below is not None:
        thresholds["blur_below"] = args.blur_below

    if not thresholds:
        raise ValueError(
            "No thresholds found. Run learn first, "
            "provide --thresholds, or specify "
            "manual threshold arguments."
        )

    return thresholds


def run_actual(args) -> int:
    source = (
        args.source.expanduser().resolve()
        if args.source is not None
        else Path.cwd().resolve()
    )

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else source / ".foto-culler-output"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = collect_images(source)

    if not paths:
        print(
            f"No supported image files found in: "
            f"{source}"
        )
        return 1

    try:
        thresholds = load_execution_thresholds(
            args,
        )
    except ValueError as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        return 1

    print(
        f"Source directory: {source}"
    )

    print(
        f"Output directory: {output}"
    )

    print_thresholds(
        thresholds,
    )

    if args.move:
        print(
            "\nMode: move files."
        )
    else:
        print(
            "\nMode: symbolic links."
        )

    if not args.yes:
        answer = input(
            "Continue? [yes/NO] "
        ).strip().lower()

        if answer not in {
            "yes",
            "y",
            "ja",
            "j",
        }:
            print(
                "Cancelled."
            )
            return 0

    metrics = load_metrics(
        paths,
        args.workers,
    )

    # Manuell als interessant markierte Bilder
    # überschreiben technische Kriterien.
    interesting_metrics = [
        metric
        for metric in metrics
        if args.labels_file
        and False
    ]

    classifications = {
        metric.path: classify(
            metric,
            thresholds,
        )
        for metric in metrics
    }

    technical_metrics = [
        metric
        for metric in metrics
        if classifications[metric.path] == "ok"
    ]

    if technical_metrics:
        technical_metrics = load_face_metrics(
            metrics=technical_metrics,
            model_path=(
                args.face_model.expanduser().resolve()
            ),
            face_confidence=args.face_confidence,
            face_nms_threshold=args.face_nms_threshold,
            face_top_k=args.face_top_k,
            face_blur_below=args.face_blur_below,
            workers=args.workers,
        )

    report_path = output / "actual-report.csv"

    write_report(
        report_path,
        metrics,
        classifications=classifications,
    )

    technical_output = output / "technisch"

    technical_counters = move_classified(
        metrics=[
            metric
            for metric in metrics
            if classifications[metric.path] != "ok"
        ],
        classifications=classifications,
        output_dir=technical_output,
        move_files=args.move,
    )

    face_classifications = {
        metric.path: classify_face(metric)
        for metric in technical_metrics
    }

    face_output = output / "ok"

    face_counters = move_classified(
        metrics=technical_metrics,
        classifications=face_classifications,
        output_dir=face_output,
        move_files=args.move,
    )

    print(
        "\nTechnical classification:"
    )

    for classification in sorted(
        technical_counters,
    ):
        print(
            f"  {classification}: "
            f"{technical_counters[classification]}"
        )

    print(
        "\nFace classification:"
    )

    for classification in sorted(
        face_counters,
    ):
        print(
            f"  {classification}: "
            f"{face_counters[classification]}"
        )

    print(
        f"\nSaved report: {report_path}"
    )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive photo culling tool "
            "for Linux."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
    )

    learn = subparsers.add_parser(
        "learn",
        help="Rate photos and learn thresholds.",
    )

    learn.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Source directory "
            "(default: current directory)"
        ),
    )

    learn.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Learning data directory "
            "(default: <source>/.foto-culler-learn)"
        ),
    )

    learn.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing ratings.",
    )

    learn.add_argument(
        "--rebuild-report",
        action="store_true",
        help=(
            "Rebuild report and thresholds."
        ),
    )

    learn.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Number of images to evaluate "
            "(default: 100, 0 = all)"
        ),
    )

    learn.add_argument(
        "--new-sample",
        action="store_true",
        help=(
            "Create a new random learning sample."
        ),
    )

    learn.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of analysis threads."
        ),
    )

    actual = subparsers.add_parser(
        "actually-do-it",
        help="Classify and organize photos.",
    )

    actual.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Source directory "
            "(default: current directory)"
        ),
    )

    actual.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Output directory "
            "(default: <source>/.foto-culler-output)"
        ),
    )

    actual.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help=(
            "Threshold JSON file."
        ),
    )

    actual.add_argument(
        "--dark-below",
        type=float,
        default=None,
        help="Brightness threshold.",
    )

    actual.add_argument(
        "--bright-above",
        type=float,
        default=None,
        help="Brightness threshold.",
    )

    actual.add_argument(
        "--blur-below",
        type=float,
        default=None,
        help="Sharpness threshold.",
    )

    actual.add_argument(
        "--move",
        action="store_true",
        help=(
            "Move files instead of creating "
            "symbolic links."
        ),
    )

    actual.add_argument(
        "--face-model",
        type=Path,
        default=DEFAULT_FACE_MODEL,
        help="Path to the YuNet ONNX model.",
    )

    actual.add_argument(
        "--face-confidence",
        type=float,
        default=0.85,
        help="Minimum face confidence.",
    )

    actual.add_argument(
        "--face-nms-threshold",
        type=float,
        default=0.3,
        help="Face detector NMS threshold.",
    )

    actual.add_argument(
        "--face-top-k",
        type=int,
        default=5000,
        help="Maximum face detections.",
    )

    actual.add_argument(
        "--face-blur-below",
        type=float,
        default=45.0,
        help="Face sharpness threshold.",
    )

    actual.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of analysis threads.",
    )

    actual.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation.",
    )

    # Intern benötigte Option, um beim tatsächlichen
    # Klassifizieren manuelle Labels zu berücksichtigen.
    actual.add_argument(
        "--labels",
        type=Path,
        default=None,
        help=(
            "JSON file containing manual labels. "
            "Defaults to source/.foto-culler-learn/labels.json."
        ),
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "learn":
        return run_learn(args)

    if args.mode == "actually-do-it":
        return run_actual(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
