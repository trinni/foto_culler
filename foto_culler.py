#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
DEFAULT_FACE_MODEL = Path.home() / ".local/share/photo-culler/face_detection_yunet_2023mar.onnx"


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
        (p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: str(p).lower(),
    )


def calculate_metrics(path: Path) -> Optional[PhotoMetrics]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    height, width = image.shape
    max_dimension = max(width, height)
    if max_dimension > 1600:
        scale = 1600 / max_dimension
        image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    return PhotoMetrics(
        path=str(path),
        brightness=float(np.mean(image)),
        sharpness=float(cv2.Laplacian(image, cv2.CV_64F).var()),
        contrast=float(np.std(image)),
        width=width,
        height=height,
    )


def load_metrics(paths: list[Path], workers: Optional[int] = None) -> list[PhotoMetrics]:
    if not paths:
        return []
    workers = max(1, workers or default_worker_count())
    results: list[Optional[PhotoMetrics]] = [None] * len(paths)
    print(f"Analyzing {len(paths)} images with {workers} threads...")
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="photo-analysis") as executor:
        futures = {executor.submit(calculate_metrics, path): index for index, path in enumerate(paths)}
        for completed, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as error:
                print(f"\nError analyzing {paths[index]}: {error}", file=sys.stderr)
            print(f"\rAnalyzing {completed}/{len(paths)}", end="", flush=True)
    print()
    return [metric for metric in results if metric is not None]


def crop_with_padding(image: np.ndarray, x: int, y: int, width: int, height: int, padding: float = 0.15) -> Optional[np.ndarray]:
    image_height, image_width = image.shape[:2]
    pad_x, pad_y = int(width * padding), int(height * padding)
    left, top = max(0, x - pad_x), max(0, y - pad_y)
    right, bottom = min(image_width, x + width + pad_x), min(image_height, y + height + pad_y)
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right]


def calculate_face_sharpness(face_image: Optional[np.ndarray]) -> float:
    if face_image is None or face_image.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
    max_dimension = max(gray.shape[:2])
    if max_dimension > 500:
        scale = 500 / max_dimension
        gray = cv2.resize(gray, (max(1, int(gray.shape[1] * scale)), max(1, int(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class YuNetDetector:
    def __init__(self, model_path: Path, confidence: float, nms_threshold: float, top_k: int):
        if not model_path.exists():
            raise FileNotFoundError(f"YuNet model not found: {model_path}")
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("This OpenCV build does not provide FaceDetectorYN.")
        self.model_path = model_path
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.top_k = top_k

    def detect(self, image: np.ndarray, face_blur_below: float) -> FaceMetrics:
        height, width = image.shape[:2]
        detector = cv2.FaceDetectorYN.create(str(self.model_path), "", (width, height), self.confidence, self.nms_threshold, self.top_k)
        _, faces = detector.detect(image)
        if faces is None:
            return FaceMetrics(0, 0, 0, 0.0, 0.0, 0.0)
        sharp_faces = blurry_faces = 0
        max_sharpness = max_blurry = max_confidence = 0.0
        for face in faces:
            x, y = max(0, int(round(face[0]))), max(0, int(round(face[1])))
            face_width, face_height = max(1, int(round(face[2]))), max(1, int(round(face[3])))
            max_confidence = max(max_confidence, float(face[14]))
            face_image = crop_with_padding(image, x, y, face_width, face_height)
            face_sharpness = calculate_face_sharpness(face_image)
            max_sharpness = max(max_sharpness, face_sharpness)
            if face_sharpness >= face_blur_below:
                sharp_faces += 1
            else:
                blurry_faces += 1
                max_blurry = max(max_blurry, face_sharpness)
        return FaceMetrics(len(faces), sharp_faces, blurry_faces, max_sharpness, max_blurry, max_confidence)


def detect_faces_for_metric(metric: PhotoMetrics, detector: YuNetDetector, face_blur_below: float) -> PhotoMetrics:
    image = cv2.imread(metric.path, cv2.IMREAD_COLOR)
    if image is None:
        return metric
    max_dimension = max(image.shape[:2])
    if max_dimension > 1600:
        scale = 1600 / max_dimension
        image = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    face_metrics = detector.detect(image, face_blur_below)
    metric.face_count = face_metrics.count
    metric.sharp_faces = face_metrics.sharp_faces
    metric.blurry_faces = face_metrics.blurry_faces
    metric.max_face_sharpness = face_metrics.max_sharpness
    metric.max_blurry_face_sharpness = face_metrics.max_blurry_face_sharpness
    metric.face_confidence = face_metrics.face_confidence
    return metric


def load_face_metrics(metrics: list[PhotoMetrics], model_path: Path, face_confidence: float, face_nms_threshold: float, face_top_k: int, face_blur_below: float, workers: Optional[int] = None) -> list[PhotoMetrics]:
    if not metrics:
        return metrics
    workers = max(1, workers or default_worker_count())
    results: list[Optional[PhotoMetrics]] = [None] * len(metrics)
    print(f"Detecting faces in {len(metrics)} images with {workers} threads...")

    def process(metric: PhotoMetrics) -> PhotoMetrics:
        detector = YuNetDetector(model_path, face_confidence, face_nms_threshold, face_top_k)
        return detect_faces_for_metric(metric, detector, face_blur_below)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="face-analysis") as executor:
        futures = {executor.submit(process, metric): index for index, metric in enumerate(metrics)}
        for completed, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as error:
                print(f"\nError detecting faces in {metrics[index].path}: {error}", file=sys.stderr)
                results[index] = metrics[index]
            print(f"\rFace detection {completed}/{len(metrics)}", end="", flush=True)
    print()
    return [metric for metric in results if metric is not None]


def percentile(values: list[float], percentage: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentage))


def calculate_thresholds(metrics: list[PhotoMetrics], labels: dict[str, str]) -> dict[str, float]:
    dark = [m.brightness for m in metrics if labels.get(m.path) == "too_dark"]
    bright = [m.brightness for m in metrics if labels.get(m.path) == "too_bright"]
    blurry = [m.sharpness for m in metrics if labels.get(m.path) == "blurry"]
    good = [m for m in metrics if labels.get(m.path) == "ok"]
    thresholds: dict[str, float] = {}
    if dark:
        thresholds["dark_below"] = percentile(dark, 95)
    if bright:
        thresholds["bright_above"] = percentile(bright, 5)
    if blurry:
        thresholds["blur_below"] = percentile(blurry, 95)
    if good:
        good_brightness = [m.brightness for m in good]
        good_sharpness = [m.sharpness for m in good]
        if "dark_below" in thresholds:
            thresholds["dark_below"] = min(thresholds["dark_below"], percentile(good_brightness, 5))
        if "bright_above" in thresholds:
            thresholds["bright_above"] = max(thresholds["bright_above"], percentile(good_brightness, 95))
        if "blur_below" in thresholds:
            thresholds["blur_below"] = min(thresholds["blur_below"], percentile(good_sharpness, 5))
    return thresholds


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def write_report(path: Path, metrics: list[PhotoMetrics], labels: Optional[dict[str, str]] = None, classifications: Optional[dict[str, str]] = None) -> None:
    labels = labels or {}
    classifications = classifications or {}
    rows = []
    for metric in metrics:
        row = asdict(metric)
        row["learn_label"] = labels.get(metric.path, "")
        row["classification"] = classifications.get(metric.path, "")
        rows.append(row)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


class KeyTestWindow:
    REQUIRED_KEYS = {"space": "Space", "shift_r": "Right Shift", "control_r": "Right Control"}

    def __init__(self, root: tk.Tk):
        self.root = root
        self.detected: set[str] = set()
        self.result = False
        root.title("Keyboard test")
        root.geometry("720x300")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self.abort)
        tk.Label(root, text="Press Space, Right Shift and Right Control", font=("Sans", 16)).pack(pady=(30, 20))
        self.status = tk.Label(root, text="", font=("Sans", 12), justify=tk.LEFT)
        self.status.pack(pady=10)
        tk.Label(root, text="Press Escape to cancel", fg="#555555").pack(pady=10)
        root.bind("<KeyPress>", self.on_key)
        root.focus_force()
        self.update_status()

    def normalize(self, event) -> Optional[str]:
        if event.keysym == "space":
            return "space"
        if event.keysym == "Shift_R" or getattr(event, "keysym_num", None) == 65506:
            return "shift_r"
        if event.keysym == "Control_R" or getattr(event, "keysym_num", None) == 65508:
            return "control_r"
        return None

    def on_key(self, event):
        if event.keysym == "Escape":
            self.abort()
            return "break"
        key = self.normalize(event)
        if key:
            self.detected.add(key)
            self.update_status()
        if self.detected == set(self.REQUIRED_KEYS):
            self.result = True
            self.root.after(350, self.finish)
        return "break"

    def update_status(self):
        self.status.configure(text="\n".join(("✓" if key in self.detected else "○") + " " + label for key, label in self.REQUIRED_KEYS.items()))

    def finish(self):
        self.root.destroy()

    def abort(self):
        self.result = False
        self.root.destroy()


def run_key_test() -> bool:
    root = tk.Tk()
    window = KeyTestWindow(root)
    root.mainloop()
    return window.result


class LearnWindow:
    def __init__(self, root: tk.Tk, metrics: list[PhotoMetrics], labels: dict[str, str], preview_width: int = 1100, preview_height: int = 760):
        self.root = root
        self.metrics = metrics
        self.labels = labels
        self.preview_width = preview_width
        self.preview_height = preview_height
        self.index = 0
        self.photo_image = None
        self.finished = False
        root.title("Photo Culler - Learn mode")
        root.geometry(f"{preview_width}x{preview_height + 120}")
        root.protocol("WM_DELETE_WINDOW", self.abort)
        self.image_label = tk.Label(root, background="#202020")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 4))
        self.info_label = tk.Label(root, text="", anchor="w", justify=tk.LEFT, font=("Sans", 11))
        self.info_label.pack(fill=tk.X, padx=10)
        tk.Label(root, text="Left: back | Right: next/OK | Up: too bright | Down: too dark | Space/Right Shift/Right Control: blurry | Esc: cancel", anchor="w", fg="#555555").pack(fill=tk.X, padx=10, pady=(2, 8))
        root.bind("<KeyPress>", self.handle_key)
        root.focus_force()
        self.show_current()

    def current_metric(self) -> Optional[PhotoMetrics]:
        return self.metrics[self.index] if 0 <= self.index < len(self.metrics) else None

    def show_current(self):
        metric = self.current_metric()
        if metric is None:
            self.finish()
            return
        path = Path(metric.path)
        try:
            with Image.open(path) as source:
                image = source.copy()
            image.thumbnail((self.preview_width - 30, self.preview_height - 30), Image.Resampling.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.photo_image, text="")
        except Exception as error:
            self.photo_image = None
            self.image_label.configure(image="", text=f"Could not load image:\n{error}", fg="white", background="#202020")
        self.info_label.configure(text=f"{self.index + 1}/{len(self.metrics)}   {path.name}\nbrightness: {metric.brightness:.2f}   sharpness: {metric.sharpness:.2f}   contrast: {metric.contrast:.2f}\nCurrent rating: {self.labels.get(metric.path, 'not rated')}")

    def handle_key(self, event):
        key = event.keysym
        key_num = getattr(event, "keysym_num", None)
        if key == "Escape":
            self.abort()
        elif key == "Left":
            self.go_back()
        elif key == "Right":
            self.go_forward()
        elif key == "Up":
            self.mark_and_advance("too_bright")
        elif key == "Down":
            self.mark_and_advance("too_dark")
        elif key == "space" or key == "Shift_R" or key == "Control_R" or key_num in {65506, 65508}:
            self.mark_and_advance("blurry")
        return "break"

    def go_back(self):
        if self.index > 0:
            self.index -= 1
        self.show_current()

    def go_forward(self):
        metric = self.current_metric()
        if metric is not None:
            self.labels.setdefault(metric.path, "ok")
            self.index += 1
            self.show_current()

    def mark_and_advance(self, label: str):
        metric = self.current_metric()
        if metric is not None:
            self.labels[metric.path] = label
            self.index += 1
            self.show_current()

    def finish(self):
        self.finished = True
        self.root.destroy()

    def abort(self):
        self.finished = False
        self.root.destroy()


def classify(metric: PhotoMetrics, thresholds: dict[str, float]) -> str:
    if "dark_below" in thresholds and metric.brightness < thresholds["dark_below"]:
        return "too_dark"
    if "bright_above" in thresholds and metric.brightness > thresholds["bright_above"]:
        return "too_bright"
    if "blur_below" in thresholds and metric.sharpness < thresholds["blur_below"]:
        return "blurry"
    return "ok"


def classify_face(metric: PhotoMetrics) -> str:
    return "sharp_face" if metric.face_count > 0 and metric.sharp_faces > 0 else "no_face_or_blurry_face"


def unique_target(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def move_classified(metrics: list[PhotoMetrics], classifications: dict[str, str], output_dir: Path) -> dict[str, int]:
    counters: dict[str, int] = {}
    for metric in metrics:
        source = Path(metric.path)
        classification = classifications[metric.path]
        target_dir = output_dir / classification
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(unique_target(target_dir / source.name)))
        counters[classification] = counters.get(classification, 0) + 1
    return counters


def print_thresholds(thresholds: dict[str, float]):
    print("\nLearned thresholds:")
    print("--------------------------------")
    for key, option in (("dark_below", "--dark-below"), ("bright_above", "--bright-above"), ("blur_below", "--blur-below")):
        if key in thresholds:
            print(f"{option} {thresholds[key]:.4f}")
        else:
            print(f"No threshold learned for {key}.")
    print("--------------------------------")


def run_learn(args) -> int:
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels_path = output / "labels.json"
    thresholds_path = output / "thresholds.json"
    report_path = output / "learn-report.csv"
    all_paths = collect_images(source)
    if not all_paths:
        print("No supported image files found.")
        return 1
    labels: dict[str, str] = {}
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as file:
            labels = json.load(file)
        print(f"Loaded {len(labels)} existing ratings.")
    pending_paths = [path for path in all_paths if str(path) not in labels]
    if not pending_paths:
        print("All images are already rated.")
        print("Delete labels.json if you want to start from scratch.")
        metrics = load_metrics(all_paths, args.workers) if args.rebuild_report else []
        if metrics:
            write_report(report_path, metrics, labels=labels)
        return 0
    print(f"{len(all_paths)} images found; {len(pending_paths)} still need rating.")
    print("\nKeyboard test: press Space, Right Shift and Right Control.")
    if not run_key_test():
        print("Keyboard test cancelled.")
        return 0
    pending_metrics = load_metrics(pending_paths, args.workers)
    root = tk.Tk()
    window = LearnWindow(root, pending_metrics, labels)
    root.mainloop()
    write_json(labels_path, labels)
    write_report(report_path, pending_metrics, labels=labels)
    if not window.finished:
        print(f"Progress saved to {labels_path}")
        return 0
    missing = [str(path) for path in pending_paths if str(path) not in labels]
    if missing:
        print(f"{len(missing)} images are still unrated.")
        return 1
    all_metrics = load_metrics(all_paths, args.workers)
    thresholds = calculate_thresholds(all_metrics, labels)
    write_json(thresholds_path, thresholds)
    write_report(report_path, all_metrics, labels=labels)
    print_thresholds(thresholds)
    print(f"Saved ratings: {labels_path}")
    print(f"Saved thresholds: {thresholds_path}")
    print(f"Saved report: {report_path}")
    return 0


def run_actual(args) -> int:
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = collect_images(source)
    if not paths:
        print("No supported image files found.")
        return 1
    thresholds: dict[str, float] = {}
    if args.thresholds:
        with args.thresholds.resolve().open("r", encoding="utf-8") as file:
            thresholds = json.load(file)
    if args.dark_below is not None:
        thresholds["dark_below"] = args.dark_below
    if args.bright_above is not None:
        thresholds["bright_above"] = args.bright_above
    if args.blur_below is not None:
        thresholds["blur_below"] = args.blur_below
    if not thresholds:
        print("No thresholds supplied.")
        return 1
    print_thresholds(thresholds)
    if not args.yes:
        answer = input("Files will be moved. Continue? [yes/NO] ").strip().lower()
        if answer not in {"yes", "y"}:
            print("Cancelled.")
            return 0
    metrics = load_metrics(paths, args.workers)
    classifications = {metric.path: classify(metric, thresholds) for metric in metrics}
    technical_bad = [metric for metric in metrics if classifications[metric.path] != "ok"]
    technical_good = [metric for metric in metrics if classifications[metric.path] == "ok"]
    if technical_good:
        technical_good = load_face_metrics(technical_good, args.face_model.expanduser().resolve(), args.face_confidence, args.face_nms_threshold, args.face_top_k, args.face_blur_below, args.workers)
    report_path = output / "actual-report.csv"
    write_report(report_path, metrics, classifications=classifications)
    counters = move_classified(technical_bad, classifications, output / "technical")
    face_classifications = {metric.path: classify_face(metric) for metric in technical_good}
    face_counters = move_classified(technical_good, face_classifications, output / "ok")
    print("\nTechnical classification:")
    for key, value in sorted(counters.items()):
        print(f"  {key}: {value}")
    print("\nFace classification:")
    for key, value in sorted(face_counters.items()):
        print(f"  {key}: {value}")
    print(f"\nReport: {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive, multithreaded photo culler for Linux")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    learn = subparsers.add_parser("learn", help="Review and rate images")
    learn.add_argument("source", type=Path)
    learn.add_argument("output", type=Path)
    learn.add_argument("--workers", type=int, default=None)
    learn.add_argument("--rebuild-report", action="store_true", help="Rebuild the report when all images are already rated")
    actual = subparsers.add_parser("actually-do-it", help="Classify and move images")
    actual.add_argument("source", type=Path)
    actual.add_argument("output", type=Path)
    actual.add_argument("--thresholds", type=Path)
    actual.add_argument("--dark-below", type=float)
    actual.add_argument("--bright-above", type=float)
    actual.add_argument("--blur-below", type=float)
    actual.add_argument("--face-model", type=Path, default=DEFAULT_FACE_MODEL)
    actual.add_argument("--face-confidence", type=float, default=0.85)
    actual.add_argument("--face-nms-threshold", type=float, default=0.3)
    actual.add_argument("--face-top-k", type=int, default=5000)
    actual.add_argument("--face-blur-below", type=float, default=45.0)
    actual.add_argument("--workers", type=int, default=None)
    actual.add_argument("--yes", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "learn":
        return run_learn(args)
    if args.mode == "actually-do-it":
        return run_actual(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
