PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install install-system check clean

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install numpy pillow opencv-contrib-python
	$(BIN)/python -m py_compile foto_culler.py

install-system:
	sudo apt update
	sudo apt install -y python3 python3-tk python3-pil python3-pil.imagetk python3-opencv python3-numpy
	$(PYTHON) -m py_compile foto_culler.py

check:
	$(PYTHON) -m py_compile foto_culler.py
	$(PYTHON) -c "import cv2, numpy, PIL, tkinter; print('OpenCV:', cv2.__version__); print('FaceDetectorYN:', hasattr(cv2, 'FaceDetectorYN'))"

clean:
	rm -rf __pycache__ .pytest_cache
