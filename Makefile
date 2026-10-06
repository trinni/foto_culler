SHELL := /bin/bash

PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
VENV_DIR ?= $(PROJECT_DIR)/.venv
PYTHON ?= python3
BIN_NAME ?= foto_culler

MODEL_DIR ?= $(HOME)/.local/share/foto-culler
MODEL_FILE := $(MODEL_DIR)/face_detection_yunet_2023mar.onnx
MODEL_URL := https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

.PHONY: help install install-user install-system venv requirements model clean

help:
	@echo "Typical first installation:"
	@echo "  make install"
	@echo
	@echo "Other targets:"
	@echo "  make install-user     Install into a user PATH directory"
	@echo "  make install-system   Install into /usr/local/bin"
	@echo "  make venv             Create/update the virtual environment"
	@echo "  make requirements     Install Python packages into the venv"
	@echo "  make model            Download the YuNet face model"
	@echo "  make clean            Remove installed files after confirmation"

install: requirements model
	@set -euo pipefail; \
	if [[ "$${EUID:-$$(id -u)}" -eq 0 ]]; then \
		echo "Root execution detected."; \
		$(MAKE) install-system; \
	else \
		$(MAKE) install-user; \
	fi

venv:
	@set -euo pipefail; \
	if [[ ! -x "$(VENV_DIR)/bin/python" ]]; then \
		echo "Creating virtual environment: $(VENV_DIR)"; \
		$(PYTHON) -m venv "$(VENV_DIR)"; \
	fi; \
	echo "Updating pip in $(VENV_DIR)"; \
	"$(VENV_DIR)/bin/python" -m pip install --upgrade pip

requirements: venv
	@set -euo pipefail; \
	echo "Installing Python packages into $(VENV_DIR)"; \
	"$(VENV_DIR)/bin/python" -m pip install \
		-r "$(PROJECT_DIR)/requirements.txt"

model:
	@set -euo pipefail; \
	mkdir -p "$(MODEL_DIR)"; \
	if [[ ! -s "$(MODEL_FILE)" ]]; then \
		if ! command -v wget >/dev/null 2>&1 && \
		   ! command -v curl >/dev/null 2>&1; then \
			echo "ERROR: wget or curl is required." >&2; \
			exit 1; \
		fi; \
		echo "Downloading YuNet model..."; \
		if command -v curl >/dev/null 2>&1; then \
			curl \
				--fail \
				--location \
				--progress-bar \
				--output "$(MODEL_FILE).part" \
				"$(MODEL_URL)"; \
		else \
			wget \
				-O "$(MODEL_FILE).part" \
				"$(MODEL_URL)"; \
		fi; \
		if [[ ! -s "$(MODEL_FILE).part" ]]; then \
			echo "ERROR: YuNet model download failed." >&2; \
			rm -f "$(MODEL_FILE).part"; \
			exit 1; \
		fi; \
		mv "$(MODEL_FILE).part" "$(MODEL_FILE)"; \
	else \
		echo "YuNet model already exists:"; \
		echo "  $(MODEL_FILE)"; \
	fi

install-user: requirements model
	@set -euo pipefail; \
	echo; \
	echo "Select the installation directory for $(BIN_NAME)."; \
	echo; \
	echo "Detected PATH entries:"; \
	path_value="$${PATH:-}"; \
	old_ifs="$$IFS"; \
	IFS=':'; \
	entries=(); \
	index=1; \
	for entry in $$path_value; do \
		entry="$${entry:-.}"; \
		case "$$entry" in \
			/*) ;; \
			*) continue ;; \
		esac; \
		if [[ -d "$$entry" ]]; then \
			status="[exists]"; \
		else \
			status="[will be created]"; \
		fi; \
		printf "  %d) %s %s\n" \
			"$$index" "$$entry" "$$status"; \
		entries+=("$$entry"); \
		((index++)); \
	done; \
	IFS="$$old_ifs"; \
	echo; \
	printf "  u) %s [recommended for user installation]\n" \
		"$$HOME/bin"; \
	echo; \
	read -r -p "Choose a number or enter u: " choice; \
	if [[ "$$choice" == "u" || "$$choice" == "U" ]]; then \
		install_dir="$$HOME/bin"; \
	elif [[ "$$choice" =~ ^[0-9]+$$ ]] && \
	     (( choice >= 1 && choice <= $${#entries[@]} )); then \
		install_dir="$${entries[$$((choice - 1))]}"; \
	else \
		echo "Invalid installation choice." >&2; \
		exit 1; \
	fi; \
	mkdir -p "$$install_dir"; \
	install \
		-m 0755 \
		"$(PROJECT_DIR)/foto_culler" \
		"$$install_dir/foto_culler"; \
	echo; \
	echo "Installed $(BIN_NAME) to:"; \
	echo "  $$install_dir/foto_culler"; \
	case ":$${PATH:-}:" in \
		*:$$install_dir:*) ;; \
		*) \
			echo; \
			echo "Note: $$install_dir is not currently in PATH."; \
			echo "Add it with:"; \
			echo "  export PATH=\"$$install_dir:\$$PATH\""; \
			;; \
	esac

install-system: requirements model
	@set -euo pipefail; \
	install_dir="/usr/local/bin"; \
	echo "Installing $(BIN_NAME) to:"; \
	echo "  $$install_dir/foto_culler"; \
	install \
		-D \
		-m 0755 \
		"$(PROJECT_DIR)/foto_culler" \
		"$$install_dir/foto_culler"; \
	echo; \
	echo "Installed $(BIN_NAME) to:"; \
	echo "  $$install_dir/foto_culler"

clean:
	@set -euo pipefail; \
	path_value="$${PATH:-}"; \
	old_ifs="$$IFS"; \
	IFS=':'; \
	wrappers=(); \
	for entry in $$path_value; do \
		entry="$${entry:-.}"; \
		if [[ -f "$$entry/foto_culler" || \
		      -L "$$entry/foto_culler" ]]; then \
			wrappers+=("$$entry/foto_culler"); \
		fi; \
	done; \
	IFS="$$old_ifs"; \
	if [[ "$${EUID:-$$(id -u)}" -eq 0 && \
	      -f "/usr/local/bin/foto_culler" ]]; then \
		already=0; \
		for item in "$${wrappers[@]}"; do \
			if [[ "$$item" == \
			      "/usr/local/bin/foto_culler" ]]; then \
				already=1; \
			fi; \
		done; \
		if (( already == 0 )); then \
			wrappers+=("/usr/local/bin/foto_culler"); \
		fi; \
	fi; \
	echo "Installed wrapper paths:"; \
	if (( $${#wrappers[@]} == 0 )); then \
		echo "  none found"; \
	else \
		printf "  %s\n" "$${wrappers[@]}"; \
	fi; \
	echo; \
	echo "Other project-related paths:"; \
	echo "  Virtual environment: $(VENV_DIR)"; \
	echo "  YuNet model:         $(MODEL_FILE)"; \
	residuals=(); \
	for item in "$(VENV_DIR)" "$(MODEL_FILE)"; do \
		if [[ -e "$$item" || -L "$$item" ]]; then \
			residuals+=("$$item"); \
		fi; \
	done; \
	if (( $${#wrappers[@]} == 0 && \
	      $${#residuals[@]} == 0 )); then \
		echo; \
		echo "Nothing to uninstall was found."; \
		exit 0; \
	fi; \
	echo; \
	echo "The following paths can be uninstalled:"; \
	for item in "$${wrappers[@]}" "$${residuals[@]}"; do \
		if [[ -n "$$item" ]]; then \
			echo "  $$item"; \
		fi; \
	done; \
	echo; \
	echo "The project source directory and source photos"; \
	echo "will not be removed."; \
	echo; \
	read -r -p \
		"Acknowledge deinstallation by typing REMOVE: " \
		confirmation; \
	if [[ "$$confirmation" != "REMOVE" ]]; then \
		echo "Deinstallation cancelled."; \
		exit 0; \
	fi; \
	for item in "$${wrappers[@]}"; do \
		if [[ -e "$$item" || -L "$$item" ]]; then \
			if [[ -w "$$item" || \
			      -w "$$(dirname "$$item")" ]]; then \
				rm -f -- "$$item"; \
				echo "Removed: $$item"; \
			else \
				echo "Permission denied: $$item" >&2; \
			fi; \
		fi; \
	done; \
	if [[ -d "$(VENV_DIR)" ]]; then \
		rm -rf -- "$(VENV_DIR)"; \
		echo "Removed: $(VENV_DIR)"; \
	fi; \
	if [[ -f "$(MODEL_FILE)" || \
	      -L "$(MODEL_FILE)" ]]; then \
		rm -f -- "$(MODEL_FILE)"; \
		echo "Removed: $(MODEL_FILE)"; \
	fi; \
	if [[ -d "$(MODEL_DIR)" ]] && \
	   [[ -z "$$(find "$(MODEL_DIR)" \
	      -mindepth 1 \
	      -maxdepth 1 \
	      -print \
	      -quit)" ]]; then \
		rmdir -- "$(MODEL_DIR)"; \
		echo "Removed empty directory: $(MODEL_DIR)"; \
	fi; \
	if [[ -d "$(HOME)/.local/share/foto-culler" ]] && \
	   [[ -z "$$(find "$(HOME)/.local/share/foto-culler" \
	      -mindepth 1 \
	      -maxdepth 1 \
	      -print \
	      -quit)" ]]; then \
		rmdir -- "$(HOME)/.local/share/foto-culler"; \
		echo "Removed empty directory: \
$(HOME)/.local/share/foto-culler"; \
	fi
