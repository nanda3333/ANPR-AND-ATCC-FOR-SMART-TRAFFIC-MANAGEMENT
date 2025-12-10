# ANPR-AND-ATCC-FOR-SMART-TRAFFIC-MANAGEMENT

# Automated Vision System: ANPR & Traffic Classification

This repository contains two main components for automated vision tasks:

1.  A **Streamlit web application (`app.py`)** for real-time inference, featuring **Automatic Number Plate Recognition (ANPR)** and **Automatic Traffic Count and Classification (ATCC)**.
2.  **Jupyter notebooks** detailing the training and data preparation processes for the underlying **YOLOv8** detection models.

***

## 🚀 Features

* **ANPR (Automatic Number Plate Recognition):** Detects license plates in images or videos using a trained YOLO model (`yolo_ANPR.pt`) and performs Optical Character Recognition (OCR) on the detected plates.
* **ATCC (Automatic Traffic Count and Classification):** Detects and classifies various traffic objects (e.g., cars, pedestrians, signs) for counting and analysis, using a trained YOLO model (`yolo_ATCC.pt`).
* **Interactive UI:** A user-friendly interface built with **Streamlit** to easily upload and process files.
* **Data Logging:** Generates a downloadable **CSV log** of all detections during video processing.

***

## 🛠️ Setup and Installation

### Prerequisites

You need a Python environment (**Python 3.8+** recommended).

### 1. Install Python Dependencies

The core application and training processes rely on the following libraries:

| Dependency | Purpose |
| :--- | :--- |
| `ultralytics` | Core library for YOLOv8 model handling and training. |
| `streamlit` | Used to run the interactive web application (`app.py`). |
| `fast_plate_ocr` | OCR library used specifically for the ANPR functionality in `app.py`. |
| `opencv-python`, `numpy`, `pillow`, `pandas` | General computer vision and data processing utilities. |

```bash
# Recommended installation for Streamlit app and basic usage
pip install -qU ultralytics opencv-python pillow pandas streamlit

# Install OCR library (needed for ANPR functionality)
pip install fast_plate_ocr

# If replicating training, ensure notebook dependencies are met
pip install numpy opencv-python-headless
