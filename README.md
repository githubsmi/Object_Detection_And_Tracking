# Object Detection and Tracking

A computer vision project that combines YOLO11 object detection with ByteTrack multi-object tracking to detect and follow objects across video frames. Built with Python, OpenCV, Ultralytics, and PyTorch, the application supports webcam input and prerecorded videos, with options to display tracking results and save annotated footage.

The project also includes an evaluation pipeline for measuring object detection performance on COCO8 and COCO128 datasets, generating metrics reports and visualization plots.

## Features

- Object detection using a pretrained YOLO11 model
- Multi-object tracking using ByteTrack
- Webcam and prerecorded video input
- Bounding boxes, class labels, and tracking IDs
- Configurable detection confidence threshold
- Manual saving of annotated video output
- Dataset evaluation through a command-line interface
- Precision, recall, mAP@50, and mAP@50–95 metrics
- Evaluation reports in JSON and CSV formats
- Performance plots and confusion matrices
- Automated tests for evaluation reports and metric validity

## Technologies Used

- Python 3.12
- Ultralytics YOLO11
- ByteTrack
- OpenCV
- PyTorch
- pytest

## Project Structure

```text
Object_Detection_And_Tracking/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── evaluation/
│   └── run_evaluation.py
├── tests/
│   └── test_evaluation.py
├── results/
│   ├── coco8_yolo11n/
│   └── coco128_yolo11n/
├── videos/              # Local input videos
├── output/              # Saved annotated videos
├── datasets/            # Downloaded evaluation datasets
├── runs/                # Local Ultralytics artifacts
└── yolo11n.pt            # Downloaded model weights
```

Some directories and model weights are generated locally and are excluded from version control. Evaluation reports and selected result plots are included in the repository.

## Getting Started

### Prerequisites

- Python 3.12
- pip
- Internet access for downloading model weights and evaluation datasets on the first run
- A webcam or a compatible video file for running the application

### 1. Clone the Repository

```powershell
git clone https://github.com/githubsmi/Object_Detection_And_Tracking.git
cd Object_Detection_And_Tracking
```

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks virtual environment activation, follow the Python and PowerShell execution-policy guidance appropriate for your system.

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pretrained model weights are downloaded when required if they are not already available locally.

## Running the Application

### Webcam Input

```powershell
python main.py --source 0
```

### Prerecorded Video

Place a compatible video file in the local `videos/` directory, then run:

```powershell
python main.py --source "videos/sample.mp4"
```

Replace `sample.mp4` with your actual video filename.

The application displays annotated frames in an OpenCV window. Use the application's supported keyboard controls to save the annotated video or exit.

## How It Works

1. **Video capture:** OpenCV reads frames from a webcam or video file.
2. **Object detection:** YOLO11 predicts object classes, bounding boxes, and confidence scores.
3. **Object tracking:** ByteTrack associates detections across consecutive frames and maintains tracking IDs when possible.
4. **Visualization:** Bounding boxes, class labels, and tracking IDs are displayed on the video.
5. **Video export:** The user can save annotated footage for later review.

Tracking IDs are temporary identifiers within a tracking session. They do not establish a person's real-world identity.

## Object Detection Evaluation

The evaluation pipeline uses Ultralytics validation to measure the pretrained YOLO11n model on supported datasets. It saves evaluation metrics and plots for later inspection.

### Supported Datasets

| Dataset option | Purpose |
|---|---|
| `coco8` | Small dataset for quick evaluation and pipeline testing |
| `coco128` | Larger small-scale dataset for additional evaluation |

These datasets are useful for checking the evaluation workflow, but neither result should be treated as a substitute for evaluation on the full COCO validation benchmark.

### Run Evaluation

Evaluate on COCO8:

```powershell
python -m evaluation.run_evaluation --dataset coco8
```

Evaluate on COCO128:

```powershell
python -m evaluation.run_evaluation --dataset coco128
```

If no dataset option is supplied, the script defaults to COCO8.

On the first run, Ultralytics may download the selected dataset and create local cache files. Evaluation is configured to run on the CPU with a 640-pixel image size and batch size of 4.

### Evaluation Metrics

- **Precision:** The proportion of predicted positive detections that are correct.
- **Recall:** The proportion of ground-truth objects that are detected.
- **mAP@50:** Mean average precision at an IoU threshold of 0.50.
- **mAP@50–95:** Mean average precision averaged across IoU thresholds from 0.50 to 0.95.
- **Inference time:** The reported model inference time per image.
- **Approximate inference FPS:** An estimate derived from inference timing, not the application's end-to-end video processing frame rate.

### Evaluation Results

The following results were obtained during local CPU-based evaluation with the pretrained YOLO11n model.

| Metric | COCO8 | COCO128 |
|---|---:|---:|
| Validation images | 4 | 128 |
| Ground-truth instances | 17 | 929 |
| Precision | 0.5705 | 0.6436 |
| Recall | 0.8500 | 0.6148 |
| mAP@50 | 0.8543 | 0.6717 |
| mAP@50–95 | 0.6312 | 0.5028 |
| Approx. inference FPS | 9.66 | 14.87 |

These results are from separate evaluations on small datasets. Differences between them do not necessarily indicate a change in model quality: dataset size, object distribution, and image content affect the measured metrics. The inference FPS values are approximate and should not be interpreted as end-to-end tracking performance.

### Generated Results

Each evaluation run creates a dataset-specific directory under `results/`, containing files such as:

- `evaluation_metrics.json` — structured metrics report
- `evaluation_metrics.csv` — tabular metrics report
- `BoxPR_curve.png` — precision-recall curve
- `BoxP_curve.png` — precision curve
- `BoxR_curve.png` — recall curve
- `BoxF1_curve.png` — F1-score curve
- `confusion_matrix.png` — confusion matrix
- `confusion_matrix_normalized.png` — normalized confusion matrix
- Validation images showing ground-truth labels and model predictions

These artifacts help inspect model behavior and compare evaluation runs. The actual plots and reports are available in the corresponding directories in the repository.

## Running Tests

The project includes automated tests for the evaluation reports.

Run the tests with:

```powershell
python -m pytest .\tests -v
```

The current test suite contains seven tests covering report existence, required fields, metric ranges, CSV/JSON consistency, inference timing, and evaluation configuration.

All seven tests passed in the development environment.

## Limitations

- Evaluation results are based on small datasets and do not establish performance on the full COCO benchmark.
- Detection accuracy depends on the pretrained model, image quality, object scale, and occlusion.
- Tracking IDs may change when tracking is interrupted or objects disappear from view.
- Detection metrics do not directly measure tracking quality.
- Reported inference speed is hardware-dependent and does not represent total video processing speed.
- CPU processing can be slower than GPU-based inference.
- Displaying the live application requires a compatible desktop environment.

## Future Improvements

- Evaluate on the full COCO validation dataset.
- Compare different YOLO model sizes under consistent settings.
- Measure end-to-end video FPS and processing latency.
- Add tracking-specific metrics such as IDF1 and HOTA.
- Investigate performance under occlusion and crowded-scene conditions.
- Add object-class filtering and configurable video export options.

## Author

Developed as a computer vision portfolio project to explore object detection, multi-object tracking, and reproducible model evaluation.
