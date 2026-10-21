# Object Detection and Tracking

A computer vision project that detects objects in video frames and tracks them across consecutive frames using YOLO11, OpenCV, and ByteTrack.

The application supports webcam input and pre-recorded videos, displays bounding boxes and object labels with tracking IDs, and allows the user to save annotated video output.

## Features

- Object detection using a pretrained YOLO11 model
- Multi-object tracking using ByteTrack
- Webcam and video-file input through OpenCV
- Bounding boxes, class labels, and tracking IDs
- Configurable detection confidence threshold
- Manual saving of annotated video output

## Technologies Used

- Python 3.12
- OpenCV
- Ultralytics YOLO11
- ByteTrack
- PyTorch (used by the Ultralytics model stack)

## Project Structure

```text
Object_Detection_And_Tracking/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── videos/    # Local input videos; not tracked by Git
└── output/    # Generated annotated videos; not tracked by Git
```

The `videos/` and `output/` directories are excluded from version control to avoid uploading large video files and generated outputs. These directories may not appear after cloning the repository, so create them locally when needed.

## Getting Started

### Prerequisites

- Python 3.12.10 recommended
- pip
- An internet connection for the initial model download
- A webcam or an MP4 video file

### 1. Clone the Repository

```bash
git clone https://github.com/githubsmi/Object_Detection_And_Tracking.git
cd Object_Detection_And_Tracking
```

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run with a Webcam

```powershell
python main.py --source 0
```

### 5. Run with a Video File

Place a video in the `videos/` directory and run:

```powershell
python main.py --source "videos/sample.mp4"
```

Replace `sample.mp4` with the actual filename.

## How It Works

1. **Video capture:** OpenCV reads frames from a webcam or video file.
2. **Object detection:** YOLO11 identifies objects and predicts their classes and bounding boxes.
3. **Object tracking:** ByteTrack associates detections across frames and assigns tracking IDs when available.
4. **Visualization:** The application draws bounding boxes, class labels, and tracking IDs on the frames.
5. **Video export:** The user can save annotated video output for later review.

## Model and Tracking

The project uses pretrained YOLO11 weights for object detection and ByteTrack for multi-object tracking.

Detection confidence can be adjusted to filter out lower-confidence predictions. However, incorrect classifications may still occur, particularly with occluded objects, unusual camera angles, or visually similar objects.

Tracking IDs identify tracked objects within a video session; they do not establish a person's real-world identity.

## Output
The application displays annotated frames in an OpenCV window.
Press `S` to start saving the annotated video and `Q` to exit
the application. Recording begins when `S` is pressed, so frames
displayed before that point are not included in the saved file.

Saved videos are written to the `output/` directory. Output
filenames are generated to avoid overwriting existing recordings.

## Limitations

- Detection accuracy depends on the pretrained model and video quality.
- Incorrect object classifications can occur.
- Tracking IDs may change when objects disappear or tracking is interrupted.
- Processing speed depends on hardware, video resolution, and model size.
- A compatible desktop environment is required for the OpenCV display window.

## Future Improvements

- Add FPS and detection statistics.
- Provide object-class filtering through a user interface.
- Compare different YOLO model sizes.
- Evaluate detection precision, recall, and tracking performance.
- Add a configurable output-video naming and export interface.

## Author

Developed as a computer vision project for practical learning.
