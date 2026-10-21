
import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Generate MOTChallenge tracking predictions using YOLO and ByteTrack."
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker configuration")
    parser.add_argument("--output", required=True, help="Output MOTChallenge TXT path")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    total_detections = 0

    try:
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            while True:
                success, frame = capture.read()
                if not success:
                    break

                frame_number += 1

                result = model.track(
                    frame,
                    persist=True,
                    tracker=args.tracker,
                    conf=args.conf,
                    classes=[0],
                    imgsz=320,
                    device="cpu",
                    half=False,
                    verbose=False,
                )[0]

                boxes = result.boxes

                if boxes is None or boxes.id is None:
                    continue

                xyxy = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()

                for coords, track_id, confidence in zip(
                    xyxy, track_ids, confidences
                ):
                    x1, y1, x2, y2 = map(float, coords)
                    width = x2 - x1
                    height = y2 - y1

                    if width <= 0 or height <= 0:
                        continue

                    writer.writerow([
                        frame_number,
                        track_id,
                        f"{x1:.2f}",
                        f"{y1:.2f}",
                        f"{width:.2f}",
                        f"{height:.2f}",
                        f"{confidence:.6f}",
                        -1,
                        -1,
                        -1,
                    ])
                    total_detections += 1

                if frame_number % 50 == 0:
                    print(f"Processed {frame_number}/{expected_frames} frames")

    finally:
        capture.release()

    print(f"\nVideo: {video_path}")
    print(f"Frames processed: {frame_number}/{expected_frames}")
    print(f"Person track detections written: {total_detections}")
    print(f"Prediction file: {output_path}")

    if frame_number != expected_frames:
        raise RuntimeError(
            f"Frame count mismatch: expected {expected_frames}, processed {frame_number}"
        )


if __name__ == "__main__":
    main()
