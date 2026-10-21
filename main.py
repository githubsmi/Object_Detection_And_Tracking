
import argparse
from pathlib import Path
from datetime import datetime

import cv2
from ultralytics import YOLO


def get_unique_output_path(source_name):
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(source_name).stem if not source_name.isdigit() else "webcam"

    output_path = output_dir / f"{base_name}_tracked.mp4"
    counter = 1

    while output_path.exists():
        output_path = output_dir / f"{base_name}_tracked_{counter}.mp4"
        counter += 1

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--conf", type=float, default=0.50)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.source}")

    writer = None
    saved = False
    frame_count = 0
    output_path = None
    window_name = "Object Detection and Tracking"

    print("Press S to save the tracked video.")
    print("Press Q to quit.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=args.conf,
                verbose=False
            )

            annotated = results[0].plot()
            frame_count += 1

            # Keep frames in memory until the user chooses to save.
            if writer is not None:
                writer.write(annotated)

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if writer is None:
                    output_path = get_unique_output_path(args.source)
                    height, width = annotated.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    if not fps or fps <= 0:
                        fps = 30.0

                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height)
                    )

                    if not writer.isOpened():
                        writer.release()
                        writer = None
                        raise RuntimeError("Could not create output video.")

                    writer.write(annotated)
                    saved = True
                    print(f"Recording saved to: {output_path.resolve()}")

                else:
                    print("Saving is already enabled for this video.")

            if key == ord("q"):
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if saved:
        print(f"Saved video: {output_path.resolve()}")
    else:
        print("No video was saved.")


if __name__ == "__main__":
    main()
