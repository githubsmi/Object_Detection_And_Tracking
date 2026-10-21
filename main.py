
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def get_unique_output_path(source_name):
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(source_name)
    base_name = (
        "webcam"
        if source_name.isdigit()
        else source_path.stem
    )

    output_path = output_dir / f"{base_name}_tracked.mp4"
    counter = 1

    while output_path.exists():
        output_path = (
            output_dir / f"{base_name}_tracked_{counter}.mp4"
        )
        counter += 1

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect and track objects in video using YOLO11 and ByteTrack."
    )

    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index or video file path (default: 0).",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="Detection confidence threshold from 0 to 1 (default: 0.50).",
    )

    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="YOLO model weights (default: yolo11n.pt).",
    )

    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker configuration (default: bytetrack.yaml).",
    )

    args = parser.parse_args()

    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1.")

    return args


def main():
    args = parse_args()

    source = (
        int(args.source)
        if args.source.isdigit()
        else args.source
    )

    model = YOLO(args.model)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Cannot open video source: {args.source}. "
            "Check the webcam index or video file path."
        )

    writer = None
    output_path = None
    saved = False
    window_name = "Object Detection and Tracking"

    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Tracker: {args.tracker}")
    print("Press S to start saving the annotated video.")
    print("Press Q to quit.")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            results = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                conf=args.conf,
                verbose=False,
            )

            annotated = results[0].plot()

            if writer is not None:
                writer.write(annotated)

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if writer is None:
                    output_path = get_unique_output_path(args.source)

                    height, width = annotated.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    if fps <= 0:
                        fps = 30.0

                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )

                    if not writer.isOpened():
                        writer.release()
                        writer = None
                        raise RuntimeError(
                            "Could not create output video. "
                            "Check the output directory and video codec."
                        )

                    # Include the frame on which saving begins.
                    writer.write(annotated)
                    saved = True
                    print(f"Saving video to: {output_path.resolve()}")

                else:
                    print("Saving is already enabled.")

            if key == ord("q"):
                break

    finally:
        cap.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

    if saved and output_path is not None:
        print(f"Saved video: {output_path.resolve()}")
    else:
        print("No video was saved.")


if __name__ == "__main__":
    main()
