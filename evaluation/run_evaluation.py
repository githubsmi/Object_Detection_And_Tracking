
import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = "yolo11n.pt"
IMAGE_SIZE = 640
BATCH_SIZE = 4
CONFIDENCE = 0.001
DEVICE = "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO11 object detection."
    )
    parser.add_argument(
        "--dataset",
        choices=["coco8", "coco128"],
        default="coco8",
        help="Dataset to evaluate (default: coco8).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_name = args.dataset
    dataset_yaml = f"{dataset_name}.yaml"
    run_name = f"{dataset_name}_yolo11n"

    print("=" * 55)
    print("YOLO11 OBJECT DETECTION EVALUATION")
    print("=" * 55)
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {dataset_yaml}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")
    print("\nLoading model...")

    model = YOLO(MODEL_PATH)

    print("Starting validation.")
    print(f"The dataset may be downloaded on the first run: {dataset_yaml}")

    metrics = model.val(
        data=dataset_yaml,
        split="val",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        conf=CONFIDENCE,
        device=DEVICE,
        plots=True,
        project=str(Path("results").resolve()),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    inference_ms = float(metrics.speed.get("inference", 0.0))
    inference_fps = (
        1000.0 / inference_ms if inference_ms > 0 else None
    )

    results_dir = Path(metrics.save_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "model": MODEL_PATH,
        "dataset": dataset_yaml,
        "split": "val",
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "device": DEVICE,
        "confidence_threshold": CONFIDENCE,
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "inference_time_ms_per_image": inference_ms,
        "approx_inference_fps": inference_fps,
        "note": (
            "Inference FPS is estimated from model inference timing "
            "only. It is not end-to-end video processing FPS. "
            "Interpret metrics in the context of the dataset size "
            "and evaluation split."
        ),
    }

    json_path = results_dir / "evaluation_metrics.json"
    csv_path = results_dir / "evaluation_metrics.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=report.keys())
        writer.writeheader()
        writer.writerow(report)

    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"mAP@50: {map50:.4f}")
    print(f"mAP@50-95: {map50_95:.4f}")

    if inference_fps is not None:
        print(f"Approx. inference FPS: {inference_fps:.2f}")

    print(f"\nResults directory: {results_dir.resolve()}")
    print(f"JSON report: {json_path.resolve()}")
    print(f"CSV report: {csv_path.resolve()}")
    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
