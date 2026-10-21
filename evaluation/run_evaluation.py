
import csv
import json
from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = "yolo11n.pt"
DATASET = "coco8.yaml"
IMAGE_SIZE = 640
BATCH_SIZE = 4
CONFIDENCE = 0.001


def main():
    print("=" * 55)
    print("YOLO11 OBJECT DETECTION EVALUATION")
    print("=" * 55)

    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print("Device: CPU")
    print("\nLoading model...")

    model = YOLO(MODEL_PATH)

    print("Starting validation.")
    print("The first run may download the COCO8 dataset.")

    metrics = model.val(
        data=DATASET,
        split="val",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        conf=CONFIDENCE,
        device="cpu",
        plots=True,
        project=str(Path("results").resolve()),
        name="coco8_yolo11n",
        exist_ok=True,
        verbose=True,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    inference_ms = float(
        metrics.speed.get("inference", 0.0)
    )

    inference_fps = (
        1000.0 / inference_ms
        if inference_ms > 0
        else None
    )

    results_dir = Path(metrics.save_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "model": MODEL_PATH,
        "dataset": DATASET,
        "split": "val",
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "device": "cpu",
        "confidence_threshold": CONFIDENCE,
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "inference_time_ms_per_image": inference_ms,
        "approx_inference_fps": inference_fps,
        "note": (
            "Inference FPS is estimated from the model's "
            "inference timing only. It is not end-to-end "
            "video processing FPS."
        ),
    }

    json_path = results_dir / "evaluation_metrics.json"
    csv_path = results_dir / "evaluation_metrics.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=report.keys()
        )
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
