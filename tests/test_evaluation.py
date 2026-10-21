
import csv
import json
from pathlib import Path


RESULTS_DIR = Path("results/coco8_yolo11n")
JSON_REPORT = RESULTS_DIR / "evaluation_metrics.json"
CSV_REPORT = RESULTS_DIR / "evaluation_metrics.csv"

REQUIRED_FIELDS = {
    "model",
    "dataset",
    "split",
    "image_size",
    "batch_size",
    "device",
    "confidence_threshold",
    "precision",
    "recall",
    "mAP50",
    "mAP50_95",
    "inference_time_ms_per_image",
    "approx_inference_fps",
}


def load_json_report():
    with JSON_REPORT.open(encoding="utf-8") as file:
        return json.load(file)


def test_json_report_exists():
    assert JSON_REPORT.is_file()


def test_csv_report_exists():
    assert CSV_REPORT.is_file()


def test_json_contains_required_fields():
    report = load_json_report()
    assert REQUIRED_FIELDS.issubset(report.keys())


def test_metrics_are_within_valid_ranges():
    report = load_json_report()

    for field in ("precision", "recall", "mAP50", "mAP50_95"):
        assert 0.0 <= report[field] <= 1.0, (
            f"{field} must be between 0 and 1"
        )


def test_inference_time_is_positive():
    report = load_json_report()
    assert report["inference_time_ms_per_image"] > 0


def test_csv_contains_matching_metrics():
    with CSV_REPORT.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1

    csv_row = rows[0]
    json_report = load_json_report()

    for field in ("precision", "recall", "mAP50", "mAP50_95"):
        assert abs(float(csv_row[field]) - json_report[field]) < 1e-6


def test_report_uses_expected_evaluation_configuration():
    report = load_json_report()

    assert report["model"] == "yolo11n.pt"
    assert report["dataset"] == "coco8.yaml"
    assert report["split"] == "val"
    assert report["device"] == "cpu"
