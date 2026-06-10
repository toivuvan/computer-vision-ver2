import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert predictions JSON to submission CSV format (class names)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to predictions JSON file (list of image entries).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--classes",
        default="public/classes.json",
        help="Path to classes.json for class id/name mapping.",
    )
    return parser.parse_args()


def load_classes(classes_path: Path) -> List[str]:
    with classes_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError("classes.json must be a list of strings.")

    return data


def normalize_number(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return float(value)
    return value


def class_to_name(value: Any, classes: List[str]) -> Optional[str]:
    if isinstance(value, int):
        return classes[value] if 0 <= value < len(classes) else None
    if isinstance(value, float):
        idx = int(value)
        if abs(value - idx) < 1e-6 and 0 <= idx < len(classes):
            return classes[idx]
        return None
    if isinstance(value, str):
        if value in classes:
            return value
        if value.isdigit():
            idx = int(value)
            return classes[idx] if 0 <= idx < len(classes) else None
        return None
    return None


def build_boxes(
    boxes: List[Dict[str, Any]],
    classes: List[str],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for box in boxes:
        bbox = box.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        class_name = class_to_name(box.get("class"), classes)
        if not class_name:
            continue

        output.append(
            {
                "x_min": normalize_number(bbox[0]),
                "y_min": normalize_number(bbox[1]),
                "x_max": normalize_number(bbox[2]),
                "y_max": normalize_number(bbox[3]),
                "class": class_name,
                "confidence": normalize_number(box.get("confidence")),
            }
        )

    return output


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    classes_path = Path(args.classes)

    classes = load_classes(classes_path)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "bounding_boxes"])

        for item in data:
            image_id = item.get("image_id")
            if not image_id:
                continue

            boxes = item.get("boxes", [])
            if not isinstance(boxes, list):
                boxes = []

            output_boxes = build_boxes(boxes, classes)
            boxes_json = json.dumps(output_boxes)
            writer.writerow([image_id, boxes_json])


if __name__ == "__main__":
    main()
