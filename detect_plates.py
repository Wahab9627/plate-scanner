"""
License Plate Detection + OCR from Video
-----------------------------------------
Pipeline: YOLOv8 (plate detection) -> crop -> EasyOCR (text recognition)

Setup (run once in your project folder):
    python -m venv venv
    venv\\Scripts\\activate        (Windows)
    source venv/bin/activate      (Mac/Linux)
    pip install ultralytics easyocr opencv-python

You also need a YOLO model trained/fine-tuned for license plates.
Two options:
  1) Use a general "vehicle" YOLO model first, then a plate-specific one.
  2) Easiest: download an open-source license-plate-trained YOLOv8 model
     (search "yolov8 license plate weights .pt" on GitHub/Roboflow Universe,
     e.g. Roboflow has several public license-plate datasets + trained weights).
     Save the .pt file into this project folder as `license_plate_yolov8.pt`.

Usage:
    python detect_plates.py --video input.mp4 --model license_plate_yolov8.pt --output out.mp4
"""

import argparse
import csv
import os
import time

import cv2
import easyocr
from ultralytics import YOLO


def clean_plate_text(text: str) -> str:
    """Keep only alphanumeric chars, uppercase it — typical plate format cleanup."""
    return "".join(c for c in text if c.isalnum()).upper()


def boxes_overlap(box1, box2, iou_thresh=0.4):
    """Simple IoU check so we don't log the same plate every single frame."""
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter_w, inter_h = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2b - x1b) * (y2b - y1b)
    union = area1 + area2 - inter_area
    if union == 0:
        return False
    return (inter_area / union) > iou_thresh


def run(video_path, model_path, output_path, csv_path, conf_thresh=0.4, ocr_conf_thresh=0.4):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}\n"
            "Download a license-plate-trained YOLOv8 .pt file and point --model at it."
        )

    print("Loading YOLO model...")
    model = YOLO(model_path)

    print("Loading EasyOCR (first run downloads model weights, be patient)...")
    reader = easyocr.Reader(["en"], gpu=False)  # set gpu=True if you have CUDA set up

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_number", "timestamp_sec", "plate_text", "confidence"])

    recent_boxes = []  # [(box, plate_text, last_seen_frame)]
    frame_num = 0
    start_time = time.time()

    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_thresh, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det_conf = float(box.conf[0])

            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue

            ocr_results = reader.readtext(crop)
            if not ocr_results:
                continue

            # take the highest-confidence text line in the crop
            best = max(ocr_results, key=lambda r: r[2])
            raw_text, ocr_conf = best[1], best[2]
            plate_text = clean_plate_text(raw_text)

            if not plate_text or ocr_conf < ocr_conf_thresh:
                continue

            # dedup: skip if we've already logged an overlapping box recently
            is_dup = any(
                boxes_overlap((x1, y1, x2, y2), b[0]) and (frame_num - b[2]) < int(fps * 2)
                for b in recent_boxes
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, plate_text, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if not is_dup:
                timestamp = frame_num / fps
                csv_writer.writerow([frame_num, f"{timestamp:.2f}", plate_text, f"{ocr_conf:.2f}"])
                recent_boxes.append(((x1, y1, x2, y2), plate_text, frame_num))
                print(f"[frame {frame_num}] {plate_text} (det={det_conf:.2f}, ocr={ocr_conf:.2f})")

        # trim old entries so this list doesn't grow forever
        recent_boxes = [b for b in recent_boxes if (frame_num - b[2]) < int(fps * 5)]

        writer.write(frame)
        frame_num += 1

    cap.release()
    writer.release()
    csv_file.close()

    elapsed = time.time() - start_time
    print(f"\nDone. Processed {frame_num} frames in {elapsed:.1f}s.")
    print(f"Annotated video: {output_path}")
    print(f"Detections log:  {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and read license plates from a video.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", default="license_plate_yolov8.pt", help="Path to YOLO .pt weights")
    parser.add_argument("--output", default="output.mp4", help="Path for annotated output video")
    parser.add_argument("--csv", default="detections.csv", help="Path for detections CSV log")
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO detection confidence threshold")
    parser.add_argument("--ocr-conf", type=float, default=0.4, help="OCR confidence threshold")
    args = parser.parse_args()

    run(args.video, args.model, args.output, args.csv, args.conf, args.ocr_conf)
