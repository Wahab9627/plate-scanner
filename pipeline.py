"""
Core detection pipeline, refactored to be callable from the web app.

Plate detection now goes through Roboflow's hosted inference API
(no local model file needed). OCR still runs locally via EasyOCR.
"""

import csv
import os
import subprocess
import time
from collections import defaultdict

import cv2
import easyocr
import imageio_ffmpeg
from inference_sdk import InferenceHTTPClient

_reader_cache = {}
_client_cache = {}

# Only send every Nth frame to the API — calling a hosted endpoint for
# every single frame of a video is slow and burns through your quota fast.
# Detections are held over the skipped frames so the output video still
# shows a box on every frame, it's just not re-detected each time.
FRAME_SKIP = 5


def get_reader():
    if "en" not in _reader_cache:
        _reader_cache["en"] = easyocr.Reader(["en"], gpu=False)
    return _reader_cache["en"]


def get_client(api_key):
    if api_key not in _client_cache:
        _client_cache[api_key] = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )
    return _client_cache[api_key]


def clean_plate_text(text: str) -> str:
    return "".join(c for c in text if c.isalnum()).upper()


def boxes_overlap(box1, box2, iou_thresh=0.4):
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


def transcode_for_browser(input_path, output_path):
    """
    OpenCV writes mp4 files using the mp4v codec, which Chrome/Firefox/Edge
    refuse to play in an HTML5 <video> tag (they expect H.264). This
    re-encodes to H.264 using a bundled ffmpeg binary so the result actually
    plays in the browser.
    """
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_path, "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-loglevel", "error",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def detect_plates_api(client, model_id, frame, conf_thresh):
    """Calls the Roboflow hosted API on one frame, returns list of (x1,y1,x2,y2,conf)."""
    result = client.infer(frame, model_id=model_id)
    boxes = []
    for pred in result.get("predictions", []):
        conf = pred.get("confidence", 0)
        if conf < conf_thresh:
            continue
        cx, cy = pred["x"], pred["y"]
        w, h = pred["width"], pred["height"]
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        boxes.append((x1, y1, x2, y2, conf))
    return boxes


def process_video(video_path, api_key, model_id, output_video_path, output_csv_path,
                   conf_thresh=0.4, ocr_conf_thresh=0.4, progress_cb=None):
    """
    Runs detection (Roboflow API) + OCR (EasyOCR, local) over a video,
    writes an annotated video and a CSV log.
    progress_cb(current_frame, total_frames) is called periodically if provided.
    Returns a summary dict with the detections list.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not api_key:
        raise ValueError("Missing Roboflow API key.")
    if not model_id:
        raise ValueError("Missing Roboflow model id.")

    client = get_client(api_key)
    reader = get_reader()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_video_path = output_video_path + ".raw.mp4"
    writer = cv2.VideoWriter(raw_video_path, fourcc, fps, (width, height))

    detections = []
    recent_boxes = []
    last_boxes = []  # boxes held over from the last API call, for skipped frames
    frame_num = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % FRAME_SKIP == 0:
            try:
                last_boxes = detect_plates_api(client, model_id, frame, conf_thresh)
            except Exception as e:
                # don't kill the whole job over one flaky API call — skip this frame
                last_boxes = []
                print(f"[warn] frame {frame_num}: API call failed: {e}")

        for (x1, y1, x2, y2, det_conf) in last_boxes:
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue

            ocr_results = reader.readtext(crop)
            if not ocr_results:
                continue

            best = max(ocr_results, key=lambda r: r[2])
            raw_text, ocr_conf = best[1], best[2]
            plate_text = clean_plate_text(raw_text)

            if not plate_text or ocr_conf < ocr_conf_thresh:
                continue

            is_dup = any(
                boxes_overlap((x1, y1, x2, y2), b[0]) and (frame_num - b[2]) < int(fps * 2)
                for b in recent_boxes
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, plate_text, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if not is_dup:
                timestamp = frame_num / fps
                detections.append({
                    "frame": frame_num,
                    "timestamp": round(timestamp, 2),
                    "plate_text": plate_text,
                    "confidence": round(ocr_conf, 2),
                })
                recent_boxes.append(((x1, y1, x2, y2), plate_text, frame_num))

        recent_boxes = [b for b in recent_boxes if (frame_num - b[2]) < int(fps * 5)]

        writer.write(frame)
        frame_num += 1

        if progress_cb and frame_num % 10 == 0:
            progress_cb(frame_num, total_frames)

    cap.release()
    writer.release()

    transcode_for_browser(raw_video_path, output_video_path)
    if os.path.exists(raw_video_path):
        os.remove(raw_video_path)

    with open(output_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_number", "timestamp_sec", "plate_text", "confidence"])
        for d in detections:
            w.writerow([d["frame"], d["timestamp"], d["plate_text"], d["confidence"]])

    # Aggregate readings of the same plate text across the whole video —
    # a single frame's OCR can misread a character, but the plate that
    # shows up most often (weighted by confidence) is the reliable answer.
    best_guess = None
    if detections:
        grouped = defaultdict(list)
        for d in detections:
            grouped[d["plate_text"]].append(d["confidence"])

        scored = [
            (text, len(confs), sum(confs) / len(confs))
            for text, confs in grouped.items()
        ]
        # rank by how many times it was seen first, then by average confidence
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        top_text, top_count, top_avg = scored[0]
        best_guess = {
            "plate_text": top_text,
            "occurrences": top_count,
            "avg_confidence": round(top_avg, 2),
        }

    elapsed = time.time() - start_time
    return {
        "detections": detections,
        "best_guess": best_guess,
        "frame_count": frame_num,
        "elapsed_sec": round(elapsed, 1),
    }