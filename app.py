import os
import threading
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from pipeline import process_video

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Set these before running the app:
#   Windows (cmd):   set ROBOFLOW_API_KEY=your_key_here
#   Windows (ps):    $env:ROBOFLOW_API_KEY="your_key_here"
#   Mac/Linux:       export ROBOFLOW_API_KEY=your_key_here
# ROBOFLOW_MODEL_ID defaults to a public license-plate model on Roboflow
# Universe; swap it for your own trained model id if you have one.
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "license-plate-recognition-rxg4e/11")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB upload cap

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# In-memory job store. Fine for a single-process demo app;
# swap for Redis/DB if you deploy with multiple workers.
jobs = {}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def run_job(job_id, video_path):
    out_video = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
    out_csv = os.path.join(OUTPUT_DIR, f"{job_id}.csv")

    def progress_cb(current, total):
        jobs[job_id]["progress"] = {"current": current, "total": total}

    try:
        jobs[job_id]["status"] = "processing"
        result = process_video(
            video_path=video_path,
            api_key=ROBOFLOW_API_KEY,
            model_id=ROBOFLOW_MODEL_ID,
            output_video_path=out_video,
            output_csv_path=out_csv,
            progress_cb=progress_cb,
        )
        jobs[job_id]["status"] = "done"
        jobs[job_id]["detections"] = result["detections"]
        jobs[job_id]["best_guess"] = result["best_guess"]
        jobs[job_id]["frame_count"] = result["frame_count"]
        jobs[job_id]["elapsed_sec"] = result["elapsed_sec"]
        jobs[job_id]["video_url"] = f"/outputs/{job_id}.mp4"
        jobs[job_id]["csv_url"] = f"/outputs/{job_id}.csv"
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
    finally:
        # clean up the uploaded source file, keep only the outputs
        if os.path.exists(video_path):
            os.remove(video_path)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400
    if not ROBOFLOW_API_KEY:
        return jsonify({"error": "Server is missing the ROBOFLOW_API_KEY environment variable"}), 500

    job_id = uuid.uuid4().hex
    filename = secure_filename(file.filename)
    saved_path = os.path.join(UPLOAD_DIR, f"{job_id}_{filename}")
    file.save(saved_path)

    jobs[job_id] = {"status": "queued", "progress": {"current": 0, "total": 0}}
    thread = threading.Thread(target=run_job, args=(job_id, saved_path), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job id"}), 404
    return jsonify(job)


@app.route("/results/<job_id>")
def results(job_id):
    return render_template("results.html", job_id=job_id)


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)