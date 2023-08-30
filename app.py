from flask import Flask, render_template, request, jsonify, url_for, send_file
import cv2
import pandas as pd
import time
import torch
from werkzeug.utils import secure_filename
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from datetime import datetime

app = Flask(__name__)
app.config["STATIC_FOLDER"] = "templates/static"

weights_path = "D:\\NTU Project Submissions\\Sem4\\yolo_model_results\\YOLO5_5m_Aug_10K\\weights\\YOLO5_5m_Aug_10K_best.pt"
# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", weights_path)

UPLOAD_FOLDER = """upload_data"""
DATA_FILE_FOLDER = """templates\\static\\imgs"""


def remove_file(folder_path):
    files = os.listdir(folder_path)

    # Iterate through the files and remove each one
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                # print(f"Deleted: {file}")
            else:
                pass
                # print(f"Skipped: {file} (not a file)")
        except Exception as e:
            print(f"Error deleting {file}: {e}")


@app.route("/project_overview")
def project_overview():
    return render_template("about_project.html")


@app.route("/home", methods=["GET", "POST"])
def home():
    return render_template("index2.html", result="")


@app.route("/detect", methods=["GET", "POST"])
def detect():
    """Detects bees in a video and gets the timestamp of each detection.

    Args:
      video_path: The path to the video file.
      weights_path: The path to the YOLOv5 best.pt weight file.

    Returns:
      A list of tuples, where each tuple contains the timestamp and bounding box
      of a bee detection.
    """
    if request.method == "POST":
        print("Entered Post section")
        if "file" not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"})
        remove_file(DATA_FILE_FOLDER)
        remove_file(UPLOAD_FOLDER)

        start_time = time.time()
        # Save the file to the upload folder
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))

        uploaded_file_name = os.path.join(UPLOAD_FOLDER, filename)
        cap = cv2.VideoCapture(uploaded_file_name)
        frame_timestamp_set = set()
        frame_count = 1  # To keep track of frame numbers
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        video_duration = int(total_frames / fps)
        keyframe_interval = fps // 2  # Set the keyframe interval based on frame rate

        detection_start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            # keyframe_interval logic
            if frame_count % keyframe_interval == 0:
                new_height, new_width = 416, 416
                new_frame = cv2.resize(
                    frame, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC
                )

                detections = model(new_frame)
                # print(f"frame_count - {frame_count}")

                if len(detections.pred[0]) > 0:
                    for x1, y1, x2, y2, conf, cls in detections.pred[0]:
                        if conf > 0.3:
                            frame_timestamp_set.add(frame_count)

        cap.release()

        detect_time = round(time.time() - detection_start_time, 3)
        cleaned_frame_timestamp = {int(value / fps) for value in frame_timestamp_set}
        video_duration_set = set(np.arange(video_duration))

        # Creating a set of values that are both in A and B
        result_set = {
            value: 1 if value in cleaned_frame_timestamp else 0
            for value in video_duration_set
        }

        df = pd.DataFrame(
            result_set.items(), columns=["frame_detected_secs", "detected_flag"]
        )

        # df.to_csv("dump.csv", index=False)
        # Convert seconds to days, hours, minutes
        df["days"], seconds_remainder = divmod(
            df["frame_detected_secs"], 86400
        )  # 86400 seconds in a day
        df["hours"], seconds_remainder = divmod(
            seconds_remainder, 3600
        )  # 3600 seconds in an hour
        df["minutes"], seconds_remainder = divmod(
            seconds_remainder, 60
        )  # 60 seconds in a minute
        df["seconds"] = seconds_remainder

        df["frame_detected_secs"] = pd.to_timedelta(df["frame_detected_secs"], unit="s")
        df["formatted_time"] = df["frame_detected_secs"].apply(
            lambda x: f"{x.days} days {x.seconds // 3600} hours {x.seconds % 3600 // 60} mins {x.seconds % 60} secs"
        )
        df["dd_hh_mm_ss"] = df["frame_detected_secs"].apply(
            lambda x: f"{x.days:02d}:{x.seconds // 3600:02d}:{x.seconds % 3600 // 60:02d}:{x.seconds % 60:02d}"
        )

        csv_file_name = (
            f"templates\\static\\imgs\\{datetime.now().date()}_timestamps.csv"
        )
        img_file_name = f"templates\\static\\imgs\\plot.jpeg"
        df.to_csv(csv_file_name, index=False)
        # Create line plot
        # Determine the appropriate time unit for plotting
        # print("video_duration: ", video_duration)
        if any(df["days"] > 0):
            time_unit = "days"
        elif any(df["hours"] > 0):
            time_unit = "hours"
        elif any(df["minutes"] > 0):
            time_unit = "minutes"
        else:
            time_unit = "seconds"

        sns.lineplot(
            x=time_unit,
            y="detected_flag",
            data=df,
            color="b",
            markersize=8,
            linestyle="-",  # Default is 'solid'
            linewidth=2,
            ci=None,
        )
        plt.xlabel(f"Time ({time_unit})")
        plt.ylabel("Bee Sighting ( Yes or No )")
        plt.legend()
        plt.title("Bee Detections Throughout the Video")
        plt.savefig(img_file_name)

        api_time = round(time.time() - start_time, 3)

        result = {
            "success": True,
            "output": {
                "filename": filename,
                "csv_filename": csv_file_name,
                "img_file_name": img_file_name,
                "detection_duration": f"Time for detection {round(detect_time/60,2)} mins or {detect_time} secs",
                "api_time": f"API running time {round(api_time/60,2)} mins or {api_time} secs ",
                "video_duration": f"Duration of video uploaded {video_duration} secs or {round(video_duration/60,2)} mins",
            },
        }

        return render_template("index.html", result=result)

    else:
        result = {
            "success": False,
            "output": {
                "csv_filename": "NA",
                "img_file_name": "NA",
                "detection_duration": "NA",
                "api_time": f"NA",
                "video_duration": f"NA",
            },
        }
        return render_template("index.html", result=result)


@app.route("/download_csv", methods=["GET"])
def download_csv():
    csv_path = f"templates\\static\\imgs\\{datetime.now().date()}_timestamps.csv"  # Replace with the actual path to your CSV file
    return send_file(csv_path, as_attachment=True, mimetype="text/csv")


@app.route("/download_image", methods=["GET"])
def download_image():
    img_path = f"templates\\static\\imgs\\plot.jpeg"  # Replace with the actual path to your image file
    return send_file(img_path, as_attachment=True, mimetype="image/jpeg")


if __name__ == "__main__":
    try:
        app.run(debug=True)
    except Exception as e:
        print(e)
