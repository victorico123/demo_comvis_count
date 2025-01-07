#WORKING
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2

app = Flask(__name__)

# Global variables for webcam state
camera = None
is_webcam_active = False

current_count = 0  # Current detected count
max_count = 0  # Max count during the session

ZONE_POLYGON = np.array([
    [0, 0],
    [640, 0],
    [640, 720],
    [0, 720]
])



def gen():
    """Video streaming generator function."""
    global camera, is_webcam_active, current_count, max_count  # Declare global variables

    if not is_webcam_active:
        return

    model = YOLO("yolov8l.pt")
    model.to('cuda')

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = ZONE_POLYGON
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=(1280, 720))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.blue(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    try:
        while is_webcam_active:
            ret, frame = camera.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            result = model(frame, agnostic_nms=True)[0]
            detections = sv.Detections.from_yolov8(result)
            detections = detections[detections.class_id == 0]

            current_count = len(detections)  # Update current detected count
            max_count = max(max_count, current_count)  # Update max count

            frame = box_annotator.annotate(
                scene=frame,
                detections=detections,
                skip_label=True
            )
            zone.trigger(detections=detections)
            frame = zone_annotator.annotate(scene=frame)

            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        if camera:
            camera.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    """Start the webcam stream."""
    global is_webcam_active, camera

    if not is_webcam_active:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            return jsonify({"status": "Failed to open webcam"}), 500
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        is_webcam_active = True
    return jsonify({"status": "Webcam started"})


@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    """Stop the webcam stream."""
    global is_webcam_active, camera

    if is_webcam_active:
        is_webcam_active = False
        if camera:
            camera.release()
            camera = None
    return jsonify({"status": "Webcam stopped"})


@app.route('/video_feed')
def video_feed():
    """Serve the video feed only if the webcam is active."""
    global is_webcam_active
    if not is_webcam_active:
        return jsonify({"error": "Webcam is not active"}), 400
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_current_count')
def get_current_count():
    """Return the current and max detected count."""
    global current_count, max_count
    return jsonify({"current_count": current_count, "max_count": max_count})

if __name__ == "__main__":
    app.run(debug=True)