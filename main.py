import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np


ZONE_POLYGON = np.array([
    [0, 0],
    [0.75, 0],
    [0.75, 1],
    [0, 1]
])

# ZONE_POLYGON = np.array([
#     [0, 0],
#     [1, 0],
#     [1, 1],
#     [0, 1]
# ])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture("./source/crowd.mp4")
    # cap = cv2.VideoCapture("./test.jpg")
    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("./model/yolo11n.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        # text_thickness=2,
        text_scale=1
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.blue(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    FPS = 1/60
    FPS_MS = int(FPS * 1000)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        detections = detections[detections.class_id == 0]
        # labels = [
        #     f"{model.model.names[class_id]} {confidence:0.2f}"
        #     for _, confidence, class_id, _
        #     in detections
        # ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            skip_label=True
        )
    
        

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)      
        
        # 
        cv2.imshow("yolov8", frame)
        if cv2.waitKey(FPS_MS) & 0xFF == ord('q'):  # Press 'q' to quit
            break
        # if (cv2.waitKey(5) == 4):
        #     break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()