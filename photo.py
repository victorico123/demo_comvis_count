# import cv2
# import argparse
# from ultralytics import YOLO
# import supervision as sv
# import numpy as np
# import argparse

# ZONE_POLYGON = np.array([
#     [0, 0],
#     [1, 0],
#     [1, 1],
#     [0, 1]
# ])

# def parse_arguments() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Crowd Photo image")
#     parser.add_argument(
#         "--image-path", 
#         default="./test.jpg",  # Set default path to photo.jpg
#         type=str
#     )
#     args = parser.parse_args()
#     return args


# def main():
#     cmd_parser = argparse.ArgumentParser(
#         description="Easeness of commandline execute for photo.py"
#     )

#     cmd_parser.add_argument(
#         "-m","--model",
#         type=str,
#         default="yolov10s.pt",
#         help="Choose your model (default:'yolov10s.pt')"
#     )

#     cmd_parser.add_argument(
#         "-a","--area",
#         type=int,
#         default=1,
#         help="Choose your area ([1] for whole screen(default), [2] for half screen)"
#     )

#     cmd_args = cmd_parser.parse_args()

#     args = parse_arguments()


#     # Load the image instead of using webcam
#     frame = cv2.imread(args.image_path)

#     if frame is None:
#         print(f"Error: Could not read image from {args.image_path}")
#         return

#     frame_width, frame_height = frame.shape[1], frame.shape[0]  # Get width and height of the image

#     model = YOLO(cmd_args.model)

#     box_annotator = sv.BoxAnnotator(
#         thickness=2,
#         # text_thickness=2,
#         text_scale=1
#     )

#     if cmd_args.area == 1 :
#         ZONE_POLYGON = np.array([
#             [0, 0],
#             [1, 0],
#             [1, 1],
#             [0, 1]
#         ])
#     elif cmd_args.area == 2:
#         ZONE_POLYGON = np.array([
#             [0, 0],
#             [0.5, 0],
#             [0.5, 1],
#             [0, 1]
#         ])
    

#     zone_polygon = (ZONE_POLYGON * np.array([frame_width, frame_height])).astype(int)
#     zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=(frame_width, frame_height))
#     zone_annotator = sv.PolygonZoneAnnotator(
#         zone=zone, 
#         color=sv.Color.blue(),
#         thickness=2,
#         text_thickness=4,
#         text_scale=2
#     )

#     # Run the detection model on the image
#     result = model(frame, agnostic_nms=True)[0]
#     detections = sv.Detections.from_yolov8(result)
#     detections = detections[detections.class_id == 0]  # Filter for class_id 0 (person)

#     # Annotate the image with bounding boxes
#     frame = box_annotator.annotate(
#         scene=frame, 
#         detections=detections, 
#         skip_label=True
#     )

#     # Annotate the frame with the polygon zone (optional)
#     zone.trigger(detections=detections)
#     frame = zone_annotator.annotate(scene=frame)

#     # Save the captured frame as an image
#     cv2.imwrite("output_image.jpg", frame)

#     # Display the image (optional)
#     cv2.imshow("Captured Image", frame)
#     cv2.waitKey(0)  # Wait for any key press to close the window

#     # Release resources
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()


import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

# Default polygon for the entire screen
ZONE_POLYGON = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Crowd Photo Detection Script")
    parser.add_argument(
        "--image-path",
        type=str,
        default="./source/test.jpg",  # Default image path
        help="Path to the input image"
    )
    parser.add_argument(
        "--model",
        type=int,
        default=1,
        choices=[1, 2],  # Area selection [1: whole screen, 2: half screen]
        help="Model used ([1] YoloV11n, [2] YoloV11n-face)"
    )
    parser.add_argument(
        "--area",
        type=int,
        default=1,
        choices=[1, 2],  # Area selection [1: whole screen, 2: half screen]
        help="Area to focus on ([1] whole screen, [2] half screen)"
    )
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()

    # Load the image
    frame = cv2.imread(args.image_path)
    if frame is None:
        print(f"Error: Could not read image from {args.image_path}")
        return

    frame_width, frame_height = frame.shape[1], frame.shape[0]

    # Adjust ZONE_POLYGON based on the area argument
    if args.area == 1:
        zone_polygon = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
    elif args.area == 2:
        zone_polygon = np.array([
            [0, 0],
            [0.75, 0],
            [0.75, 1],
            [0, 1]
        ])
    else:
        print("Error: Invalid area selected")
        return

    # Scale the polygon to match the image dimensions
    zone_polygon = (zone_polygon * np.array([frame_width, frame_height])).astype(int)

    # Load the YOLO model
    try:
        model_location=""
        if args.model == 1:
            model_location="./model/yolo11l.pt"           
        elif args.model == 2:
            model_location="./model/yolov11n-face.pt"
        model = YOLO(model_location)
    except Exception as e:
        print(f"Error: Could not load YOLO model from {args.model}. Exception: {e}")
        return

    # Define annotators
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_scale=1
    )
    zone = sv.PolygonZone(
        polygon=zone_polygon,
        frame_resolution_wh=(frame_width, frame_height)
    )
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.blue(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    # Run the YOLO model on the image
    result = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)
    detections = detections[detections.class_id == 0]  # Filter for "person" class (class_id=0)

    # Annotate the image with bounding boxes
    frame = box_annotator.annotate(
        scene=frame,
        detections=detections,
        skip_label=True
    )

    # Annotate the frame with the polygon zone
    zone.trigger(detections=detections)
    frame = zone_annotator.annotate(scene=frame)

    # Save the result
    output_path = "output_image.jpg"
    cv2.imwrite(output_path, frame)
    print(f"Output image saved to {output_path}")

    # Display the image
    cv2.imshow("Captured Image", frame)
    cv2.waitKey(0)  # Wait for any key press to close the window

    # Release resources
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

