# Object Tracking and Depth Measurement with YOLOv8 and RealSense Camera

This project performs real-time object detection using the YOLOv8 model combined with depth sensing from an Intel RealSense camera. It captures synchronized RGB and depth frames, detects objects with YOLOv8, estimates their distances using depth data, and saves detection images with bounding boxes and distance information.

## Features
- Real-time object detection with YOLOv8 (using `yolov8m.pt` model).
- Depth measurement from RealSense depth camera aligned with RGB frames.
- Displays RGB and depth streams with annotated bounding boxes, labels, confidence scores, and distance.
- Saves up to 5 detection images separately for RGB and depth views.
- Uses Non-Maximum Suppression to filter overlapping detections.
- Random distinct colors assigned to detected object classes.

## Requirements
- Python 3.x
- pyrealsense2
- OpenCV (`cv2`)
- numpy
- ultralytics (YOLOv8)
- pyyaml

## Usage
1. Connect the Intel RealSense camera.
2. Place `yolov8m.pt` model and `coco128.yaml` in the working directory.
3. Run the script: python3 Object_recon_depth.py

## Output
- Images with detections and depth information saved under:
- `detections/rgb/`
- `detections/depth/`

## Notes
- Distance is calculated at the centroid of each detected bounding box.
- The script saves a maximum of 5 detection images during runtime.
- Adjust confidence and NMS thresholds as needed.

This setup enables robust object detection and distance estimation combining state-of-the-art YOLOv8 with depth data from RealSense for applications such as robotics, automation, and scene understanding
