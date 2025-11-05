import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import yaml
import os
from datetime import datetime

# Load YOLOv8 model
model = YOLO('yolov8m.pt')

# Load class names
with open('coco128.yaml') as f:
    data = yaml.safe_load(f)
CLASSES = data['names']

# Random colors
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Output folders
os.makedirs("detections/rgb", exist_ok=True)
os.makedirs("detections/depth", exist_ok=True)

# Initialize RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

# Save limit
SAVE_LIMIT = 5
saved_count = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.075), cv2.COLORMAP_JET)

        results = model(color_image, stream=True)

        class_ids = []
        confidences = []
        bboxes = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf
                if confidence > 0.3:
                    xyxy = box.xyxy.tolist()[0]
                    bboxes.append(xyxy)
                    confidences.append(float(confidence))
                    class_ids.append(box.cls.tolist())

        result_boxes = cv2.dnn.NMSBoxes(bboxes, confidences, 0.25, 0.45, 0.5)
        font = cv2.FONT_HERSHEY_PLAIN

        for i in range(len(bboxes)):
            if i in result_boxes:
                label = str(CLASSES[int(class_ids[i][0])])
                bbox = list(map(int, bboxes[i]))
                x1, y1, x2, y2 = bbox
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # ---- Distance in meters ----
                depth_m = depth_frame.get_distance(cx, cy)
                depth_str = f"{depth_m:.2f} m"

                label_text = f"{label} {confidences[i]:.2f} {depth_str}"

                color = tuple(map(int, colors[i]))
                tl = 2
                tf = max(tl - 1, 1)
                t_size = cv2.getTextSize(label_text, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3

                # Draw on RGB
                cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(color_image, (x1, y1), c2, color, -1, cv2.LINE_AA)
                cv2.circle(color_image, (cx, cy), 5, color, -1)
                cv2.putText(color_image, label_text, (x1, y1 - 2), 0, tl / 3,
                            (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)

                # Draw on depth
                cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (0, 0, 0), 2)
                cv2.rectangle(depth_colormap, (x1, y1), c2, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(depth_colormap, (cx, cy), 5, (0, 0, 0), -1)
                cv2.putText(depth_colormap, label_text, (x1, y1 - 2), 0, tl / 3,
                            (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)

                # Save detection image (limit 5)
                if saved_count < SAVE_LIMIT:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    rgb_path = f"detections/rgb/{timestamp}_{label}.jpg"
                    depth_path = f"detections/depth/{timestamp}_{label}.jpg"
                    cv2.imwrite(rgb_path, color_image)
                    cv2.imwrite(depth_path, depth_colormap)
                    print(f"[Saved] {rgb_path} | {depth_path}")
                    saved_count += 1

        # Display streams
        cv2.imshow("RGB Stream", color_image)
        cv2.imshow("Depth Stream", depth_colormap)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break

        # Stop saving after 5 detections
        if saved_count >= SAVE_LIMIT:
            print("✅ Saved 5 detection images. Continuing live view only.")
            SAVE_LIMIT = -1  # disable further saves

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
