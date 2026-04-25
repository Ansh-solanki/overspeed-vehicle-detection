import cv2
import torch
import numpy as np
import csv
from sort.sort import Sort

# -------------------------------
# Load YOLOv5 model
# -------------------------------
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# -------------------------------
# Initialize tracker
# -------------------------------
tracker = Sort()

# -------------------------------
# Video input
# -------------------------------
cap = cv2.VideoCapture("short.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

# -------------------------------
# Output video
# -------------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    'final_output.mp4',
    fourcc,
    fps,
    (int(cap.get(3)), int(cap.get(4)))
)

# -------------------------------
# CSV setup (NEW)
# -------------------------------
csv_file = open('violations.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Vehicle_ID", "Speed_km_h", "Frame", "Timestamp_sec"])

# -------------------------------
# Config
# -------------------------------
SPEED_LIMIT = 60
SCALE = 0.05

# -------------------------------
# Memory structures
# -------------------------------
id_positions = {}
id_speeds = {}
violation_timer = {}

HOLD_FRAMES = 15

# -------------------------------
# Counters
# -------------------------------
violation_count = 0
counted_ids = set()
frame_count = 0

# -------------------------------
# Main loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = model(frame)

    detections = []

    # -------------------------------
    # Detection filtering
    # -------------------------------
    for *box, conf, cls in results.xyxy[0]:
        if int(cls) in [2, 5, 7]:  # car, bus, truck
            x1, y1, x2, y2 = map(int, box)
            detections.append([x1, y1, x2, y2, float(conf)])

    if len(detections) > 0:
        detections = np.array(detections)
    else:
        detections = np.empty((0, 5))

    # -------------------------------
    # Tracking
    # -------------------------------
    tracked = tracker.update(detections)

    for obj in tracked:
        x1, y1, x2, y2, obj_id = map(int, obj)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # -------------------------------
        # Speed calculation
        # -------------------------------
        if obj_id in id_positions:
            prev_cx, prev_cy = id_positions[obj_id]

            distance = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)

            if distance > 2:
                real_distance = distance * SCALE
                speed = real_distance * fps * 3.6

                if obj_id in id_speeds:
                    id_speeds[obj_id] = 0.7 * id_speeds[obj_id] + 0.3 * speed
                else:
                    id_speeds[obj_id] = speed

        id_positions[obj_id] = (cx, cy)

        # -------------------------------
        # Overspeed detection + CSV logging
        # -------------------------------
        if obj_id in id_speeds:
            speed_val = id_speeds[obj_id]

            if speed_val > SPEED_LIMIT:
                violation_timer[obj_id] = HOLD_FRAMES

                # Log only once
                if obj_id not in counted_ids:
                    violation_count += 1
                    counted_ids.add(obj_id)

                    timestamp = frame_count / fps

                    csv_writer.writerow([
                        obj_id,
                        int(speed_val),
                        frame_count,
                        round(timestamp, 2)
                    ])

                    # -------------------------------
                    # SAVE VEHICLE SNAPSHOT (NEW)
                    # -------------------------------
                    h, w, _ = frame.shape

                    # safe crop (avoid out of bounds)
                    x1_c = max(0, x1)
                    y1_c = max(0, y1)
                    x2_c = min(w, x2)
                    y2_c = min(h, y2)

                    vehicle_crop = frame[y1_c:y2_c, x1_c:x2_c]

                    if vehicle_crop.size > 0:
                        cv2.imwrite(f"output/images/vehicle_{obj_id}_{frame_count}.jpg", vehicle_crop)

        # -------------------------------
        # Draw violation
        # -------------------------------
        if obj_id in violation_timer and violation_timer[obj_id] > 0:

            violation_timer[obj_id] -= 1

            speed_val = int(id_speeds.get(obj_id, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            label = f"OVERSPEED | ID {obj_id} | {speed_val} km/h"

            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # -------------------------------
    # UI overlays
    # -------------------------------
    cv2.putText(frame, "Overspeed Vehicle Detection System",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

    cv2.putText(frame, f"Violations: {violation_count}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2)

    # -------------------------------
    # Output
    # -------------------------------
    out.write(frame)
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()