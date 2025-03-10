from ultralytics import YOLO
import cv2
import os

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # You can use other versions like "yolov8s.pt", "yolov8m.pt", etc.

# Define the class IDs for car, bus, and truck
vehicle_classes = [2, 5, 7]

# Create a folder to save detected object images
output_folder = "inputs/detected_objects"
os.makedirs(output_folder, exist_ok=True)

# Load a video
video_path = "inputs/vlc-record-2025-02-07-10h03m40s-Media Presentation-.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Initialize object counter
object_counter = 28
tracked_objects = {}  # Dictionary to store tracked objects and their counts

# Initialize the tracker (ByteTrack is used here)
tracker = model.track(source=video_path, stream=True, persist=True, tracker="bytetrack.yaml")

# Process the video frame by frame
for result in tracker:
    frame = result.orig_img  # Get the current frame

    # Get tracking IDs and bounding boxes
    if result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in pixel coordinates
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
        track_ids = result.boxes.id.cpu().numpy().astype(int)  # Tracking IDs

        # Loop through detected objects
        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            if class_id in vehicle_classes:  # Filter for car, bus, and truck
                # Check if the object is already counted
                if track_id not in tracked_objects:
                    tracked_objects[track_id] = True  # Mark as counted
                    object_counter += 1  # Increment counter

                    # Crop and save the detected object
                    x1, y1, x2, y2 = map(int, box)
                    object_img = frame[y1:y2, x1:x2]  # Crop the object
                    object_img_path = os.path.join(output_folder, f"object_{object_counter}.jpg")
                    cv2.imwrite(object_img_path, object_img)

                # Draw bounding box and label
                x1, y1, x2, y2 = map(int, box)  # Ensure coordinates are integers
                label = f"{model.names[class_id]} {track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the total count on the frame
    cv2.putText(frame, f"Total Vehicles: {object_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the annotated frame
    cv2.imshow("Vehicle Tracking", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()