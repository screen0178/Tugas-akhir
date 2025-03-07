from ultralytics import YOLO
import os
import streamlit as st
import cv2


st.set_page_config(
    page_title="Streamlit APP"
)

st.title("Vehicle detection")


# Add text input for video path
video_path = st.text_input("Enter video path:", "tf.mp4")

cam1, cam2 = st.columns((3, 3))
video_1 = cam1.empty()
video_2 = cam2.empty()

# Add a submit button
start_button = st.button("Start Detection")

if start_button:
    model = YOLO("yolov8n.pt") 
    vehicle_classes = [2, 5, 7]

    output_folder = "inputs"
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    object_counter = 0
    tracked_objects = {}  # Dictionary to store tracked objects and their counts

    tracker = model.track(source=video_path, stream=True, persist=True, tracker="bytetrack.yaml")

    for result in tracker:
        frame = result.orig_img  # Get the current frame
        video_1.image(frame, channels="RGB")
    
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
        # cv2.imshow("Vehicle Tracking", frame)
        video_2.image(frame, channels="RGB")
    
    
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()