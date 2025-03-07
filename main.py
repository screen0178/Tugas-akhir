import cv2
import glob
import os
import threading

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from ultralytics import YOLO
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def SuperRes(img_path):
    model_name = "RealESRGAN_x4plus"
    model_path = os.path.join("weights", model_name + ".pth")
    netscale = 4
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
    )

    f_input = img_path
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    denoise_strength = 0.5
    outscale = 4
    suffix = "_out"
    tile = 0
    tile_pad = 10
    pre_pad = 0
    fp32 = False
    alpha_upsampler = "realesrgan"
    ext = "auto"
    gpu_id = 0
    dni_weight = None

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=gpu_id,
    )

    if os.path.isfile(f_input):
        paths = [f_input]
    else:
        paths = sorted(glob.glob(os.path.join(f_input, "*")))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print("Testing", idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = "RGBA"
        else:
            img_mode = None

        try:
            output, _ = upsampler.enhance(img, outscale=outscale)
        except RuntimeError as error:
            print("Error", error)
            print(
                "If you encounter CUDA out of memory, try to set --tile with a smaller number."
            )
        else:
            if ext == "auto":
                extension = extension[1:]
            else:
                extension = ext
            if img_mode == "RGBA":  # RGBA images should be saved in png format
                extension = "png"
            if suffix == "":
                save_path = os.path.join(output_dir, f"{imgname}.{extension}")
            else:
                save_path = os.path.join(output_dir, f"{imgname}_{suffix}.{extension}")
            cv2.imwrite(save_path, output)


# Load YOLOv8 model
model = YOLO("yolov8n.pt")  

# Define the class IDs for car, bus, and truck
vehicle_classes = [2, 5, 7]

# Create a folder to save detected object images
output_folder = "inputs"
os.makedirs(output_folder, exist_ok=True)

# Load a video
# video_path = "tf.mp4"  # Replace with your video path
video_path = "https://pelindung.bandung.go.id:3443/video/HIKSVISION/Dagcik.m3u8"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Initialize object counter
object_counter = 0
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

                    # Spawn a new thread to process the saved image
                    threading.Thread(target=SuperRes, args=(object_img_path,)).start()

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