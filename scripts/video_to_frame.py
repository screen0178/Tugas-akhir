import cv2
import os
import argparse

def video_to_frames(video_path, output_folder, frame_interval):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_count = 0
    saved_frame_count = 0
    
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()
        
        # If the frame was not grabbed, we've reached the end of the video
        if not ret:
            break
        
        # Save the frame only if it's a multiple of the frame interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")
            saved_frame_count += 1
        
        frame_count += 1
    
    # Release the video capture object
    cap.release()
    print(f"Finished extracting {saved_frame_count} frames (every {frame_interval} frames).")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract frames from a video and save them as images.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input video file.")
    parser.add_argument("-o", "--output", required=True, help="Path to the output folder to save frames.")
    parser.add_argument("-n", "--interval", type=int, default=75, help="Save every N-th frame (default: 75).")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the function to extract frames
    video_to_frames(args.input, args.output, args.interval)

if __name__ == "__main__":
    main()