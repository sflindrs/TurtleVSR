import cv2
import os
import shutil
import math

def extract_frames(video_path, frames_dir, target_fps=None, clear_output_dir=True):
    """
    Convert a video file into individual frames saved to a specified directory.
    
    This function extracts frames from a video at the original frame rate or at a 
    reduced frame rate if target_fps is specified. It calculates the interval 
    between frames to extract based on the original video's FPS and the target FPS.
    
    Args:
        video_path (str): Path to the input video file
        frames_dir (str): Directory where extracted frames will be saved
        target_fps (int, optional): Target frame rate for extraction. If None, use original FPS.
            Default: None.
        clear_output_dir (bool, optional): Whether to clear the output directory 
                                          if it already exists. Default: True.
    
    Returns:
        int: Number of frames extracted
    """
    # Handle the output directory
    if clear_output_dir and os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0
    
    # Get the frame rate and total frames of the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps if original_fps > 0 else 0
    
    # If target_fps is not specified or is greater than original_fps, use original_fps
    if target_fps is None or target_fps >= original_fps or original_fps <= 0:
        target_fps = original_fps
        interval = 1
    else:
        # Calculate the interval between frames to extract
        interval = max(1, int(original_fps / target_fps))
    
    # Calculate the number of digits needed for frame numbering
    # Use log10(n) + 1 to get the number of digits in n
    # Add 1 to handle edge cases and ensure we have enough digits
    num_digits = max(4, int(math.log10(total_frames)) + 2) if total_frames > 0 else 4
    
    # Print video information
    print(f"Video Information:")
    print(f"  - Path: {video_path}")
    print(f"  - Original FPS: {original_fps:.2f}")
    print(f"  - Target FPS: {target_fps}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Duration: {duration:.2f} seconds")
    print(f"  - Extracting every {interval} frame(s)")
    print(f"  - Using {num_digits} digits for frame numbering")
    
    # Extract frames
    frame_count = 0
    saved_frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame if it's in the interval
        if frame_count % interval == 0:
            frame_filename = os.path.join(frames_dir, f'frame_{saved_frame_count+1:0{num_digits}d}.png')
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
            
        frame_count += 1
        
        # Optional: Show progress every 100 frames
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
    
    # Release resources
    cap.release()
    
    print(f'Extracted {saved_frame_count} frames to {frames_dir}')
    return saved_frame_count

# Example usage:
# extract_frames('video.mp4', 'output_dir')
# extract_frames('video.mp4', 'output_dir', target_fps=5)
# extract_frames('video.mp4', 'output_dir', clear_output_dir=False)
