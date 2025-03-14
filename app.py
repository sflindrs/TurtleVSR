import gradio as gr
import torch
import cv2
import os
import sys
import numpy as np
from pathlib import Path
import tempfile
import shutil
import time
import random
import string
from PIL import Image
import threading
import ctypes
import inspect

# Add the repository path to sys.path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import Turtle modules
from basicsr.utils.options import parse
from importlib import import_module
from basicsr.inference_no_ground_truth import main as inference_no_gt
from video_to_frames import extract_frames

# Global dictionary to keep track of running threads
active_processes = {}

# Constants
SUPPORTED_TASKS = {
    "Video Super-Resolution": {
        "model_path": "trained_models/SuperResolution.pth",
        "config_file": "options/Turtle_SR_MVSR.yml",
        "model_type": "SR"
    },
    "Video Deblurring": {
        "model_path": "trained_models/GoPro_Deblur.pth",
        "config_file": "options/Turtle_Deblur_Gopro.yml",
        "model_type": "t1"
    },
    "Video Deraining": {
        "model_path": "trained_models/NightRain.pth",
        "config_file": "options/Turtle_Derain.yml",
        "model_type": "t0"
    },
    "Rain Drop Removal": {
        "model_path": "trained_models/RainDrop.pth",
        "config_file": "options/Turtle_Derain_VRDS.yml",
        "model_type": "t1"
    },
    "Video Desnowing": {
        "model_path": "trained_models/Desnow.pth",
        "config_file": "options/Turtle_Desnow.yml",
        "model_type": "t0"
    },
    "Video Denoising": {
        "model_path": "trained_models/Denoising.pth",
        "config_file": "options/Turtle_Denoise_Davis.yml",
        "model_type": "t0"
    }
}

def _async_raise(tid, exctype):
    """Raises an exception in the threads with id tid"""
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("Invalid thread id")
    elif res != 1:
        # "if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def stop_thread(thread_id):
    """Terminates a thread by its ID"""
    if thread_id in active_processes:
        thread = active_processes[thread_id]
        _async_raise(thread.ident, KeyboardInterrupt)
        active_processes.pop(thread_id, None)
        return True
    return False

def generate_random_id(length=8):
    """Generate a random ID string"""
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for i in range(length))

def create_side_by_side_video(input_frames_dir, output_frames_dir, output_video_path, fps=20):
    """
    Create a side-by-side comparison video
    """
    # Get all input and output frames
    input_frames = sorted([f for f in os.listdir(input_frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    output_frames = sorted([f for f in os.listdir(output_frames_dir) if 'Pred' in f and f.endswith('.png')])
    
    if not input_frames or not output_frames:
        return "Error: No frames found"
    
    # Map output frames to their frame numbers
    output_frame_map = {}
    for output_frame in output_frames:
        # Extract frame number from names like "Frame_1_Pred.png"
        if '_' in output_frame:
            parts = output_frame.split('_')
            if len(parts) >= 2 and parts[0] == "Frame":
                try:
                    frame_num = int(parts[1])
                    output_frame_map[frame_num] = output_frame
                except ValueError:
                    continue
    
    # Get dimensions from first input frame
    input_frame_path = os.path.join(input_frames_dir, input_frames[0])
    frame = cv2.imread(input_frame_path)
    h, w, _ = frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w*2, h))
    
    # Create side-by-side comparisons
    for i, input_frame in enumerate(input_frames):
        frame_num = i + 1
        input_img = cv2.imread(os.path.join(input_frames_dir, input_frame))
        
        # Find matching output frame
        if frame_num in output_frame_map:
            output_frame = output_frame_map[frame_num]
            output_img = cv2.imread(os.path.join(output_frames_dir, output_frame))
        else:
            # If no matching output frame, use a blank frame
            output_img = np.zeros_like(input_img)
        
        # Make sure the frames are the same size
        if output_img.shape[:2] != input_img.shape[:2]:
            output_img = cv2.resize(output_img, (input_img.shape[1], input_img.shape[0]))
        
        combined = np.hstack((input_img, output_img))
        video_writer.write(combined)
    
    video_writer.release()
    return output_video_path

def create_comparison_with_slider(input_frames_dir, output_frames_dir, output_video_path, fps=20):
    """
    Create a video with a sliding comparison between input and output frames
    
    Args:
        input_frames_dir (str): Directory containing original input frames
        output_frames_dir (str): Directory containing processed output frames
        output_video_path (str): Path to save the output video
        fps (int): Frames per second for the output video
        
    Returns:
        str: Path to created video or error message
    """
    # Get all input frames (from the original directory)
    input_frames = sorted([
        f for f in os.listdir(input_frames_dir) 
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    # Get all output frames - should have consistent naming like "Frame_00001_Pred.png"
    output_frames = sorted([
        f for f in os.listdir(output_frames_dir) 
        if 'Pred' in f and f.endswith('.png')
    ])
    
    if not input_frames or not output_frames:
        return "Error: No frames found"
    
    print(f"Input frames: {len(input_frames)}")
    print(f"Output frames: {len(output_frames)}")
    
    # Get frame dimensions from first input frame
    input_frame_path = os.path.join(input_frames_dir, input_frames[0])
    frame = cv2.imread(input_frame_path)
    if frame is None:
        return f"Error: Could not read first input frame at {input_frame_path}"
        
    h, w, _ = frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    
    # Create frame mapping from output frame names to frame numbers
    frame_map = {}
    
    # Check if we're using new padding format (Frame_00001_Pred.png) or old format
    using_new_format = False
    if output_frames and '_' in output_frames[0]:
        parts = output_frames[0].split('_')
        if len(parts) >= 2 and parts[0] == "Frame" and parts[1].isdigit():
            using_new_format = True
    
    for output_frame in output_frames:
        # Extract frame number from names
        frame_num = None
        if using_new_format and '_' in output_frame:
            parts = output_frame.split('_')
            if len(parts) >= 2 and parts[0] == "Frame":
                try:
                    frame_num = int(parts[1])
                    frame_map[frame_num] = output_frame
                except ValueError:
                    continue
        else:
            # Fallback: try to extract frame number using regex
            import re
            match = re.search(r'(\d+)', output_frame)
            if match:
                try:
                    frame_num = int(match.group(1))
                    frame_map[frame_num] = output_frame
                except ValueError:
                    continue
    
    if not frame_map:
        return "Error: Could not extract frame numbers from output frames"
        
    print(f"Extracted {len(frame_map)} frame mappings from output frames")
    
    # Use the number of input frames as the base for creating the video
    frames_processed = 0
    frame_errors = 0
    
    # Use input frames as base sequence and find corresponding output frames
    for i, input_frame in enumerate(input_frames):
        frame_num = i + 1  # 1-indexed frame numbers
        
        # Calculate slider position based on frame index
        slider_position = int((i / len(input_frames)) * w)
        
        # Read input frame
        input_img = cv2.imread(os.path.join(input_frames_dir, input_frame))
        if input_img is None:
            print(f"Warning: Could not read input frame {input_frame}")
            frame_errors += 1
            continue
        
        # Find corresponding output frame
        if frame_num in frame_map:
            output_frame = frame_map[frame_num]
            output_path = os.path.join(output_frames_dir, output_frame)
            output_img = cv2.imread(output_path)
            
            # Make sure the output image exists and has the right size
            if output_img is None or output_img.shape[:2] != input_img.shape[:2]:
                print(f"Warning: Output frame {output_frame} is missing or has incorrect size")
                print(f"Using input frame as fallback")
                output_img = input_img.copy()  # Fallback to input if there's an issue
                frame_errors += 1
        else:
            # If we don't have a matching output frame, use the input frame
            print(f"Warning: No matching output frame for input frame {frame_num}")
            output_img = input_img.copy()
            frame_errors += 1
        
        # Create combined frame with sliding comparison
        combined_frame = input_img.copy()
        combined_frame[:, :slider_position] = output_img[:, :slider_position]
        
        # Draw the slider line
        cv2.line(combined_frame, (slider_position, 0), (slider_position, h), (0, 255, 0), 2)
        
        video_writer.write(combined_frame)
        frames_processed += 1
    
    video_writer.release()
    
    print(f"Video creation complete: {frames_processed} frames processed, {frame_errors} frame errors")
    
    if frame_errors > 0:
        return f"Video created with {frame_errors} frame errors. Check logs for details."
    
    return output_video_path

def create_regular_output_video(output_frames_dir, output_video_path, fps=20):
    """
    Create a video from processed output frames
    
    Args:
        output_frames_dir (str): Directory containing processed output frames
        output_video_path (str): Path to save the output video
        fps (int): Frames per second for the output video
        
    Returns:
        str: Path to created video or error message
    """
    # Get all output frames - look for files with "Pred" in the name
    output_frames = sorted([
        f for f in os.listdir(output_frames_dir) 
        if f.endswith('.png') and 'Pred' in f
    ])
    
    if not output_frames:
        return "Error: No output frames found"
    
    print(f"Found {len(output_frames)} output frames")
    
    # Extract frame numbers to ensure proper ordering
    frame_numbers = []
    for frame_name in output_frames:
        # Extract frame number using regex
        import re
        match = re.search(r'Frame_(\d+)_Pred', frame_name)
        if match:
            try:
                frame_num = int(match.group(1))
                frame_numbers.append((frame_num, frame_name))
            except ValueError:
                continue
    
    if not frame_numbers:
        return "Error: Could not extract frame numbers from output frames"
    
    # Sort by frame number
    frame_numbers.sort(key=lambda x: x[0])
    ordered_frames = [item[1] for item in frame_numbers]
    
    # Verify that we have a continuous sequence with no gaps
    if len(ordered_frames) > 1:
        first_num = frame_numbers[0][0]
        last_num = frame_numbers[-1][0]
        expected_count = last_num - first_num + 1
        
        if len(ordered_frames) != expected_count:
            print(f"Warning: Expected {expected_count} frames but found {len(ordered_frames)}")
            print(f"First frame: {first_num}, Last frame: {last_num}")
            
            # Check for gaps in the sequence
            actual_nums = set(item[0] for item in frame_numbers)
            expected_nums = set(range(first_num, last_num + 1))
            missing_nums = expected_nums - actual_nums
            
            if missing_nums:
                print(f"Missing frame numbers: {sorted(missing_nums)}")
    
    # Read the first frame to get dimensions
    sample_frame = cv2.imread(os.path.join(output_frames_dir, ordered_frames[0]))
    if sample_frame is None:
        return f"Error: Could not read first output frame at {os.path.join(output_frames_dir, ordered_frames[0])}"
        
    h, w, _ = sample_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        output_video_path, 
        fourcc, 
        fps, 
        (w, h)
    )
    
    # Process frames in their correct numerical order
    frames_processed = 0
    frame_errors = 0
    
    for frame_name in ordered_frames:
        frame_path = os.path.join(output_frames_dir, frame_name)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Warning: Could not read output frame {frame_name}")
            frame_errors += 1
            continue
            
        video_writer.write(frame)
        frames_processed += 1
                
    video_writer.release()
    
    print(f"Video creation complete: {frames_processed} frames processed, {frame_errors} frame errors")
    
    if frame_errors > 0:
        return f"Video created with {frame_errors} frame errors. Check logs for details."
    
    return output_video_path

def validate_video(video_path):
    """Validate video before processing"""
    if video_path is None:
        return False, "Please upload a video file."
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Unable to open video file. Please check the format."
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Estimate processing time (rough estimate based on resolution)
        base_fps = 2.0  # Base processing speed (frames per second)
        resolution_factor = (width * height) / (1280 * 720)  # Relative to 720p
        est_fps = base_fps / max(0.5, resolution_factor)  # Adjust based on resolution
        est_time = total_frames / est_fps
        est_minutes = int(est_time // 60)
        est_seconds = int(est_time % 60)

        aspect_ratio = width / height
        orientation = "Vertical" if aspect_ratio < 1 else "Horizontal"
        
        if width * height > 3840 * 2160:
            return False, f"{orientation} video resolution ({width}x{height}) is very high. Consider resizing for faster processing."
        
        return True, f"{orientation} video validated. Resolution: {width}x{height}, Frames: {total_frames}, FPS: {fps:.1f}. Estimated processing time: {est_minutes}m {est_seconds}s"
    except Exception as e:
        return False, f"Error validating video: {str(e)}"

def process_video_thread(thread_id, video_path, task_name, tile_size, tile_overlap, sample_rate, noise_level, 
                         denoising_strength, advanced_settings, output_format, use_custom_model, 
                         custom_model_path, custom_config_path, progress=None, 
                         # Additional parameters
                         output_fps=None, frame_limit=None, device_id=None, model_dim=None,
                         ffn_expansion_factor=None, batch_size=None):
    """Threaded function to process videos"""
    try:
        # Create a dummy progress callback if none was provided
        if progress is None:
            progress = lambda value, desc: print(f"Progress: {value*100:.1f}% - {desc}")
            
        result = process_video(video_path, task_name, tile_size, tile_overlap, sample_rate, noise_level, 
                               denoising_strength, advanced_settings, output_format, use_custom_model,
                               custom_model_path, custom_config_path, progress,
                               # Additional parameters
                               output_fps, frame_limit, device_id, model_dim, 
                               ffn_expansion_factor, batch_size)
        # Remove from active processes when done
        active_processes.pop(thread_id, None)
        return result
    except KeyboardInterrupt:
        print(f"Processing was canceled for thread {thread_id}")
        # Clean up any temporary files here
        return (None, None, "Processing was canceled")
    except Exception as e:
        import traceback
        print(f"Error in thread {thread_id}: {str(e)}")
        print(traceback.format_exc())
        active_processes.pop(thread_id, None)
        return (None, None, f"Error during processing: {str(e)}")

def process_video(video_path, task_name, tile_size, tile_overlap, sample_rate, noise_level, 
                 denoising_strength, advanced_settings, output_format, use_custom_model=False, 
                 custom_model_path="", custom_config_path="", progress=None,
                 # Additional parameters
                 output_fps=None, frame_limit=None, device_id=None, model_dim=None,
                 ffn_expansion_factor=None, batch_size=None):
    """Process the uploaded video and return the result"""
    
    # Create a default progress callback if none was provided
    if progress is None:
        progress = lambda value, desc: print(f"Progress: {value*100:.1f}% - {desc}")
    
    if video_path is None:
        return None, None, "Please upload a video to process."
    
    # Validate video first
    valid, message = validate_video(video_path)
    if not valid:
        return None, None, message
    
    if task_name not in SUPPORTED_TASKS:
        return None, None, f"Unknown task: {task_name}. Supported tasks are: {', '.join(SUPPORTED_TASKS.keys())}"
    
    # Create temporary working directories
    job_id = generate_random_id()
    temp_dir = os.path.join(tempfile.gettempdir(), f"turtle_{job_id}")
    frames_dir = os.path.join(temp_dir, "frames")
    output_dir = os.path.join(temp_dir, "output")
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Get task-specific parameters
        task_params = SUPPORTED_TASKS[task_name]
        
        # Check for custom model if specified
        if use_custom_model and custom_model_path and custom_config_path:
            model_path = custom_model_path
            config_file = custom_config_path
        else:
            model_path = task_params['model_path']
            config_file = task_params['config_file']
            
        model_type = task_params['model_type']
        
        # Calculate target fps based on sample rate
        # 1.0 = original fps, 0.5 = half fps, etc.
        target_fps = None if sample_rate >= 1.0 else None  # None means use original fps
        
        # Apply frame limit if specified
        frames_to_process = int(frame_limit) if frame_limit is not None and frame_limit > 0 else None
        
        # Extract frames from video
        progress(0.05, desc="Preparing to extract frames from video")
        num_frames = extract_frames(video_path, frames_dir, target_fps=target_fps, clear_output_dir=True)
        
        # If frame limit is specified, limit the number of frames to process
        if frames_to_process is not None and frames_to_process < num_frames:
            # Keep only the first 'frames_to_process' frames
            all_frames = sorted(os.listdir(frames_dir))
            for frame_file in all_frames[frames_to_process:]:
                os.remove(os.path.join(frames_dir, frame_file))
            num_frames = frames_to_process
            print(f"Limited to processing {num_frames} frames.")
        
        if num_frames == 0:
            return None, None, "Failed to extract frames from video."
        
        progress(0.3, desc=f"Extracted {num_frames} frames. Starting model inference.")
        
        # Adjust noise level for denoising task
        if task_name == "Video Denoising" and noise_level is not None:
            noise_sigma = float(noise_level) / 255.0
            print(f"Using noise level: {noise_level} ({noise_sigma})")
        else:
            noise_sigma = 50.0 / 255.0  # Default
        
        # Handle device
        device_str = f"cuda:{device_id}" if device_id is not None else "cuda"
        
        # Run inference
        inference_no_gt(
            model_path=model_path,
            model_name=f"{task_name}_{job_id}",
            data_dir=frames_dir,
            config_file=config_file,
            tile=tile_size,
            tile_overlap=tile_overlap,
            save_image=True,
            model_type=model_type,
            do_pacthes=True,
            image_out_path=output_dir,
            noise_sigma=noise_sigma,
            progress_callback=lambda value, text: progress(0.3 + 0.5 * value, desc=text)
        )
        
        # Create result video
        progress(0.8, desc="Creating result videos")
        
        # Find the model output directory - it's structured as output_dir/task_name_job_id/frames_dir_basename
        model_dir = os.path.join(output_dir, f"{task_name}_{job_id}")
        frames_dir_basename = os.path.basename(frames_dir)
        result_dir = os.path.join(model_dir, frames_dir_basename)
        
        # If the result_dir doesn't exist, use model_dir as a fallback
        if not os.path.exists(result_dir):
            result_dir = model_dir
            
        # Verify that result directory contains processed frames
        output_frames = [f for f in os.listdir(result_dir) if 'Pred' in f and f.endswith('.png')]
        if not output_frames:
            return None, None, f"No output frames found in {result_dir}. Processing may have failed."
        
        # Get frame rate from the original video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default to 30 fps if we can't get the frame rate
        cap.release()
        
        # Use specified output FPS if provided
        if output_fps is not None and output_fps > 0:
            fps = output_fps
        
        # Set the video codec based on output format
        video_codec = 'avc1' if output_format == 'MP4' else 'vp9'
        video_ext = '.mp4' if output_format == 'MP4' else '.webm'
        
        # Create the comparison video with slider
        progress(0.85, desc="Creating comparison video")
        comparison_video_path = os.path.join(temp_dir, f"comparison_{job_id}{video_ext}")
        comparison_result = create_comparison_with_slider(
            frames_dir,
            result_dir,
            comparison_video_path,
            fps=fps
        )
        
        if isinstance(comparison_result, str) and comparison_result.startswith("Error"):
            print(f"Warning creating comparison video: {comparison_result}")
        
        # Create regular output video from just the processed frames
        progress(0.95, desc="Creating output video")
        output_video_path = os.path.join(temp_dir, f"output_{job_id}{video_ext}")
        output_result = create_regular_output_video(
            result_dir,
            output_video_path,
            fps=fps
        )
        
        if isinstance(output_result, str) and output_result.startswith("Error"):
            print(f"Warning creating output video: {output_result}")
        
        # Resource usage info
        gpu_info = ""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_info = f" | GPU Memory: {memory_used:.2f}GB / {memory_total:.2f}GB"
        
        progress(1.0, desc=f"Processing complete{gpu_info}")
        
        # Check that both videos exist and are valid
        output_video_exists = os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0
        comparison_video_exists = os.path.exists(comparison_video_path) and os.path.getsize(comparison_video_path) > 0
        
        if not output_video_exists and not comparison_video_exists:
            return None, None, "Failed to create output videos. Check logs for details."
        
        # Return results, with appropriate messages if one video failed
        if not output_video_exists:
            return None, comparison_video_path, "Comparison video created, but output video failed."
        
        if not comparison_video_exists:
            return output_video_path, None, "Output video created, but comparison video failed."
        
        return output_video_path, comparison_video_path, f"Processing completed successfully. Processed {num_frames} frames.{gpu_info}"
        
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"Error during processing: {str(e)}")
        print(trace)
        return None, None, f"Error during processing: {str(e)}\n{trace}"
    
    finally:
        # Don't clean up immediately so we can debug if needed
        # We can add a cleanup timer to remove these files after some time
        pass

def validate_image(image_path):
    """Validate image before processing"""
    if image_path is None:
        return False, "Please upload an image file."
    
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        # Estimate processing time based on resolution
        resolution_factor = (width * height) / (1280 * 720)  # Relative to 720p
        est_time = max(2, resolution_factor * 2)  # Base time of 2 seconds
        
        if width * height > 3840 * 2160:
            return False, f"Image resolution ({width}x{height}) is very high. Consider resizing for faster processing."
        
        aspect_ratio = width / height
        orientation = "Vertical" if aspect_ratio < 1 else "Horizontal"
        
        return True, f"{orientation} image validated. Resolution: {width}x{height}. Estimated processing time: {est_time:.1f} seconds"
    except Exception as e:
        return False, f"Error validating image: {str(e)}"

def image_process(image_path, task_name, tile_size, tile_overlap, noise_level, denoising_strength, 
                 advanced_settings, use_custom_model=False, custom_model_path="", 
                 custom_config_path="", progress=None,
                 # Additional parameters
                 device_id=None, model_dim=None, ffn_expansion_factor=None):
    """Process a single image and return the result"""
    
    # Create a default progress callback if none was provided
    if progress is None:
        progress = lambda value, desc: print(f"Progress: {value*100:.1f}% - {desc}")
    
    # Validate image first
    valid, message = validate_image(image_path)
    if not valid:
        return None, message
    
    if task_name not in SUPPORTED_TASKS:
        return None, f"Unknown task: {task_name}. Supported tasks are: {', '.join(SUPPORTED_TASKS.keys())}"
    
    # Create temporary working directories
    job_id = generate_random_id()
    temp_dir = os.path.join(tempfile.gettempdir(), f"turtle_img_{job_id}")
    frames_dir = os.path.join(temp_dir, "frames")
    output_dir = os.path.join(temp_dir, "output")
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Get task-specific parameters
        task_params = SUPPORTED_TASKS[task_name]
        
        # Check for custom model if specified
        if use_custom_model and custom_model_path and custom_config_path:
            model_path = custom_model_path
            config_file = custom_config_path
        else:
            model_path = task_params['model_path']
            config_file = task_params['config_file']
            
        model_type = task_params['model_type']
        
        progress(0.1, desc="Preparing images for processing")
        
        # Save the image to the frames directory
        img = Image.open(image_path)
        img_save_path = os.path.join(frames_dir, "frame_0001.png")
        img.save(img_save_path)
        
        # Create a duplicate of the image for sequence processing
        # Since the model expects at least 2 frames
        img_save_path2 = os.path.join(frames_dir, "frame_0002.png")
        img.save(img_save_path2)
        
        progress(0.3, desc="Starting model inference")
        
        # Adjust noise level for denoising task
        if task_name == "Video Denoising" and noise_level is not None:
            noise_sigma = float(noise_level) / 255.0
            print(f"Using noise level: {noise_level} ({noise_sigma})")
        else:
            noise_sigma = 50.0 / 255.0  # Default
        
        # Handle device
        device_str = f"cuda:{device_id}" if device_id is not None else "cuda"
        
        # Run inference
        inference_no_gt(
            model_path=model_path,
            model_name=f"{task_name}_{job_id}",
            data_dir=frames_dir,
            config_file=config_file,
            tile=tile_size,
            tile_overlap=tile_overlap,
            save_image=True,
            model_type=model_type,
            do_pacthes=True,
            image_out_path=output_dir,
            noise_sigma=noise_sigma,
            progress_callback=lambda value, text: progress(0.3 + 0.6 * value, desc=text)
        )
        
        progress(0.9, desc="Processing complete, retrieving output image")
        
        # Find the model output directory
        model_dir = os.path.join(output_dir, f"{task_name}_{job_id}")
        frames_dir_basename = os.path.basename(frames_dir)
        result_dir = os.path.join(model_dir, frames_dir_basename)
        
        # If the result_dir doesn't exist, use model_dir as a fallback
        if not os.path.exists(result_dir):
            result_dir = model_dir
            
        # Find the first output image that contains 'Pred' in its name
        output_files = [f for f in os.listdir(result_dir) if 'Pred' in f and f.endswith('.png')]
        if not output_files:
            return None, f"No output images found in {result_dir}. Processing may have failed."
        
        # Output the first processed frame
        output_image_path = os.path.join(result_dir, output_files[0])
        
        progress(1.0, desc="Processing complete")
        
        return output_image_path, "Image processing completed successfully."
        
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"Error during processing: {str(e)}")
        print(trace)
        return None, f"Error during processing: {str(e)}"
    
    finally:
        # We'll leave cleanup for later to enable debugging
        pass

def cancel_processing(job_id):
    """Cancel active processing job"""
    if job_id in active_processes:
        print(f"Attempting to cancel job {job_id}")
        result = stop_thread(job_id)
        if result:
            return f"Processing job {job_id} has been canceled."
        else:
            return f"Failed to cancel job {job_id}"
    else:
        return f"No active job found with ID {job_id}"

def create_ui():
    """Create the Gradio UI"""
    # Set page title and theme
    title = "Turtle 🐢: Unified Video Restoration"
    
    # Updated CSS to support dark theme and improve interactions
    css = """
    :root {
        --background-primary: #1f2937;
        --background-secondary: #111827;
        --text-primary: #f9fafb;
        --border-color: #374151;
        --input-background: #374151;
        --input-text: #f9fafb;
    }

    .dark, [data-theme='dark'] {
        background-color: var(--background-primary) !important;
        color: var(--text-primary) !important;
    }

    .gradio-container {
        background-color: var(--background-primary) !important;
    }

    #title { 
        text-align: center; 
        font-size: 2.5rem !important; 
        font-weight: bold;
        margin-bottom: 0.5rem !important;
        color: var(--text-primary) !important;
    }

    .video-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        max-height: 70vh;
        overflow: auto;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 10px;
        background-color: var(--background-secondary) !important;
    }

    .top-controls {
        display: flex;
        gap: 10px;
        margin-bottom: 15px;
        padding: 10px;
        background-color: var(--background-secondary) !important;
        border-radius: 5px;
        position: sticky;
        top: 0;
        z-index: 100;
    }

    /* Ensure inputs and buttons have proper styling */
    .gradio-container input, 
    .gradio-container select, 
    .gradio-container textarea, 
    .gradio-container button {
        background-color: var(--input-background) !important;
        color: var(--input-text) !important;
        border-color: var(--border-color) !important;
    }

    /* Improve interaction states */
    .gradio-container input:focus, 
    .gradio-container select:focus, 
    .gradio-container textarea:focus {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2) !important;
    }

    /* Tabs styling */
    .tabitem {
        background-color: var(--background-primary) !important;
        color: var(--text-primary) !important;
    }

    /* Accordion styling */
    .accordionitem {
        background-color: var(--background-secondary) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-color) !important;
    }

    /* Ensure sliders and dropdowns are visible */
    .gradio-container .slider, 
    .gradio-container .dropdown {
        background-color: var(--input-background) !important;
        color: var(--input-text) !important;
    }

    /* Fix for video and image components */
    .gradio-video, .gradio-image {
        background-color: var(--background-secondary) !important;
        border-color: var(--border-color) !important;
    }
    """
    
    # Import Gradio theme
    dark_theme = gr.Theme(
        css=css,
        # Optional: You can customize colors further here
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.gray
    )
    
    with gr.Blocks(css=css, theme=dark_theme, title=title) as app:
        current_job_id = gr.State(value=None)
        
        gr.Markdown("# Turtle 🐢", elem_id="title")
        gr.Markdown("## Unified Video Restoration")
        
        with gr.Tabs():
            with gr.Tab("Video Processing"):
                with gr.Row(elem_classes="top-controls"):
                    # Move the process and cancel buttons to the top
                    with gr.Column(scale=1):
                        task = gr.Dropdown(
                            choices=list(SUPPORTED_TASKS.keys()),
                            value="Video Super-Resolution",
                            label="Restoration Task"
                        )
                    with gr.Column(scale=1):
                        process_button = gr.Button("Process Video", variant="primary")
                        
                    with gr.Column(scale=1):
                        cancel_button = gr.Button("Cancel Processing", variant="stop")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Input section
                        input_video = gr.Video(label="Input Video")
                        
                        # Control panel section
                        with gr.Accordion("Basic Settings", open=True):
                            with gr.Row():
                                tile_size = gr.Slider(
                                    minimum=32, maximum=512, value=256, step=32,
                                    label="Tile Size",
                                    info="Larger values use more memory but may provide better quality"
                                )
                                tile_overlap = gr.Slider(
                                    minimum=0, maximum=128, value=16, step=4,
                                    label="Tile Overlap",
                                    info="Higher overlap can reduce tiling artifacts"
                                )
                            with gr.Row():
                                sample_rate = gr.Slider(
                                    minimum=0.1, maximum=1.0, value=1.0, step=0.1,
                                    label="Frame Sample Rate",
                                    info="Lower values process fewer frames (faster but may reduce temporal consistency)"
                                )
                                output_format = gr.Dropdown(
                                    choices=["MP4", "WebM"],
                                    value="MP4",
                                    label="Output Format"
                                )
                            with gr.Row():
                                output_fps = gr.Number(
                                    value=0, 
                                    label="Output FPS (0 = same as input)",
                                    info="Set the output video frame rate (0 to use original)"
                                )
                                frame_limit = gr.Number(
                                    value=0,
                                    label="Frames to Process (0 = all)",
                                    info="Limit the number of frames to process (0 for all frames)"
                                )
                        
                        # Advanced Settings
                        with gr.Accordion("Advanced Settings", open=False):
                            with gr.Row():
                                noise_level = gr.Slider(
                                    minimum=0, maximum=255, value=50, step=5,
                                    label="Noise Level",
                                    info="Only applies to denoising tasks (higher = more noise removal)"
                                )
                                denoising_strength = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                                    label="Denoising Strength",
                                    info="Strength of denoising effect"
                                )
                            
                            with gr.Row():
                                device_id = gr.Dropdown(
                                    choices=[str(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"],
                                    value="0" if torch.cuda.is_available() else "cpu",
                                    label="GPU Device",
                                    info="Select GPU device to use"
                                )
                                batch_size = gr.Number(
                                    value=1,
                                    label="Batch Size",
                                    info="Number of tiles to process at once"
                                )
                            
                            with gr.Row():
                                model_dim = gr.Number(
                                    value=64,
                                    label="Model Dimension",
                                    info="Base dimension for model (don't change unless you know what you're doing)"
                                )
                                ffn_expansion_factor = gr.Number(
                                    value=2.66,
                                    label="FFN Expansion Factor",
                                    info="Expansion factor for feed-forward network"
                                )
                            
                            # Additional advanced parameters as free text
                            advanced_params = gr.Textbox(
                                label="Additional Parameters",
                                placeholder="Enter as key=value pairs, one per line or comma-separated",
                                lines=2
                            )
                            
                            # Custom model section
                            use_custom_model = gr.Checkbox(
                                label="Use Custom Model",
                                value=False
                            )
                            custom_model_group = gr.Group(visible=False)
                            with custom_model_group:
                                custom_model_path = gr.Textbox(
                                    label="Custom Model Path",
                                    placeholder="Path to custom model file (.pth)"
                                )
                                custom_config_path = gr.Textbox(
                                    label="Custom Config Path",
                                    placeholder="Path to custom config file (.yml)"
                                )
                            
                            # Make custom model section visible when checkbox is checked
                            use_custom_model.change(
                                fn=lambda x: gr.update(visible=x),
                                inputs=[use_custom_model],
                                outputs=[custom_model_group]
                            )
                        
                        # Status output
                        status_output = gr.Textbox(
                            label="Status",
                            placeholder="Upload a video and select processing options..."
                        )
                        
                    # Output section
                    with gr.Column(scale=1):
                        with gr.Column(elem_classes="video-container"):
                            gr.Markdown("### Result Video")
                            result_video = gr.Video(label="")
                        
                        with gr.Column(elem_classes="video-container"):
                            gr.Markdown("### Side-by-Side Comparison")
                            comparison_video = gr.Video(label="")

    return app

if __name__ == "__main__":
    app = create_ui()
    app.queue().launch(share=True)
