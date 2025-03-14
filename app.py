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

# Add the repository path to sys.path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import Turtle modules
from basicsr.utils.options import parse
from importlib import import_module
from basicsr.inference_no_ground_truth import main as inference_no_gt
from basicsr.inference_no_ground_truth import VideoLoader
from video_to_frames import extract_frames

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
        
        if width * height > 3840 * 2160:
            return False, f"Video resolution ({width}x{height}) is very high. Consider resizing for faster processing."
        
        return True, f"Video validated. Resolution: {width}x{height}, Frames: {total_frames}, FPS: {fps:.1f}. Estimated processing time: {est_minutes}m {est_seconds}s"
    except Exception as e:
        return False, f"Error validating video: {str(e)}"

def process_video(video_path, task_name, tile_size, tile_overlap, output_format, use_custom_model=False, custom_model_path="", custom_config_path="", progress=gr.Progress()):
    """Process the uploaded video and return the result"""
    
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
        
        # Extract frames from video
        progress(0.05, "Preparing to extract frames from video")
        num_frames = extract_frames(video_path, frames_dir)
        
        if num_frames == 0:
            return None, None, "Failed to extract frames from video."
        
        progress(0.3, f"Extracted {num_frames} frames. Starting model inference.")
        
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
            progress_callback=lambda value, text: progress(0.3 + 0.5 * value, text)
        )
        
        # Create result video
        progress(0.8, "Creating result videos")
        
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
        
        # Set the video codec based on output format
        video_codec = 'avc1' if output_format == 'MP4' else 'vp9'
        video_ext = '.mp4' if output_format == 'MP4' else '.webm'
        
        # Create the comparison video with slider
        progress(0.85, "Creating comparison video")
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
        progress(0.95, "Creating output video")
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
        
        progress(1.0, f"Processing complete{gpu_info}")
        
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
        # Clean up on error
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
        
        return True, f"Image validated. Resolution: {width}x{height}. Estimated processing time: {est_time:.1f} seconds"
    except Exception as e:
        return False, f"Error validating image: {str(e)}"

def image_process(image_path, task_name, tile_size, tile_overlap, use_custom_model=False, custom_model_path="", custom_config_path="", progress=gr.Progress()):
    """Process a single image and return the result"""
    
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
        
        progress(0.1, "Preparing images for processing")
        
        # Save the image to the frames directory
        img = Image.open(image_path)
        img_save_path = os.path.join(frames_dir, "frame_0001.png")
        img.save(img_save_path)
        
        # Create a duplicate of the image for sequence processing
        # Since the model expects at least 2 frames
        img_save_path2 = os.path.join(frames_dir, "frame_0002.png")
        img.save(img_save_path2)
        
        # Run inference
        progress(0.3, "Running Turtle model inference")
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
            progress_callback=lambda value, text: progress(0.3 + 0.6 * value, text)  # Pass the progress callback
        )
        
        # Get the output image
        progress(0.9, "Retrieving processed image")
        result_dir = os.path.join(output_dir, f"{task_name}_{job_id}")
        output_files = [f for f in os.listdir(result_dir) if f.endswith('.png') and 'Pred' in f]
        
        if not output_files:
            return None, "No output image was produced."
        
        # Get the first processed image
        output_image_path = os.path.join(result_dir, output_files[0])
        output_image = Image.open(output_image_path)
        
        # Resource usage info
        gpu_info = ""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            gpu_info = f" | GPU Memory: {memory_used:.2f}GB used"
        
        progress(1.0, f"Processing complete{gpu_info}")
        
        # Return results
        return output_image, f"Image processing completed successfully.{gpu_info}"
        
    except Exception as e:
        # Print the full traceback for debugging
        import traceback
        trace = traceback.format_exc()
        print(f"Error during processing: {str(e)}")
        print(trace)
        return None, f"Error during processing: {str(e)}"
    
    finally:
        # Cleanup temporary files
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

def get_model_details(task_name):
    """Return details about the selected model"""
    if task_name not in SUPPORTED_TASKS:
        return "Task not found"
    
    task_info = SUPPORTED_TASKS[task_name]
    info = f"""
    # {task_name}
    
    **Model Path**: {task_info['model_path']}
    **Config File**: {task_info['config_file']}
    **Model Type**: {task_info['model_type']}
    
    ## Processing Settings
    - **Tile Size**: Larger values need more memory but may improve quality
    - **Tile Overlap**: Higher values reduce seam artifacts between tiles
    - **Output Format**: MP4 is more compatible, WebM has better compression
    
    ## Custom Model (Advanced)
    If you have your own trained Turtle model, you can specify custom paths.
    """
    return info

def create_turtle_gui():
    """Create and configure the Gradio interface with improved UI"""
    
    # Custom CSS for better accessibility
    custom_css = """
    /* High contrast theme */
    .primary-btn { background-color: #0056b3; color: white; }
    .stop-btn { background-color: #dc3545; color: white; }
    /* Focus indicators */
    :focus { outline: 3px solid #0056b3; }
    /* Larger text */
    .label { font-size: 16px; font-weight: bold; }
    """
    
    with gr.Blocks(title="Turtle Video Restoration", css=custom_css) as app:
        gr.Markdown("""
        <div role="banner">
            <h1 id="mainHeading">🐢 Turtle Video Restoration</h1>
            <p role="doc-subtitle">Restore videos with state-of-the-art AI models</p>
        </div>
        
        This is a web interface for the Turtle model, a state-of-the-art model for video restoration tasks such as super-resolution, deblurring, deraining, and more.
            
## Instructions
1. Select the task you want to perform
2. Upload a video or image
3. Adjust processing settings if needed
4. Click 'Process' to start restoration
5. View and download the results
            
For more information, see [the Turtle GitHub repository](https://github.com/kjanjua26/Turtle).
        """)
        
        # Add keyboard shortcuts via JS
        app.load(None, None, None, js="""
            function setup_shortcuts() {
                document.addEventListener('keydown', (e) => {
                    // Ctrl+Enter to process
                    if(e.ctrlKey && e.key === 'Enter') {
                        document.querySelector('button.primary-btn').click();
                    }
                    // Escape to cancel
                    if(e.key === 'Escape') {
                        document.querySelector('button.stop-btn').click();
                    }
                });
            }
            if (document.readyState === 'complete') {
                setup_shortcuts();
            } else {
                window.addEventListener('load', setup_shortcuts);
            }
        """)
        
        with gr.Tabs():
            with gr.TabItem("Video Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        task_selector = gr.Dropdown(
                            choices=list(SUPPORTED_TASKS.keys()),
                            label="Select Restoration Task",
                            value=list(SUPPORTED_TASKS.keys())[0]
                        )
                        
                        video_input = gr.Video(label="Upload Video")
                        validation_msg = gr.Textbox(label="Validation", interactive=False)
                        
                        with gr.Accordion("Processing Settings", open=True):
                            model_info = gr.Markdown("Select a task to view model details")
                            
                            # Structured controls instead of free-form text
                            tile_size = gr.Slider(
                                minimum=64, maximum=640, value=320, step=64, 
                                label="Tile Size (larger needs more memory)"
                            )
                            
                            tile_overlap = gr.Slider(
                                minimum=16, maximum=256, value=128, step=16, 
                                label="Tile Overlap"
                            )
                            
                            output_format = gr.Radio(
                                choices=["MP4", "WebM"], value="MP4", 
                                label="Output Format"
                            )
                            
                            # Custom model options
                            use_custom_model = gr.Checkbox(
                                label="Use custom model (advanced)", 
                                value=False
                            )
                            
                            with gr.Group(visible=False) as custom_model_group:
                                custom_model_path = gr.Textbox(
                                    label="Custom Model Path",
                                    placeholder="Path to your custom model .pth file"
                                )
                                custom_config_path = gr.Textbox(
                                    label="Custom Config File Path",
                                    placeholder="Path to your custom config .yml file"
                                )
                            
                            # Show/hide custom model inputs
                            use_custom_model.change(
                                lambda x: {"visible": x}, 
                                inputs=[use_custom_model], 
                                outputs=[custom_model_group]
                            )
                        
                        with gr.Row():
                            process_btn = gr.Button("Process Video", variant="primary", elem_classes="primary-btn")
                            cancel_btn = gr.Button("Cancel", variant="stop", elem_classes="stop-btn")
                            
                    with gr.Column(scale=1):
                        with gr.Tabs():
                            with gr.TabItem("Restored Video"):
                                video_output = gr.Video(label="Restored Video")
                                download_restored_btn = gr.Button("Download Restored Video")
                                
                            with gr.TabItem("Comparison (Slider)"):
                                comparison_output = gr.Video(label="Original vs Restored")
                                download_comparison_btn = gr.Button("Download Comparison")
                                
                        status_text = gr.Textbox(label="Status", lines=2)
                
                # Connect components with callbacks
                task_selector.change(
                    get_model_details, 
                    inputs=[task_selector], 
                    outputs=[model_info]
                )
                
                # Validate video when uploaded
                video_input.change(
                    validate_video,
                    inputs=[video_input],
                    outputs=[validation_msg]
                )
                
                process_btn.click(
                    process_video, 
                    inputs=[
                        video_input, task_selector, tile_size, tile_overlap, 
                        output_format, use_custom_model, custom_model_path, 
                        custom_config_path
                    ],
                    outputs=[video_output, comparison_output, status_text]
                )
                
                # Download functionality
                download_restored_btn.click(
                    lambda x: x if x else None,
                    inputs=[video_output],
                    outputs=[gr.File(label="Download")]
                )
                
                download_comparison_btn.click(
                    lambda x: x if x else None,
                    inputs=[comparison_output],
                    outputs=[gr.File(label="Download")]
                )
                
            with gr.TabItem("Image Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_task_selector = gr.Dropdown(
                            choices=list(SUPPORTED_TASKS.keys()),
                            label="Select Restoration Task",
                            value=list(SUPPORTED_TASKS.keys())[0]
                        )
                        
                        image_input = gr.Image(type="filepath", label="Upload Image")
                        img_validation_msg = gr.Textbox(label="Validation", interactive=False)
                        
                        with gr.Accordion("Processing Settings", open=True):
                            img_model_info = gr.Markdown("Select a task to view model details")
                            
                            # Structured controls for image processing
                            img_tile_size = gr.Slider(
                                minimum=64, maximum=640, value=320, step=64, 
                                label="Tile Size (larger needs more memory)"
                            )
                            
                            img_tile_overlap = gr.Slider(
                                minimum=16, maximum=256, value=128, step=16, 
                                label="Tile Overlap"
                            )
                            
                            # Custom model options for image processing
                            img_use_custom_model = gr.Checkbox(
                                label="Use custom model (advanced)", 
                                value=False
                            )
                            
                            with gr.Group(visible=False) as img_custom_model_group:
                                img_custom_model_path = gr.Textbox(
                                    label="Custom Model Path",
                                    placeholder="Path to your custom model .pth file"
                                )
                                img_custom_config_path = gr.Textbox(
                                    label="Custom Config File Path",
                                    placeholder="Path to your custom config .yml file"
                                )
                            
                            # Show/hide custom model inputs
                            img_use_custom_model.change(
                                lambda x: {"visible": x}, 
                                inputs=[img_use_custom_model], 
                                outputs=[img_custom_model_group]
                            )
                        
                        with gr.Row():
                            img_process_btn = gr.Button("Process Image", variant="primary", elem_classes="primary-btn")
                            img_cancel_btn = gr.Button("Cancel", variant="stop", elem_classes="stop-btn")
                            
                    with gr.Column(scale=1):
                        with gr.Row():
                            image_output = gr.Image(label="Restored Image")
                            
                        img_status_text = gr.Textbox(label="Status", lines=2)
                        download_img_btn = gr.Button("Download Restored Image")
                
                # Connect components with callbacks
                img_task_selector.change(
                    get_model_details, 
                    inputs=[img_task_selector], 
                    outputs=[img_model_info]
                )
                
                # Validate image when uploaded
                image_input.change(
                    validate_image,
                    inputs=[image_input],
                    outputs=[img_validation_msg]
                )
                
                img_process_btn.click(
                    image_process, 
                    inputs=[
                        image_input, img_task_selector, img_tile_size, 
                        img_tile_overlap, img_use_custom_model, 
                        img_custom_model_path, img_custom_config_path
                    ],
                    outputs=[image_output, img_status_text]
                )
                
                # Enable download of processed image
                download_img_btn.click(
                    lambda x: x if x else None,
                    inputs=[image_output],
                    outputs=[gr.File(label="Download Processed Image")]
                )
                
            with gr.TabItem("Help & About"):
                gr.Markdown("""
                # Help & About
                
                ## About Turtle
                
                Turtle is a state-of-the-art model for video restoration tasks. It was developed by researchers at Huawei and presented in the paper "Learning Truncated Causal History Model for Video Restoration" accepted at NeurIPS 2024.
                
                ## Supported Tasks
                
                - **Video Super-Resolution**: Enhances the resolution of low-resolution videos
                - **Video Deblurring**: Removes blur from videos
                - **Video Deraining**: Removes rain streaks from videos
                - **Rain Drop Removal**: Removes rain drops from videos
                - **Video Desnowing**: Removes snow from videos
                - **Video Denoising**: Removes noise from videos
                
                ## Processing Settings
                
                - **Tile Size**: Size of tiles for processing (64-640 pixels)
                  - Larger values produce better results but require more GPU memory
                  - If you experience out-of-memory errors, try reducing this value
                
                - **Tile Overlap**: Overlap between adjacent tiles (16-256 pixels)
                  - Higher values reduce visible seams at tile boundaries
                  - Values around 1/3 of tile size typically work well
                
                - **Output Format**: Choose between MP4 (more compatible) and WebM (better compression)
                
                ## Keyboard Shortcuts
                
                - **Ctrl+Enter**: Start processing
                - **Escape**: Cancel processing
                
                ## Troubleshooting
                
                - **Out of memory errors**: Try reducing the tile size
                - **Slow processing**: For faster results, use smaller videos or lower resolutions
                - **Poor results at tile boundaries**: Increase the tile overlap
                - **Model not found**: Ensure the model paths are correct if using custom models
                
                ## Citation
                
                ```
                @inproceedings{ghasemabadilearning,
                  title={Learning Truncated Causal History Model for Video Restoration},
                  author={Ghasemabadi, Amirhosein and Janjua, Muhammad Kamran and Salameh, Mohammad and Niu, Di},
                  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
                }
                ```
                """)
    
    return app

if __name__ == "__main__":
    app = create_turtle_gui()
    app.launch(share=True, debug=True)
