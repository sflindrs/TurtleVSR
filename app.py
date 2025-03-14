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
    """
    # Get a sample frame to determine dimensions
    input_frames = sorted([f for f in os.listdir(input_frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    output_frames = sorted([f for f in os.listdir(output_frames_dir) if f.endswith('.png') and 'Pred' in f])
    
    if not input_frames or not output_frames:
        return "Error: No frames found"
    
    input_frame_path = os.path.join(input_frames_dir, input_frames[0])
    frame = cv2.imread(input_frame_path)
    h, w, _ = frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    
    total_frames = min(len(input_frames), len(output_frames))
    
    # Map between frame numbers in input and output
    frame_map = {}
    for output_frame in output_frames:
        # Extract frame number from names like "Frame_1_Pred.png"
        if '_' in output_frame:
            parts = output_frame.split('_')
            if len(parts) >= 2 and parts[0] == "Frame":
                try:
                    frame_num = int(parts[1])
                    frame_map[frame_num] = output_frame
                except ValueError:
                    continue
    
    for i, input_frame in enumerate(input_frames[:total_frames]):
        # Get corresponding frame number (i+1 because frames are usually 1-indexed)
        frame_num = i + 1
        
        # Calculate slider position based on frame index
        slider_position = int((i / total_frames) * w)
        
        # Read input frame
        input_img = cv2.imread(os.path.join(input_frames_dir, input_frame))
        
        # Find corresponding output frame
        if frame_num in frame_map:
            output_frame = frame_map[frame_num]
            output_img = cv2.imread(os.path.join(output_frames_dir, output_frame))
        else:
            # If we don't have a matching output frame, use the input frame
            output_img = input_img.copy()
        
        # Make sure the frames are the same size
        if output_img.shape[:2] != input_img.shape[:2]:
            output_img = cv2.resize(output_img, (input_img.shape[1], input_img.shape[0]))
        
        # Create combined frame with sliding comparison
        combined_frame = input_img.copy()
        combined_frame[:, :slider_position] = output_img[:, :slider_position]
        
        # Draw the slider line
        cv2.line(combined_frame, (slider_position, 0), (slider_position, h), (0, 255, 0), 2)
        
        video_writer.write(combined_frame)
    
    video_writer.release()
    return output_video_path

def process_video(video_path, task_name, advanced_options, progress=gr.Progress()):
    """Process the uploaded video and return the result"""
    
    if video_path is None:
        return None, None, "Please upload a video to process."
    
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
        # Parse advanced options
        options = {}
        if advanced_options:
            options_list = advanced_options.split('\n')
            for opt in options_list:
                if ':' in opt:
                    key, value = opt.split(':', 1)
                    options[key.strip()] = value.strip()
        
        # Get task-specific parameters
        task_params = SUPPORTED_TASKS[task_name]
        model_path = options.get('model_path', task_params['model_path'])
        config_file = options.get('config_file', task_params['config_file'])
        model_type = options.get('model_type', task_params['model_type'])
        
        # Determine processing options
        tile_size = int(options.get('tile_size', 320))
        tile_overlap = int(options.get('tile_overlap', 128))
        
        # Extract frames from video
        progress(0, "Extracting frames from video")
        extract_frames(video_path, frames_dir)
        
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
            progress_callback=lambda value, text: progress(value, text)  # Add this line to pass progress updates
        )
        
        # Create result video
        progress(0.8, "Creating result video")
        
        # Find the model output directory - it's structured as output_dir/task_name_job_id/frames_dir_basename
        model_dir = os.path.join(output_dir, f"{task_name}_{job_id}")
        frames_dir_basename = os.path.basename(frames_dir)
        result_dir = os.path.join(model_dir, frames_dir_basename)
        
        # If the result_dir doesn't exist, use model_dir as a fallback
        if not os.path.exists(result_dir):
            result_dir = model_dir
        
        # Get frame rate from the original video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Create the comparison video with slider
        comparison_video_path = os.path.join(temp_dir, f"comparison_{job_id}.mp4")
        comparison_video_path = create_comparison_with_slider(
            frames_dir,
            result_dir,
            comparison_video_path,
            fps=fps
        )
        
        # Create regular output video
        output_video_path = os.path.join(temp_dir, f"output_{job_id}.mp4")
        
        # Get all output frames - look for files with "Pred" in the name
        output_frames = sorted([
            f for f in os.listdir(result_dir) 
            if f.endswith('.png') and 'Pred' in f
        ])
        
        # Create a video from the output frames
        if output_frames:
            # Read the first frame to get dimensions
            sample_frame = cv2.imread(os.path.join(result_dir, output_frames[0]))
            h, w, _ = sample_frame.shape
            
            video_writer = cv2.VideoWriter(
                output_video_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                fps, 
                (w, h)
            )
            
            for frame_name in output_frames:
                frame = cv2.imread(os.path.join(result_dir, frame_name))
                video_writer.write(frame)
                
            video_writer.release()
        
        progress(1.0, "Processing complete")
        
        # Verify files exist
        if not os.path.exists(output_video_path):
            return None, None, f"Failed to create output video. Check log for details."
        
        if not os.path.exists(comparison_video_path):
            return output_video_path, None, "Output video created, but comparison video failed."
        
        # Return results
        return output_video_path, comparison_video_path, "Processing completed successfully."
        
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        # Clean up on error
        return None, None, f"Error during processing: {str(e)}\n{trace}"
    
    finally:
        # Don't clean up immediately so we can debug if needed
        # We can add a cleanup timer to remove these files after some time
        pass

def image_process(image_path, task_name, advanced_options, progress=gr.Progress()):
    """Process a single image and return the result"""
    
    if image_path is None:
        return None, "Please upload an image to process."
    
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
        # Parse advanced options
        options = {}
        if advanced_options:
            options_list = advanced_options.split('\n')
            for opt in options_list:
                if ':' in opt:
                    key, value = opt.split(':', 1)
                    options[key.strip()] = value.strip()
        
        # Get task-specific parameters
        task_params = SUPPORTED_TASKS[task_name]
        model_path = options.get('model_path', task_params['model_path'])
        config_file = options.get('config_file', task_params['config_file'])
        model_type = options.get('model_type', task_params['model_type'])
        
        # Determine processing options
        tile_size = int(options.get('tile_size', 320))
        tile_overlap = int(options.get('tile_overlap', 128))
        
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
            progress_callback=lambda value, text: progress(value, text)  # Pass the progress callback
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
        
        progress(1.0, "Processing complete")
        
        # Return results
        return output_image, "Image processing completed successfully."
        
    except Exception as e:
        # Clean up on error
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
    
    ## Advanced Settings (Optional)
    You can modify these settings in the Advanced Options section:
    
    ```
    model_path: {task_info['model_path']}
    config_file: {task_info['config_file']}
    model_type: {task_info['model_type']}
    tile_size: 320
    tile_overlap: 128
    ```
    """
    return info

def create_turtle_gui():
    """Create and configure the Gradio interface"""
    
    with gr.Blocks(title="Turtle Video Restoration") as app:
        with gr.Row():
            gr.Markdown("""
            # 🐢 Turtle Video Restoration
            
            This is a web interface for the Turtle model, a state-of-the-art model for video restoration tasks such as super-resolution, deblurring, deraining, and more.
            
            ## Instructions
            1. Select the task you want to perform
            2. Upload a video or image
            3. Click 'Process' to start restoration
            4. View the results
            
            For more information, see [the Turtle GitHub repository](https://github.com/kjanjua26/Turtle).
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
                        
                        with gr.Accordion("Advanced Options", open=False):
                            model_info = gr.Markdown("Select a task to view model details")
                            advanced_options = gr.Textbox(label="Advanced Options (key: value format)", lines=5)
                            
                        process_btn = gr.Button("Process Video", variant="primary")
                            
                    with gr.Column(scale=1):
                        with gr.Tabs():
                            with gr.TabItem("Restored Video"):
                                video_output = gr.Video(label="Restored Video")
                            with gr.TabItem("Comparison (Slider)"):
                                comparison_output = gr.Video(label="Original vs Restored")
                                
                        status_text = gr.Textbox(label="Status", lines=2)
                
                # Connect components with callbacks
                task_selector.change(get_model_details, inputs=[task_selector], outputs=[model_info])
                
                process_btn.click(
                    process_video, 
                    inputs=[video_input, task_selector, advanced_options],
                    outputs=[video_output, comparison_output, status_text]
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
                        
                        with gr.Accordion("Advanced Options", open=False):
                            img_model_info = gr.Markdown("Select a task to view model details")
                            img_advanced_options = gr.Textbox(label="Advanced Options (key: value format)", lines=5)
                            
                        img_process_btn = gr.Button("Process Image", variant="primary")
                            
                    with gr.Column(scale=1):
                        image_output = gr.Image(label="Restored Image")
                        img_status_text = gr.Textbox(label="Status", lines=2)
                
                # Connect components with callbacks
                img_task_selector.change(get_model_details, inputs=[img_task_selector], outputs=[img_model_info])
                
                img_process_btn.click(
                    image_process, 
                    inputs=[image_input, img_task_selector, img_advanced_options],
                    outputs=[image_output, img_status_text]
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
                
                ## Advanced Options
                
                Advanced users can customize the following parameters:
                
                - `model_path`: Path to a custom model
                - `config_file`: Path to a custom config file
                - `model_type`: Model architecture type (t0, t1, SR)
                - `tile_size`: Size of tiles for processing (larger values need more memory)
                - `tile_overlap`: Overlap between tiles (affects quality at tile boundaries)
                
                ## Troubleshooting
                
                - **Out of memory errors**: Try reducing the tile size
                - **Slow processing**: Check your GPU drivers or try a smaller video
                - **Model not found**: Ensure the model paths are correct in the advanced options
                
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
