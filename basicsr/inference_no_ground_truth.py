import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
from scipy.ndimage import gaussian_filter
import math
from statistics import mean
from PIL import Image

import os
import glob
import sys
from pathlib import Path
from tqdm import tqdm
import time

sys.path.append(str(Path(__file__).parents[1]))
from basicsr.utils.util import tensor2img

sys.path.append(str(Path(__file__).parents[3]))
from basicsr.utils.options import parse
from importlib import import_module


class VideoLoader(torch.utils.data.Dataset):
    def __init__(self, data_path, video):
        super().__init__()
        self.data_path = data_path
        self.in_files = sorted(glob.glob(data_path+ "/*.*"))
        self.len = len(self.in_files)
        print(f"> # of Frames in {video}: {len(self.in_files)}")
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.reverse = transforms.Compose([transforms.ToPILImage()])
        
        Img = Image.open(self.in_files[0])
        Img = np.array(Img)
        H, W, C = Img.shape
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_in = Image.open(self.in_files[index])
        img_in = np.array(img_in)
        

        return (None, 
                self.transform(np.array(img_in)).type(torch.FloatTensor))


def run_inference_patched(img_lq_prev,
                          img_lq_curr,
                          model, device,
                          tile,
                          tile_overlap,
                          prev_patch_dict_k=None, 
                          prev_patch_dict_v=None,
                          img_multiple_of = 8,
                          scale=1,
                          model_type='t0'):
    
    # Move input tensors to device just before processing
    img_lq_prev = img_lq_prev.to(device)
    img_lq_curr = img_lq_curr.to(device)
    
    height, width = img_lq_curr.shape[2], img_lq_curr.shape[3]
    
    H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
    padh = H-height if height%img_multiple_of!=0 else 0
    padw = W-width if width%img_multiple_of!=0 else 0

    img_lq_curr = torch.nn.functional.pad(img_lq_curr, (0, padw, 0, padh), 'reflect')
    img_lq_prev = torch.nn.functional.pad(img_lq_prev, (0, padw, 0, padh), 'reflect')
    
    # test the image tile by tile
    b, c, h, w = img_lq_curr.shape

    tile = min(tile, h, w)
    assert tile % 8 == 0, "tile size should be multiple of 8"
    tile_overlap = tile_overlap

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    
    # Create accumulators on CPU first
    E = torch.zeros(b, c, h, w)
    W = torch.zeros_like(E)

    print(f"E: {E.shape}")
    print(f"W: {W.shape}")

    patch_dict_k = {}
    patch_dict_v = {}
    
    # Process tile by tile
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            # Extract patches
            in_patch_curr = img_lq_curr[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            in_patch_prev = img_lq_prev[..., h_idx:h_idx+tile, w_idx:w_idx+tile]

            # prepare for SR following EAVSR.
            if model_type == "SR":
                in_patch_prev = torch.nn.functional.interpolate(in_patch_prev, 
                                                               scale_factor=1/4,
                                                               mode="bicubic")
                in_patch_curr = torch.nn.functional.interpolate(in_patch_curr, 
                                                               scale_factor=1/4, 
                                                               mode="bicubic")

            x = torch.concat((in_patch_prev.unsqueeze(0), 
                             in_patch_curr.unsqueeze(0)), dim=1)

            if prev_patch_dict_k is not None and prev_patch_dict_v is not None:
                # Load cached states to GPU only when needed
                key = f"{h_idx}-{w_idx}"
                if key in prev_patch_dict_k:
                    old_k_cache = [x.to(device) if x is not None else None for x in prev_patch_dict_k[key]]
                    old_v_cache = [x.to(device) if x is not None else None for x in prev_patch_dict_v[key]]
                else:
                    old_k_cache = None
                    old_v_cache = None
            else:
                old_k_cache = None
                old_v_cache = None
            
            # Run inference with mixed precision
            with torch.cuda.amp.autocast():
                out_patch, k_c, v_c = model(x.float(), old_k_cache, old_v_cache)
                
            # Immediately move results to CPU and free GPU memory
            patch_dict_k[f"{h_idx}-{w_idx}"] = [x.detach().cpu() if x is not None else None for x in k_c]
            patch_dict_v[f"{h_idx}-{w_idx}"] = [x.detach().cpu() if x is not None else None for x in v_c]
            
            # Move output to CPU and add to accumulated results
            out_patch_cpu = out_patch.detach().cpu()
            out_patch_mask = torch.ones_like(out_patch_cpu)
            
            E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_cpu)
            W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
            
            # Clear unnecessary references to free up memory
            del out_patch, k_c, v_c, out_patch_cpu, out_patch_mask
            if old_k_cache is not None:
                del old_k_cache, old_v_cache
            
            # Force CUDA to release memory
            torch.cuda.empty_cache()
    
    # Move input tensors back to CPU to free GPU memory
    img_lq_prev = img_lq_prev.cpu()
    img_lq_curr = img_lq_curr.cpu()
    
    # Complete the final computation on CPU
    restored = E.div_(W)
    restored = torch.clamp(restored, 0, 1)
    
    return restored, patch_dict_k, patch_dict_v

def load_model(path, model):
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device)['params'])
    model = model.to(device)
    model.eval()
    print(f"> Loaded Model.")
    return model, device

def flatten(xss):
    return [x for xs in xss for x in xs]

def run_inference(video_name, test_loader, 
                  model, device,
                  model_name,
                  save_img, do_patched, 
                  image_out_path, tile, 
                  tile_overlap,
                  model_type,
                  progress_callback=None):

    # Enable memory caching
    torch.backends.cudnn.benchmark = True

    # Set memory allocation strategy
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use up to 95% of GPU memory

    # Start timing
    start_time = time.time()
    frame_start_time = start_time

    previous_frame = None

    k_cache, v_cache = None, None
    total_frames = len(test_loader.dataset)
    
    for ix in range(total_frames):
        # Calculate progress and update if callback is provided
        if progress_callback is not None:
            # Convert to range 0.3-0.8 (assuming extraction was 0-0.3 and video creation will be 0.8-1.0)
            progress_value = 0.3 + (0.5 * (ix + 1) / total_frames)
            progress_callback(progress_value, f"Processing frame {ix+1}/{total_frames}")
        
        print(f"Processing frame {ix+1}/{total_frames}")
        current_frame = test_loader.dataset[ix][1]

        if previous_frame is None:
            previous_frame = current_frame
            
        c, h, w = current_frame.shape
        if do_patched:
            # do inference in patches, and concatenate before computing PSNR/SSIM.
            x2, k_cache, v_cache = run_inference_patched(
                                        previous_frame.unsqueeze(0),
                                        current_frame.unsqueeze(0),
                                        model, device, tile=tile, 
                                        tile_overlap=tile_overlap,
                                        prev_patch_dict_k=k_cache, 
                                        prev_patch_dict_v=v_cache,
                                        model_type=model_type)
        else:
            # superresolution
            if model_type == "SR":
                previous_frame = torch.nn.functional.interpolate(previous_frame.unsqueeze(0), 
                                                                scale_factor=1/4,
                                                                mode="bicubic").squeeze(0)
                current_frame = torch.nn.functional.interpolate(current_frame.unsqueeze(0), 
                                                                scale_factor=1/4, 
                                                                mode="bicubic").squeeze(0)
            # do inference on whole frame if the memory can be fit.
            x = torch.concat((previous_frame.unsqueeze(0), 
                            current_frame.unsqueeze(0)), dim=0).unsqueeze(0).to(device)
            x2, k_cache, v_cache = model(x, k_cache, v_cache)
            x2 = torch.clamp(x2, 0, 1)

        x2 = x2.squeeze(0)
        x2 = x2[:, :h, :w]
        
        if save_img:
            # Create necessary directories
            base_path = image_out_path
            base_path = os.path.join(base_path, model_name)
            os.makedirs(base_path, exist_ok=True)
            base_path = os.path.join(base_path, video_name)
            os.makedirs(base_path, exist_ok=True)

            # Save both input and prediction images separately
            file_name_inp = os.path.join(base_path, f"Frame_{ix+1}_Input.png") 
            file_name_pred = os.path.join(base_path, f"Frame_{ix+1}_Pred.png")   
            
            # Convert tensors to numpy arrays for saving
            input_img = current_frame.permute(1, 2, 0).detach().cpu().numpy()
            pred_img = x2.permute(1, 2, 0).detach().cpu().numpy()
            
            # Save images using OpenCV
            cv2.imwrite(file_name_inp, cv2.cvtColor((input_img*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(file_name_pred, cv2.cvtColor((pred_img*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        previous_frame = current_frame
        
        # Calculate and print time taken for current frame
        frame_end_time = time.time()
        frame_time = frame_end_time - frame_start_time
        print(f"Frame {ix+1} processed in {frame_time:.3f} seconds")
        frame_start_time = time.time()

    # Calculate and print time taken for current frame
    end_time = time.time()
    frames_processed = len(test_loader.dataset)
    print(f"Processed {frames_processed} frames in {end_time - start_time:.2f} seconds")
    print(f"FPS: {frames_processed / (end_time - start_time):.2f}")

    return None, None

def create_video_model(opt, model_type="t0"):
    if model_type == "t0":
        module = import_module('basicsr.models.archs.turtle_arch')
        model = module.make_model(opt)
    elif model_type == "t1":
        module = import_module('basicsr.models.archs.turtle_t1_arch')
        model = module.make_model(opt)
    elif model_type == "SR":
        module = import_module('basicsr.models.archs.turtlesuper_t1_arch')
        model = module.make_model(opt)
    else:
        print("Model type not defined")
        exit()
    return model

def main(model_path,
         model_name,
         data_dir,
         config_file,
         tile,
         tile_overlap,
         save_image,
         model_type,
         do_pacthes,
         image_out_path,
         noise_sigma=50.0/255.0,
         sample=True,
         y_channel_PSNR=False,
         progress_callback=None):

    print(f"model_type: {model_type}")
    print(f"do_patches: {do_pacthes}")
    print(f"tile: {tile}")
    print(f"tile_overlap: {tile_overlap}")
    print(F"sample: {sample}")
    
    opt = parse(config_file, is_train=True)
    model = create_video_model(opt, model_type)

    model, device = load_model(model_path, model)

    # The frames directory itself is our "video"
    video_name = os.path.basename(data_dir)
    
    # Create the data loader for all frames in this directory
    data = VideoLoader(data_dir, None)
    test_loader = torch.utils.data.DataLoader(data,
                                          batch_size=1, 
                                          num_workers=1, 
                                          shuffle=False)
    
    # Run inference once for all frames
    _, _ = run_inference(video_name,
                  test_loader,
                  model,
                  device,
                  model_name,
                  save_img=save_image,
                  do_patched=do_pacthes,
                  image_out_path=image_out_path,
                  tile=tile,
                  tile_overlap=tile_overlap,
                  model_type=model_type,
                  progress_callback=progress_callback)  # Pass the progress callback

    return 0, 0


if __name__ == "__main__":
    st = time.time()

    #----------------------------------------------------------------------------------------------------------
    #Super-Resolution

    """
    if the input video is already in lower spatial resolution than your desired output resolution, 
    this downsampling step should be commented in the inference code(lines 100 & 178).
    """
    config = "/options/Turtle_SR_MVSR.yml"
    model_path = "/trained_models/SuperResolution.pth"
    model_name = "SR_test"
    print(model_name)
    _, _ = main(model_path=model_path,
                model_name=model_name, 
                config_file=config,

                data_dir="/data_dir/", #Path to the desired video frames folder

                model_type="SR",

                save_image=True,
                image_out_path="/outputs/",

                do_pacthes=True,
                tile=320,
                tile_overlap=128)

    end = time.time()
    print(f"Completed in {end-st}s")
    
    # #----------------------------------------------------------------------------------------------------------
    # #desnowing
    # config = "/options/Turtle_Desnow.yml"
    # model_path = "/trained_models/Desnow.pth"
    # model_name = "Turtle_desnow_simple_full"
    # print(model_name)
    # _, _ = main(model_path=model_path,
    #             model_name=model_name, 
    #             config_file=config,

    #             data_dir="/data_dir/", #Path to the desired video frames folder

    #             model_type="t0",

    #             save_image=True,
    #             image_out_path="/outputs/",

    #             do_pacthes=True,
    #             tile=320,
    #             tile_overlap=128)

    # end = time.time()
    # print(f"Completed in {end-st}s")


    # # ----------------------------------------------------------------------------------------------------------
    # #Deraining, night
    # config = "/options/Turtle_Derain.yml"
    # model_path = "/trained_models/NightRain.pth"
    # model_name = "Turtle_Derain_simple_320_128_30"
    # print(model_name)
    # _, _ = main(model_path=model_path,
    #             model_name=model_name, 
    #             config_file=config,
    #             data_dir="/data_dir/", #Path to the desired video frames folder
    #             model_type="t0", #simple_parallel(For big patches), simple

    #             save_image=True,
    #             image_out_path="/outputs/",

    #             do_pacthes=True,
    #             tile=320,
    #             tile_overlap=128)

    # end = time.time()
    # print(f"Completed in {end-st}s")

    # ----------------------------------------------------------------------------------------------------------
    # Deraining, raindrop

    # config = "/options/Turtle_Derain_VRDS.yml"
    # model_path = "/trained_models/RainDrop.pth"
    # model_name = "Turtle_RainDrop_simple_320_128"
    # print(model_name)
    # _, _ = main(model_path=model_path,
    #             model_name=model_name, 
    #             config_file=config,
    #             data_dir="/data_dir/", #Path to the desired video frames folder
    #             model_type="t1", #simple_parallel(For big patches), simple

    #             save_image=True,
    #             image_out_path="/home/amir/codes/temp/",

    #             do_pacthes=True,
    #             tile=320,
    #             tile_overlap=128)

    # end = time.time()
    # print(f"Completed in {end-st}s")

    # #----------------------------------------------------------------------------------------------------------
    # # Deblurring, Gopro
    # config = "/options/Turtle_Deblur_Gopro.yml"
    # model_path = "/trained_models/net_g_200000.pth"
    # model_name = "Turtle_GoPro_simple_320_128_200k_kamran_no_pos"
    # print(model_name)
    # _, _ = main(model_path=model_path,
    #             model_name=model_name, 
    #             config_file=config,

    #             data_dir="/data_dir/", #Path to the desired video frames folder
    #             model_type="t1", #simple_parallel(For big patches), simple

    #             save_image=False,
    #             image_out_path="path_to_save_images",

    #             do_pacthes=True,
    #             tile=320,
    #             tile_overlap=128)

    # end = time.time()
    # print(f"Completed in {end-st}s")


    # #----------------------------------------------------------------------------------------------------------
    # # Deblur, BSD
    # #90Kmodel
    # config = "/options/Turtle_Derain_VRDS.yml"
    # model_path = "/trained_models/BSD.pth"
    # model_name = "Turtle_BSD082_simple_320_128"
    # print(model_name)
    # _, _ = main(model_path=model_path,
    #             model_name=model_name, 
    #             config_file=config,

    #             data_dir="/data_dir/", #Path to the desired video frames folder
    #             model_type="t0", #simple_parallel(For big patches), simple

    #             save_image=True,
    #             image_out_path="/home/amir/codes/temp/",

    #             do_pacthes=True,
    #             tile=320,
    #             tile_overlap=128)

    # end = time.time()
    # print(f"Completed in {end-st}s")
