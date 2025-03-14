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
