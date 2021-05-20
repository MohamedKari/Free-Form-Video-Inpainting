from math import trunc
import os
import json
import glob
from pathlib import Path
from typing import List, Tuple
import time

import torch
from PIL import Image, ImageOps
import numpy as np
from torchvision.transforms.functional import to_pil_image

import model.free_form_inpainting_archs as module_arch
from train import override_data_setting, main

CHECKPOINT_PATH = "../data/v0.2.3_GatedTSM_inplace_noskip_b2_back_L1_vgg_style_TSMSNTPD128_1_1_10_1_VOR_allMasks_load135_e135_pdist0.1256.pth"
DATASET_CONFIG_PATH = "other_configs/inference_example.json"
OUTPUT_ROOT_DIRECTORY = "../data/test_outputs"
IMAGE_DIRECTORY = "../custom-examples/7-tram-frames-square/"


def to_tensor(image: Image.Image) -> torch.Tensor:
    image_np = np.asarray(image)
    
    assert image_np.dtype == np.uint8

    image_tensor = torch.tensor(image_np, device="cuda")
    image_tensor = image_tensor.type(torch.float32)
    image_tensor = image_tensor / 255.
    
    if len(image_tensor.shape) == 3:
        # color image
        image_tensor = image_tensor.permute([2, 0, 1])
    elif len(image_tensor.shape) == 2:
        # greyscale image 
        image_tensor = torch.unsqueeze(image_tensor, 0)
    
    
    return image_tensor


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def load_data() -> Tuple[List[torch.tensor], List[torch.tensor]]:
    print("getting frame_paths")
    frame_paths = sorted(glob.glob(str(Path(IMAGE_DIRECTORY) / "1-resized/batch/*.jpg")))[0:3]
    mask_paths = sorted(glob.glob(str(Path(IMAGE_DIRECTORY) / "3-mask/batch/*.png")))[0:3]
    print("frame_paths", frame_paths, flush=True)
    print("mask_paths", mask_paths, flush=True)

    width, height = 300, 300

    frames = [ Image.open(str(frame_path)) for frame_path in frame_paths ]
    frames = [ frame.resize((width, height)) for frame in frames ]
    frames = [ to_tensor(frame) for frame in frames ]
   
    masks = [ Image.open(str(mask_path)).convert("L") for mask_path in mask_paths ]
    masks = [ masks.resize((width, height)) for masks in masks ]
    masks = [ ImageOps.invert(mask) for mask in masks ]
    masks = [ mask.point(lambda x: 0 if x < 255 else 255) for mask in masks ]
    masks = [ to_tensor(mask) for mask in masks ]

    return frames, masks

class Inpainter:
    def __init__(self) -> None:

        print("Initing inpainter", flush=True)

        config = torch.load(CHECKPOINT_PATH)['config']
        dataset_config = json.load(open(DATASET_CONFIG_PATH))
        config = override_data_setting(config, dataset_config)
        # main(config, CHECKPOINT_PATH, OUTPUT_ROOT_DIRECTORY, None)

        checkpoint = torch.load(CHECKPOINT_PATH)
        self.model = get_instance(module_arch, 'arch', config)
        self.model = self.model.to("cuda:0")
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.summary()
        
        print("Loading data", flush=True)
        data = load_data()

        for frame, mask in zip(*data):
            print(frame.shape, mask.shape, flush=True)

        frames, masks = data 
        frame_tensor, mask_tensor = self.pack_to_input_tensor(frames, masks)
        print("frame_tensor.shape, mask_tensor.shape", frame_tensor.shape, mask_tensor.shape)
        
        inpainted_tensor = self.inpaint(frame_tensor, mask_tensor)
        batch_count, batch_size, channel_count, height, width = inpainted_tensor.shape

        timestamp = int(time.time() * 1000)
        output_dir = Path(OUTPUT_ROOT_DIRECTORY) / str(timestamp) 
        output_dir.mkdir(parents=True, exist_ok=False)
        for i in range(batch_size):
            inpainted_image = to_pil_image(inpainted_tensor[0,i].cpu())
            inpainted_image.save(str(output_dir / f"{i}.jpg"))



    def pack_to_input_tensor(self, frames: List[torch.Tensor], masks: List[torch.Tensor]):
        frames = torch.stack(frames)
        frames = torch.unsqueeze(frames, dim=0)

        masks = torch.stack(masks)
        masks = torch.unsqueeze(masks, dim=0)

        return frames, masks

    def inpaint(self, frames: torch.Tensor, masks: torch.Tensor):
        assert len(frames.shape) == 5
        assert len(masks.shape) == 5
        
        batch_count, batch_size, channel_count, height, width = frames.shape
        assert batch_count == 1
        assert channel_count == 3

        batch_count, batch_size, channel_count, height, width = masks.shape
        assert batch_count == 1
        assert channel_count == 1

        with torch.no_grad():
            start_time = time.time()
            output_dict = self.model(frames, masks, None)
            print(f"Model Inference took {time.time() - start_time}")

        return output_dict["outputs"]