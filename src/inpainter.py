from math import trunc
import os
import json
import glob
from pathlib import Path
from typing import List, Tuple
import time
from collections import deque

import torch
from PIL import Image, ImageOps
import numpy as np
from torchvision.transforms.functional import to_pil_image

import model.free_form_inpainting_archs as module_arch
from train import override_data_setting, main
from data_utils import load_data, to_tensor

CHECKPOINT_PATH = "../data/v0.2.3_GatedTSM_inplace_noskip_b2_back_L1_vgg_style_TSMSNTPD128_1_1_10_1_VOR_allMasks_load135_e135_pdist0.1256.pth"
DATASET_CONFIG_PATH = "other_configs/inference_example.json"
OUTPUT_ROOT_DIRECTORY = "../data/test_outputs"
IMAGE_DIRECTORY = "../custom-examples/7-tram-frames-square/"


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class Inpainter:
    def __init__(self) -> None:
        
        config = torch.load(CHECKPOINT_PATH)['config']
        dataset_config = json.load(open(DATASET_CONFIG_PATH))
        config = override_data_setting(config, dataset_config)
        checkpoint = torch.load(CHECKPOINT_PATH)

        self.model = get_instance(module_arch, 'arch', config)
        self.model = self.model.to("cuda:0")
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.summary()

        self.window_size = 4

        self.input_frame_chunk = deque(maxlen=self.window_size)
        self.input_mask_chunk = deque(maxlen=self.window_size)

        self.first_chunk_processed = False

        # self._test_realtime()

    def _test_one_off(self):
        print("Loading data", flush=True)
        data = load_data(IMAGE_DIRECTORY)

        for frame, mask in zip(*data):
            print(frame.shape, mask.shape, flush=True)

        frames, masks = data 
        frame_tensor, mask_tensor = self.pack_to_input_tensor(frames, masks)
        print("frame_tensor.shape, mask_tensor.shape", frame_tensor.shape, mask_tensor.shape)
        
        inpainted_tensor = Inpainter.inpaint_one_off(frame_tensor, mask_tensor)
        batch_count, batch_size, channel_count, height, width = inpainted_tensor.shape

        timestamp = int(time.time() * 1000)
        output_dir = Path(OUTPUT_ROOT_DIRECTORY) / str(timestamp) 
        output_dir.mkdir(parents=True, exist_ok=False)
        for i in range(batch_size):
            inpainted_image = to_pil_image(inpainted_tensor[0,i].cpu())
            inpainted_image.save(str(output_dir / f"{i}.jpg"))


    def _test_realtime(self):
        data = load_data(IMAGE_DIRECTORY)

        timestamp = int(time.time() * 1000)
        output_dir = Path(OUTPUT_ROOT_DIRECTORY) / str(timestamp) 
        output_dir.mkdir(parents=True, exist_ok=False)
        i = 0
        for frame, mask in zip(*data):
            print(frame.shape, mask.shape, flush=True)
            inpainted_tensor = self.inpaint_next(frame, mask)
            inpainted_image = to_pil_image(inpainted_tensor.cpu())
            inpainted_image.save(str(output_dir / f"{i}.jpg"))
            i += 1


    def pack_to_input_tensor(self, frames: List[torch.Tensor], masks: List[torch.Tensor]):
        frames = torch.stack(frames)
        frames = torch.unsqueeze(frames, dim=0)

        masks = torch.stack(masks)
        masks = torch.unsqueeze(masks, dim=0)

        return frames, masks


    def inpaint_one_off(self, frames_tensor: torch.Tensor, masks_tensor: torch.Tensor):
        assert len(frames_tensor.shape) == 5
        assert len(masks_tensor.shape) == 5
        
        batch_count, batch_size, channel_count, height, width = frames_tensor.shape
        assert batch_count == 1
        assert channel_count == 3

        batch_count, batch_size, channel_count, height, width = masks_tensor.shape
        assert batch_count == 1
        assert channel_count == 1

        with torch.no_grad():
            start_time = time.time()
            output_dict = self.model(frames_tensor, masks_tensor, None)
            print(f"Model Inference took {time.time() - start_time}")

        return output_dict["outputs"]


    def inpaint_next(self, frame: torch.Tensor, mask: torch.Tensor, ramp_up=True) -> torch.Tensor:
        assert len(frame.shape) == 3
        assert len(mask.shape) == 3
        assert frame.shape[0] == 3
        assert mask.shape[0] == 1

        input_frame = frame
        input_frame = torch.unsqueeze(input_frame, dim=0)
        input_frame = torch.unsqueeze(input_frame, dim=0)

        input_mask = mask
        input_mask = torch.unsqueeze(input_mask, dim=0)
        input_mask = torch.unsqueeze(input_mask, dim=0)


        assert torch.all((mask == 0.) + (mask == 1.)) # or
        assert torch.all((frame >= 0.) * (frame <= 1.)) # and

        print("valrange", torch.min(mask), torch.max(mask), torch.min(frame), torch.max(frame))

        self.input_frame_chunk.append(input_frame)
        self.input_mask_chunk.append(input_mask)

        if ramp_up and len(self.input_mask_chunk) < self.window_size:
            return torch.zeros_like(frame)
            
        input_tensor = torch.cat(list(self.input_frame_chunk), dim=1)
        input_mask_chunk = torch.cat(list(self.input_mask_chunk), dim=1)

        output_tensor = self.inpaint_one_off(input_tensor, input_mask_chunk)

        current_inpainted_frame = output_tensor[:,-1:]
        
        if self.first_chunk_processed:
            self.input_frame_chunk.pop()
            self.input_frame_chunk.append(current_inpainted_frame)

            self.input_mask_chunk.pop()
            self.input_mask_chunk.append(torch.ones_like(input_mask))
        else:

            self.input_frame_chunk = deque(
                torch.split(output_tensor, 1, dim=1),
                self.input_frame_chunk.maxlen
            )

            self.input_mask_chunk = deque(
                [torch.ones_like(input_mask)] * self.input_mask_chunk.maxlen,
                self.input_mask_chunk.maxlen
            )
            print("self.input_mask_chunk[0].shape", self.input_mask_chunk[0].shape)
            
            self.first_chunk_processed = True
            
        return current_inpainted_frame[0, 0]
    