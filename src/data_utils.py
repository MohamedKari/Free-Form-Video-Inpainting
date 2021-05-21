from typing import List, Tuple
import glob
import torch
from pathlib import Path
from io import BytesIO

from PIL import Image, ImageOps
import numpy as np

def to_tensor(image: Image.Image, device="cuda") -> torch.Tensor:
    image_np = np.asarray(image)
    
    assert image_np.dtype == np.uint8

    image_tensor = torch.tensor(image_np, device=device)
    image_tensor = image_tensor.type(torch.float32)
    image_tensor = image_tensor / 255.
    
    if len(image_tensor.shape) == 3:
        # color image
        image_tensor = image_tensor.permute([2, 0, 1])
    elif len(image_tensor.shape) == 2:
        # greyscale image 
        image_tensor = torch.unsqueeze(image_tensor, 0)
    
    
    return image_tensor

def load_data(image_directory, device="cuda") -> Tuple[List[torch.tensor], List[torch.tensor]]:
    print("getting frame_paths")
    frame_paths = sorted(glob.glob(str(Path(image_directory) / "1-resized/batch/*.jpg")))[0:15]
    mask_paths = sorted(glob.glob(str(Path(image_directory) / "3-mask/batch/*.png")))[0:15]
    print("frame_paths", frame_paths, flush=True)
    print("mask_paths", mask_paths, flush=True)

    width, height = 500, 500

    frames = [ Image.open(str(frame_path)) for frame_path in frame_paths ]
    frames = [ frame.resize((width, height)) for frame in frames ]
    frames = [ to_tensor(frame, device) for frame in frames ]
   
    masks = [ Image.open(str(mask_path)).convert("L") for mask_path in mask_paths ]
    masks = [ masks.resize((width, height)) for masks in masks ]
    masks = [ ImageOps.invert(mask) for mask in masks ]
    masks = [ mask.point(lambda x: 0 if x < 255 else 255) for mask in masks ]
    masks = [ to_tensor(mask, device) for mask in masks ]

    return frames, masks

def tensor_bytes_to_tensor(tensor_bytes: bytes, device="cuda") -> torch.Tensor:
    bytesio = BytesIO(tensor_bytes)
    bytesio.seek(0)
    tensor = torch.load(bytesio, map_location=device)
    return tensor

def tensor_to_tensor_bytes(tensor: torch.Tensor) -> bytes:
    if tensor is None:
        return bytes(False)

    tensor_bytesio = BytesIO()
    torch.save(tensor, tensor_bytesio)
    tensor_bytesio.seek(0)
    tensor_bytes = tensor_bytesio.read()
    return tensor_bytes