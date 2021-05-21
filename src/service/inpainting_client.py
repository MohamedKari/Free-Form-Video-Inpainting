import logging
from pathlib import Path
import time
from io import BytesIO

import grpc
import fire
from torchvision.transforms.functional import to_pil_image
import torch

from .inpainting_service_pb2 import Empty, InpaintRequest

from .inpainting_service_pb2_grpc import InpainterStub

from ..data_utils import load_data

class InpainterClient:

    def __init__(self, ip_address: str, port: str, ping_timeout_in_seconds: float = 60):
        
        self.ping_timeout_in_seconds = ping_timeout_in_seconds
        
        self.channel = grpc.insecure_channel(
            target=f"{ip_address}:{port}", 
            options=[
                ("grpc.max_send_message_length", 20_000_000),
                ("grpc.max_receive_message_length", 20_000_000),
                ("grpc.max_message_length", 20_000_000)
            ])    

        self.stub = InpainterStub(self.channel)
        
        logging.getLogger(__name__).info("Sending Ping to server with timeout %s ...)", self.ping_timeout_in_seconds)
        
        self.stub.Ping(
            Empty(), 
            wait_for_ready=True, 
            timeout=self.ping_timeout_in_seconds)

        self.frame_counter = 0
        
    
    def test(self):
        response = self.stub.StartInpaintingSession(
            Empty(),
            wait_for_ready=True,
            timeout=self.ping_timeout_in_seconds
        )

        # return

        session_id = response.SessionId
        
        data = load_data("custom-examples/7-tram-frames-square/", device="cpu")

        timestamp = int(time.time() * 1000)
        output_dir = Path("client-output") / str(timestamp) 
        output_dir.mkdir(parents=True, exist_ok=False)
        
        i = 0
        for frame, mask in zip(*data):
            frame_bytesio = BytesIO()
            torch.save(frame, frame_bytesio)
            frame_bytesio.seek(0)
            frame_bytes = frame_bytesio.read()

            mask_bytesio = BytesIO()
            torch.save(mask, mask_bytesio)
            mask_bytesio.seek(0)
            mask_bytes = mask_bytesio.read()

            request = InpaintRequest(SessionId=session_id, Image=frame_bytes, Mask=mask_bytes)

            response = self.stub.Inpaint(request, wait_for_ready=True, timeout=self.ping_timeout_in_seconds)

            image_bytesio = BytesIO(response.InpaintedFrame)
            image_bytesio.seek(0)
            image_tensor = torch.load(image_bytesio, map_location="cpu")
            inpainted_image = to_pil_image(image_tensor)
            
            inpainted_image.save(str(output_dir / f"{i}.jpg"))
            i += 1

"""
message InpaintResponse {
    bytes InpaintedFrame = 1;
    int32 InpaintedFrameId = 2;
}
"""


def run(ip_address: str , port: int):
    inpainter_client = InpainterClient(ip_address, port)
    inpainter_client.test()

if __name__ == "__main__":
    fire.Fire(run)