import logging

import grpc
import fire

from .inpainting_service_pb2 import Empty

from .inpainting_service_pb2_grpc import InpainterStub

class InpainterClient:

    def __init__(self, ip_address: str, port: str, ping_timeout_in_seconds: float = 60):
        
        self.ping_timeout_in_seconds = ping_timeout_in_seconds
        
        self.channel = grpc.insecure_channel(
            target=f"{ip_address}:{port}", 
            options=[
                ("grpc.max_send_message_length", 10_000_000),
                ("grpc.max_receive_message_length", 10_000_000),
                ("grpc.max_message_length", 10_000_000)
            ])    

        self.stub = InpainterStub(self.channel)
        
        logging.getLogger(__name__).info("Sending Ping to server with timeout %s ...)", self.ping_timeout_in_seconds)
        

        self.stub.Ping(
            Empty(), 
            wait_for_ready=True, 
            timeout=self.ping_timeout_in_seconds)

        self.frame_counter = 0
        
        response = self.stub.StartInpaintingSession(
            Empty(),
            wait_for_ready=True,
            timeout=self.ping_timeout_in_seconds
        )

        print("session_id", response.SessionId)


        # self.stub.


def run(ip_address: str , port: int):
    inpainter_client = InpainterClient(ip_address, port)

if __name__ == "__main__":
    fire.Fire(run)