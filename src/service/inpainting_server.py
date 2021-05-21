


from typing import Dict
from concurrent.futures import ThreadPoolExecutor
import logging
from io import BytesIO
import signal
import sys
import pprint
import time

import grpc
import torch
from torch.tensor import Tensor

from inpainter import Inpainter

from service.log import setup_logger

from service.inpainting_service_pb2_grpc import InpainterServicer, add_InpainterServicer_to_server

from service.inpainting_service_pb2 import (
    Empty,
    StartInpaintingSessionResponse,
    InpaintRequest,
    InpaintResponse, 
    ReportBenchmarksResponse)

from data_utils import tensor_to_tensor_bytes, tensor_bytes_to_tensor

class InpainterServicer(InpainterServicer):

    def __init__(self):
        super().__init__()

        self.sessions: Dict[int, Inpainter] = {}
        self.total_session_counter: int = -1

    def log_request(self, rpc_name, frame_id, request, context): # pylint: disable=unused-argument
        logging.getLogger(__name__).debug(
            "[%s] Received gRPC request for method %s by peer %s with metadata %s", 
            frame_id,
            rpc_name,
            context.peer(),
            context.invocation_metadata())

    def Ping(self, request: Empty, context) -> Empty:
        self.log_request("Ping", None, request, context)
        return Empty()

    def StartInpaintingSession(self, request, context) -> StartInpaintingSessionResponse:
        self.log_request("StartInpaintingSession", None, request, context)
        
        self.total_session_counter += 1

        session_id = str(self.total_session_counter)
        self.sessions[session_id] = Inpainter()
        
        context.set_code(grpc.StatusCode.OK)
        context.set_details('Created new session.')
        
        return StartInpaintingSessionResponse(
            SessionId=session_id
        )

    def Inpaint(self, request: InpaintRequest, context) -> InpaintResponse:
        try:
            session_id: str = request.SessionId
            image_bytes: bytes = request.Image
            mask_bytes: bytes = request.Mask
            logging.getLogger(__name__).debug("retrieving inpainter for session_id %s", session_id)
            inpainter = self.sessions[session_id]

            self.log_request("Inpaint", None, request, context)

            # with benchmarks("deep_video_inpainting.input_deserialization"):
            image_tensor = tensor_bytes_to_tensor(image_bytes)
            mask_tensor = tensor_bytes_to_tensor(mask_bytes)
            logging.getLogger(__name__).debug("Received image tensor of shape %s and mask tensor of shape %s ", image_tensor.shape, mask_tensor.shape)

            # with benchmarks("deep_video_inpainting.inpainting"):
            inpainted_frame_tensor = inpainter.inpaint_next(
                image_tensor,
                mask_tensor)

            # with benchmarks("deep_video_inpainting.output_serialization"):
            # logging.getLogger(__name__).debug("inpainted_frame.shape: %s", inpainted_frame_tensor.shape)
            inpainted_bytes = tensor_to_tensor_bytes(inpainted_frame_tensor)

            context.set_code(grpc.StatusCode.OK)
            context.set_details("Everything OK.")

            # logging.getLogger(__name__).debug("[%s] Responding with offsetted frame ...", inpainted_frame_id)
            return InpaintResponse(
                InpaintedFrame=inpainted_bytes,
                InpaintedFrameId=inpainter.frame_counter
            )
        except Exception as e:
            print(type(e))
            print(e)
            raise e

"""
    def ReportBenchmarks(self, request: google_dot_protobuf_dot_empty__pb2, context) -> ReportBenchmarksResponse:
        self.log_request("ReportBenchmarks", None, request, context)

        benchmark_report = pprint.pformat(benchmarks.report())

        return ReportBenchmarksResponse(
            BenchmarkReport=benchmark_report
        )


"""

def register_stop_signal_handler(grpc_server):

    def signal_handler(signalnum, _):
        logging.getLogger(__name__).info("Processing signal %s received...", signalnum)
        grpc_server.stop(None)
        sys.exit("Exiting after cancel request.")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def serve():
    server = grpc.server(
        ThreadPoolExecutor(max_workers=1),
        options=[
            ("grpc.max_send_message_length", 20_000_000),
            ("grpc.max_receive_message_length", 20_000_000),
            ("grpc.max_message_length", 20_000_000)
        ])

    add_InpainterServicer_to_server(
        InpainterServicer(), 
        server)

    server.add_insecure_port("[::]:50051")
    register_stop_signal_handler(server)
    server.start()
    
    while True:
        time.sleep(24 * 60 * 60)

    # server.wait_for_termination()

if __name__ == "__main__":
    setup_logger()
    serve()