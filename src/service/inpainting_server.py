


from typing import Dict
from concurrent.futures import ThreadPoolExecutor
import logging
from io import BytesIO
import signal
import sys
import pprint
import time

import grpc

from inpainter import Inpainter

from service.log import setup_logger

from service.inpainting_service_pb2_grpc import InpainterServicer, add_InpainterServicer_to_server

from service.inpainting_service_pb2 import (
    Empty,
    StartInpaintingSessionResponse,
    InpaintRequest,
    InpaintResponse, 
    ReportBenchmarksResponse)

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

"""

 

    def Inpaint(self, request: InpaintRequest, context) -> InpaintResponse:
        try:
            session_id: str = request.SessionId
            image: bytes = request.Image
            mask: bytes = request.Mask
            logging.getLogger(__name__).debug("retrieving inpainter for session_id %s", session_id)
            inpainter = self.sessions[session_id]

            self.log_request("Inpaint", inpainter.T, request, context)

            with benchmarks("deep_video_inpainting.input_deserialization"):

                image_bytesio = BytesIO(image)
                image_bytesio.seek(0)
                image_tensor = torch.load(image_bytesio, map_location="cpu") / 255.

                mask_bytesio = BytesIO(mask)
                mask_bytesio.seek(0)
                mask_tensor = torch.load(mask_bytesio, map_location="cpu") / 255.

            with benchmarks("deep_video_inpainting.inpainting"):
                inpainted_frame_np, inpainted_frame_id = inpainter.inpaint(
                    image_tensor,
                    mask_tensor)

            with benchmarks("deep_video_inpainting.output_serialization"):
                if inpainted_frame_np is not None: 
                    logging.getLogger(__name__).debug("inpainted_frame.shape: %s", inpainted_frame_np.shape)
                    inpainted_bytesio = BytesIO()
                    torch.save(inpainted_frame_np, inpainted_bytesio)
                    inpainted_bytesio.seek(0)
                    inpainted_bytes = inpainted_bytesio.read()
                else:
                    inpainted_bytes = bytes(False)

            context.set_code(grpc.StatusCode.OK)
            context.set_details("Everything OK.")

            logging.getLogger(__name__).debug("[%s] Responding with offsetted frame ...", inpainted_frame_id)
            return InpaintResponse(
                InpaintedFrame=inpainted_bytes,
                InpaintedFrameId=inpainted_frame_id
            )
        except Exception as e:
            print(type(e))
            print(e)
            raise e

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
            ("grpc.max_send_message_length", 10_000_000),
            ("grpc.max_receive_message_length", 10_000_000),
            ("grpc.max_message_length", 10_000_000)
        ])

    add_InpainterServicer_to_server(
        InpainterServicer(), 
        server)

    # cd src && python train.py -r ../data/v0.2.3_GatedTSM_inplace_noskip_b2_back_L1_vgg_style_TSMSNTPD128_1_1_10_1_VOR_allMasks_load135_e135_pdist0.1256.pth --dataset_config  other_configs/inference_example.json -od ../data/test_outputs

    server.add_insecure_port("[::]:50051")
    register_stop_signal_handler(server)
    server.start()
    
    while True:
        time.sleep(24 * 60 * 60)

    # server.wait_for_termination()

if __name__ == "__main__":
    setup_logger()
    serve()