from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from whisperlivekit import WhisperLiveKit, parse_args
from whisperlivekit.audio_processor import AudioProcessor

import asyncio
import logging
import os, sys
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

kit = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global kit
    kit = WhisperLiveKit()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get():
    return HTMLResponse(kit.web_interface())


async def handle_websocket_results(websocket, results_generator):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            await websocket.send_json(response)
    except Exception as e:
        logger.warning(f"Error in WebSocket results handler: {e}")


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    audio_processor = AudioProcessor()

    await websocket.accept()
    logger.info("WebSocket connection opened.")
            
    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))

    try:
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except WebSocketDisconnect:
        logger.warning("WebSocket disconnected.")
    finally:
        websocket_task.cancel()
        await audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up.")

@app.get("/api/status")
async def get_status():
    """Returns the status of all ASR instances and client connections."""
    global kit
    
    if not kit:
        return {"error": "WhisperLiveKit not initialized"}
    
    # Get instance information
    instances = []
    for i, service in enumerate(kit.batch_services):
        # Get clients connected to this service
        clients = []
        for client_id, client_service in AudioProcessor._clients_per_service.items():
            if client_service == service:
                clients.append(str(client_id))
        
        instance_info = {
            "id": i,
            "backend": kit.args.backend,
            "model": kit.args.model if hasattr(kit.args, "model") else "unknown",
            "clients_connected": len(clients),
            "max_clients": AudioProcessor.MAX_CLIENTS_PER_INSTANCE,
            "client_ids": clients
        }
        instances.append(instance_info)
    
    # Calculate total clients directly from the clients we gathered
    total_clients = sum(len(instance_info["client_ids"]) for instance_info in instances)
    
    # Build the complete status response
    status = {
        "total_instances": len(kit.batch_services),
        "total_clients": total_clients,
        "max_clients_per_instance": AudioProcessor.MAX_CLIENTS_PER_INSTANCE,
        "instances": instances,
        "diarization_enabled": kit.args.diarization if hasattr(kit.args, "diarization") else False,
    }
    
    return status

@app.get("/monitor")
async def get_monitor():
    """Serves the monitoring interface HTML page."""
    html_content = ""
    monitor_path = "/home/musadiq/test/whisperlivekit/web/index.html"
    
    try:
        with open(monitor_path, "r") as f:
            html_content = f.read()
    except FileNotFoundError:
        return HTMLResponse(f"<html><body><h1>Error: Monitor page not found</h1><p>Could not find {monitor_path}</p></body></html>")
    
    return HTMLResponse(html_content)

def main():
    """Entry point for the CLI command."""
    import uvicorn
    
    args = parse_args()
    
    uvicorn_kwargs = {
        "app": "whisperlivekit.basic_server:app",
        "host":args.host, 
        "port":args.port, 
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }
    
    ssl_kwargs = {}
    if args.ssl_certfile or args.ssl_keyfile:
        if not (args.ssl_certfile and args.ssl_keyfile):
            raise ValueError("Both --ssl-certfile and --ssl-keyfile must be specified together.")
        ssl_kwargs = {
            "ssl_certfile": args.ssl_certfile,
            "ssl_keyfile": args.ssl_keyfile
        }


    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}

    uvicorn.run(**uvicorn_kwargs)

if __name__ == "__main__":
    main()
