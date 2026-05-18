"""FastAPI server for the MIRA web interface."""

import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from structagent.web.session import AgentSession
from structagent.web.tool_processor import ViewerUpdate

# Import tools to ensure they are registered
from structagent.tools import TOOL_SCHEMAS  # noqa: F401


app = FastAPI(title="MIRA Web Interface")

# Session management
sessions: dict[str, AgentSession] = {}
session_lock = asyncio.Lock()


def get_session(websocket: WebSocket) -> AgentSession:
    """Get or create a session for the given WebSocket connection."""
    # Use a hash of the WebSocket ID as session key
    session_id = str(id(websocket))
    if session_id not in sessions:
        sessions[session_id] = AgentSession()
    return sessions[session_id]


async def cleanup_session(websocket: WebSocket) -> None:
    """Remove session when WebSocket disconnects."""
    session_id = str(id(websocket))
    if session_id in sessions:
        del sessions[session_id]


@app.get("/api/health")
async def health_check() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({"status": "ok"})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for chat and viewer interactions."""
    await websocket.accept()
    session = get_session(websocket)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            message_type = message.get("type")
            payload = message.get("payload", {})

            if message_type == "chat_message":
                # Handle chat message
                query = payload.get("query")
                context = payload.get("context")

                if not query:
                    await websocket.send_json({
                        "event": "error",
                        "data": {"message": "No query provided"},
                    })
                    continue

                # Process chat and send events
                async for event in session.chat(query, context):
                    await websocket.send_json({
                        "event": event["event_type"],
                        "data": event["data"],
                    })

            elif message_type == "viewer_action":
                # Handle viewer action (e.g., user clicked something in viewer)
                action = payload.get("action")
                pdb_id = payload.get("pdb_id")

                if action == "set_pdb":
                    session.set_pdb_id(pdb_id)
                    await websocket.send_json({
                        "event": "viewer_update",
                        "data": {
                            "action": "set_pdb",
                            "pdb_id": pdb_id,
                            "message": f"PDB ID set to {pdb_id}",
                        },
                    })
                else:
                    await websocket.send_json({
                        "event": "error",
                        "data": {"message": f"Unknown viewer action: {action}"},
                    })

            else:
                await websocket.send_json({
                    "event": "error",
                    "data": {"message": f"Unknown message type: {message_type}"},
                })

    except WebSocketDisconnect:
        await cleanup_session(websocket)
    except Exception as e:
        await websocket.send_json({
            "event": "error",
            "data": {"message": str(e)},
        })
        await cleanup_session(websocket)


# Mount static files if the directory exists
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
