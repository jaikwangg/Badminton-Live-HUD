"""30 Hz WebSocket transport for contract v1 GameState."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from hud.fusion.state import FusionState


def create_ws_app(state: FusionState) -> Any:
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    except ImportError as exc:
        raise RuntimeError("Install FastAPI and Uvicorn to run the WebSocket server") from exc

    globals()["WebSocket"] = WebSocket
    clients: set[asyncio.Queue[str]] = set()

    async def broadcast() -> None:
        while True:
            payload = json.dumps(state.snapshot(), separators=(",", ":"))
            for queue in tuple(clients):
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                try:
                    queue.put_nowait(payload)
                except asyncio.QueueFull:
                    pass
            await asyncio.sleep(1 / 30)

    @asynccontextmanager
    async def lifespan(_app: Any) -> AsyncIterator[None]:
        task = asyncio.create_task(broadcast())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(lifespan=lifespan)

    @app.websocket("/")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        clients.add(queue)

        async def send_latest() -> None:
            while True:
                await websocket.send_text(await queue.get())

        sender = asyncio.create_task(send_latest())
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            sender.cancel()
            clients.discard(queue)

    return app


async def ingest_worker_messages(state: FusionState) -> None:
    """Subscribe to bounded worker PUB sockets and update fusion state."""
    try:
        import zmq
        import zmq.asyncio
    except ImportError as exc:
        raise RuntimeError("Install pyzmq to receive worker messages") from exc
    context = zmq.asyncio.Context.instance()
    poller = zmq.asyncio.Poller()
    sockets = []
    for worker in ("players", "shuttle", "audio"):
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVHWM, 4)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.connect(f"ipc:///tmp/hud_{worker}")
        poller.register(socket, zmq.POLLIN)
        sockets.append(socket)
    try:
        while True:
            ready = await poller.poll(timeout=100)
            for socket, _ in ready:
                try:
                    state.ingest(await socket.recv_string())
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
    finally:
        for socket in sockets:
            socket.close(linger=0)
