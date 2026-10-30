"""30 Hz WebSocket transport for contract v1 GameState."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from hud.config import AppConfig, endpoint
from hud.fusion.state import FusionState

# ponytail: a client stalled this many events behind loses the oldest ones; raise if seen.
_MAX_CARRIED_EVENTS = 256
_BROADCAST_HZ = 30  # CONTRACTS: Fusion -> Clients at 30 Hz


async def run_at_rate(
    tick: Callable[[], None],
    hz: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Call ``tick`` on a fixed monotonic grid of 1/hz; a late tick skips missed slots."""
    period = 1 / hz
    deadline = clock()
    while True:
        tick()
        deadline += period
        now = clock()
        if deadline < now:  # late by a full period or more: drop the missed ticks
            deadline += math.ceil((now - deadline) / period) * period
        await sleep(deadline - now)


def offer_latest(queue: asyncio.Queue[dict[str, Any]], snapshot: dict[str, Any]) -> None:
    """Replace an unsent snapshot with the newest one, carrying its unsent events forward.

    Events are drained once per snapshot for all clients, so dropping a stale frame must
    not drop its events (e.g. a `point_suggested` the operator still has to confirm).
    """
    if queue.full():
        try:
            missed = queue.get_nowait()
        except asyncio.QueueEmpty:
            missed = None
        if missed is not None and missed["events"]:
            events = (missed["events"] + snapshot["events"])[-_MAX_CARRIED_EVENTS:]
            snapshot = {**snapshot, "events": events}
    try:
        queue.put_nowait(snapshot)
    except asyncio.QueueFull:
        pass


def create_ws_app(state: FusionState) -> Any:
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    except ImportError as exc:
        raise RuntimeError("Install FastAPI and Uvicorn to run the WebSocket server") from exc

    globals()["WebSocket"] = WebSocket
    clients: set[asyncio.Queue[dict[str, Any]]] = set()

    def broadcast() -> None:
        snapshot = state.snapshot()
        for queue in tuple(clients):
            offer_latest(queue, snapshot)

    @asynccontextmanager
    async def lifespan(_app: Any) -> AsyncIterator[None]:
        task = asyncio.create_task(run_at_rate(broadcast, _BROADCAST_HZ))
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(lifespan=lifespan)

    @app.websocket("/")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
        clients.add(queue)

        async def send_latest() -> None:
            while True:
                snapshot = await queue.get()
                await websocket.send_text(json.dumps(snapshot, separators=(",", ":")))

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


async def ingest_worker_messages(state: FusionState, config: AppConfig) -> None:
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
        socket.connect(endpoint(config, worker))
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
