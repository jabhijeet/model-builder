import asyncio
from typing import Callable


class EventBus:
    def __init__(self):
        self._subscribers: list[Callable] = []

    def subscribe(self, callback: Callable) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        self._subscribers.remove(callback)

    async def emit(self, run_id: int, node_id: str, event_type: str, payload: dict) -> None:
        for cb in list(self._subscribers):
            if asyncio.iscoroutinefunction(cb):
                await cb(run_id, node_id, event_type, payload)
            else:
                cb(run_id, node_id, event_type, payload)
