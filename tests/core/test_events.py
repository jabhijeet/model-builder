import pytest
from model_builder.core.events import EventBus


async def test_emit_calls_subscriber():
    bus = EventBus()
    received = []

    async def handler(run_id, node_id, event_type, payload):
        received.append((run_id, node_id, event_type, payload))

    bus.subscribe(handler)
    await bus.emit(1, "ingest", "state_change", {"state": "running"})
    assert received == [(1, "ingest", "state_change", {"state": "running"})]


async def test_emit_calls_multiple_subscribers():
    bus = EventBus()
    counts = [0, 0]

    async def h1(*_): counts[0] += 1
    async def h2(*_): counts[1] += 1

    bus.subscribe(h1)
    bus.subscribe(h2)
    await bus.emit(1, "node", "event", {})
    assert counts == [1, 1]


async def test_unsubscribe():
    bus = EventBus()
    received = []

    async def handler(*args): received.append(args)

    bus.subscribe(handler)
    bus.unsubscribe(handler)
    await bus.emit(1, "node", "event", {})
    assert received == []


async def test_sync_subscriber_called():
    bus = EventBus()
    received = []

    def sync_handler(run_id, node_id, event_type, payload):
        received.append(event_type)

    bus.subscribe(sync_handler)
    await bus.emit(1, "n", "done", {})
    assert received == ["done"]
