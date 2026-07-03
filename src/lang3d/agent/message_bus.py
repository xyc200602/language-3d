"""Message bus for cross-agent communication."""

from __future__ import annotations

import logging
import threading as _threading
import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """A message published on the message bus."""

    sender: str
    type: str  # "tool_call", "tool_result", "thinking", "status", "artifact"
    payload: Any = None
    timestamp: float = field(default_factory=_time.time)


class MessageBus:
    """Simple publish-subscribe message bus for inter-agent communication."""

    MAX_MESSAGES = 1000

    def __init__(self) -> None:
        self._messages: list[AgentMessage] = []
        self._subscribers: list[Callable[[AgentMessage], None]] = []
        self._lock = _threading.Lock()

    def subscribe(self, callback: Callable[[AgentMessage], None]) -> None:
        """Register a message handler callback."""
        with self._lock:
            self._subscribers.append(callback)

    def publish(self, message: AgentMessage) -> None:
        """Publish a message and notify all subscribers."""
        with self._lock:
            self._messages.append(message)
            if len(self._messages) > self.MAX_MESSAGES:
                self._messages = self._messages[-self.MAX_MESSAGES:]
            subs = list(self._subscribers)
        for callback in subs:
            try:
                callback(message)
            except Exception as e:
                logger.warning("message_bus subscriber failed: %s", e)

    def get_messages(
        self,
        agent_id: str | None = None,
        type: str | None = None,
    ) -> list[AgentMessage]:
        """Query messages by sender agent ID and/or message type."""
        with self._lock:
            results = self._messages
            if agent_id is not None:
                results = [m for m in results if m.sender == agent_id]
            if type is not None:
                results = [m for m in results if m.type == type]
            return list(results)

    def get_artifacts(self) -> list[str]:
        """Get all file paths reported as artifacts."""
        with self._lock:
            artifacts: list[str] = []
            for msg in self._messages:
                if msg.type == "artifact" and isinstance(msg.payload, str):
                    artifacts.append(msg.payload)
                elif msg.type == "artifact" and isinstance(msg.payload, list):
                    artifacts.extend(str(p) for p in msg.payload)
            return list(dict.fromkeys(artifacts))

    def clear(self) -> None:
        """Clear all messages."""
        with self._lock:
            self._messages.clear()
