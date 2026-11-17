"""Small provider-neutral tool-calling contract."""
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ModelReply:
    message: dict
    usage: dict = field(default_factory=dict)


class ChatModel(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> ModelReply: ...
