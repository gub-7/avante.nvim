from typing import Literal

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "tool", "system"]
    content: str
    timestamp: str
    tool_name: str | None = None


class ChatTurnUpsert(BaseModel):
    base_uri: str
    chat_id: str
    title: str | None = None
    project_root: str
    messages: list[ChatMessage]
    updated_at: str

