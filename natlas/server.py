"""OpenAI-compatible HTTP API for N-ATLaS.

Any app or library that speaks the OpenAI Chat Completions format can switch
to N-ATLaS by changing its base URL:

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything")
    client.chat.completions.create(model="n-atlas", messages=[...])

Run:  natlas serve --port 8000
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Iterator, List, Optional, Protocol, Union

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .client import ATTRIBUTION, ChatResult, normalize_messages

SERVED_MODEL_NAME = "n-atlas"


class Engine(Protocol):
    """Anything with these methods can back the server (the real model or a test fake)."""

    model_id: str

    def chat(self, messages: List[dict], **overrides: Any) -> ChatResult: ...

    def stream_chat(self, messages: List[dict], **overrides: Any) -> Iterator[str]: ...


# ----------------------------------------------------------------- schemas
class ContentPart(BaseModel):
    type: str = "text"
    text: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(p.text or "" for p in self.content if p.type == "text")


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = SERVED_MODEL_NAME
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(default=None, ge=1.0, le=2.0)
    stream: bool = False


# -------------------------------------------------------------------- app
def create_app(engine: Optional[Engine] = None, api_key: Optional[str] = None) -> FastAPI:
    """Build the API. If no engine is passed, N-ATLaS is loaded from env settings:

    NATLAS_MODEL_ID (default NCAIR1/N-ATLaS), NATLAS_QUANT (4bit|8bit|none),
    HF_TOKEN, and NATLAS_API_KEY (optional bearer key clients must send).
    """
    api_key = api_key if api_key is not None else os.environ.get("NATLAS_API_KEY")
    state = {"engine": engine}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Load the model at startup so the first request is fast.
        if os.environ.get("NATLAS_LAZY_LOAD") != "1":
            get_engine()
        yield

    app = FastAPI(
        title="N-ATLaS DevKit API",
        version="0.1.0",
        description="OpenAI-compatible API for N-ATLaS. " + ATTRIBUTION,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    def get_engine() -> Engine:
        if state["engine"] is None:
            from .client import DEFAULT_MODEL_ID, NAtlas

            state["engine"] = NAtlas(
                model_id=os.environ.get("NATLAS_MODEL_ID", DEFAULT_MODEL_ID),
                quantization=os.environ.get("NATLAS_QUANT", "4bit"),
            )
        return state["engine"]

    def check_key(authorization: Optional[str] = Header(default=None)) -> None:
        if not api_key:
            return
        if authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @app.get("/health")
    def health() -> dict:
        eng = state["engine"]
        return {
            "status": "ok",
            "model_loaded": eng is not None,
            "model_id": getattr(eng, "model_id", None),
            "attribution": ATTRIBUTION,
        }

    @app.get("/v1/models", dependencies=[Depends(check_key)])
    def list_models() -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": SERVED_MODEL_NAME,
                    "object": "model",
                    "created": 0,
                    "owned_by": "NCAIR / Awarri Technologies",
                }
            ],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(check_key)])
    def chat_completions(req: ChatCompletionRequest):
        try:
            messages = normalize_messages(
                [{"role": m.role, "content": m.text()} for m in req.messages],
                system_prompt=None,  # engine adds its own default if none given
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        overrides = {
            "max_new_tokens": req.max_completion_tokens or req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "repetition_penalty": req.repetition_penalty,
        }
        eng = get_engine()
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if req.stream:
            return StreamingResponse(
                _sse(eng, messages, overrides, completion_id, created),
                media_type="text/event-stream",
            )

        result = eng.chat(messages, **overrides)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": SERVED_MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        }

    return app


def _sse(eng: Engine, messages, overrides, completion_id: str, created: int) -> Iterator[str]:
    def chunk(delta: dict, finish: Optional[str] = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": SERVED_MODEL_NAME,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield chunk({"role": "assistant"})
    for piece in eng.stream_chat(messages, **overrides):
        yield chunk({"content": piece})
    yield chunk({}, finish="stop")
    yield "data: [DONE]\n\n"


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
