"""API tests using a fake engine, so they run anywhere (no GPU, no model download)."""

import json

import pytest
from fastapi.testclient import TestClient

from natlas.client import ChatResult, normalize_messages, resolve_language
from natlas.server import create_app


class FakeEngine:
    model_id = "fake/n-atlas"

    def __init__(self):
        self.last_messages = None
        self.last_overrides = None

    def chat(self, messages, **overrides):
        self.last_messages, self.last_overrides = messages, overrides
        return ChatResult(text="Sannu! Yaya kake?", prompt_tokens=12, completion_tokens=5)

    def stream_chat(self, messages, **overrides):
        self.last_messages, self.last_overrides = messages, overrides
        yield from ["Sannu", "! ", "Yaya kake?"]


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def client(engine):
    return TestClient(create_app(engine=engine, api_key=""))


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "Awarri" in body["attribution"]


def test_models(client):
    assert client.get("/v1/models").json()["data"][0]["id"] == "n-atlas"


def test_chat_completion(client, engine):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "n-atlas",
            "messages": [{"role": "user", "content": "Sannu"}],
            "max_tokens": 50,
            "temperature": 0.2,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Sannu! Yaya kake?"
    assert body["usage"]["total_tokens"] == 17
    assert engine.last_overrides["max_new_tokens"] == 50
    assert engine.last_overrides["temperature"] == 0.2


def test_content_parts_are_joined(client, engine):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": [{"type": "text", "text": "Ẹ "}, {"type": "text", "text": "kú àárọ̀"}]}]},
    )
    assert r.status_code == 200
    assert engine.last_messages[-1]["content"] == "Ẹ kú àárọ̀"


def test_streaming(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Sannu"}], "stream": True},
    ) as r:
        lines = [l for l in r.iter_lines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[6:]) for l in lines[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "Sannu! Yaya kake?"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_bad_role_rejected(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "robot", "content": "hi"}]})
    assert r.status_code == 400


def test_api_key_required(engine):
    c = TestClient(create_app(engine=engine, api_key="secret"))
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert c.post("/v1/chat/completions", json=body).status_code == 401
    ok = c.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


def test_normalize_adds_system_prompt():
    msgs = normalize_messages([{"role": "user", "content": "hi"}])
    assert msgs[0]["role"] == "system"
    assert normalize_messages([{"role": "user", "content": "hi"}], system_prompt=None)[0]["role"] == "user"


def test_resolve_language():
    assert resolve_language("ha") == "Hausa"
    assert resolve_language("yoruba") == "Yoruba"
    assert resolve_language(None) is None
    with pytest.raises(ValueError):
        resolve_language("french")


from natlas.client import detect_language, system_prompt_for


@pytest.mark.parametrize("text,expected", [
    ("menene ake nufi da gwagwarmaya", "Hausa"),
    ("Yaya zan kare kaina daga zazzaɓin cizon sauro?", "Hausa"),
    ("Báwo ni mo ṣe lè dáàbò bo ara mi lọ́wọ́ ibà?", "Yoruba"),
    ("Kí ni àwọn àmì àrùn ibà?", "Yoruba"),
    ("Kedu ka m ga-esi chebe onwe m pụọ n'ọrịa ịba?", "Igbo"),
    ("Gịnị bụ ihe mgbaàmà nke ịba?", "Igbo"),
    ("Explain in simple terms how a farmer can save money.", None),
    ("", None),
])
def test_detect_language(text, expected):
    assert detect_language(text) == expected


def test_default_system_prompt_pins_reply_language():
    msgs = normalize_messages([{"role": "user", "content": "Báwo ni mo ṣe lè dáàbò bo ara mi lọ́wọ́ ibà?"}])
    assert "Always reply in Yoruba" in msgs[0]["content"]
    assert "Always reply" not in system_prompt_for("What is malaria?")
