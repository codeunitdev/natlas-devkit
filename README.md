# N-ATLaS DevKit

**Build with Nigeria's own language model in minutes.**
An open-source Python SDK, OpenAI-compatible API server, web playground and starter examples for
[N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS), the multilingual LLM for Hausa, Yorùbá, Igbo and Nigerian-accented English.

Built by **CodeUnit Technologies Limited** for the National AI Innovation Challenge (NAIC) 2026, Problem Statement 1: Developer Infrastructure.

> N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/codeunitdev/natlas-devkit/blob/main/notebooks/N_ATLAS_DevKit_Colab.ipynb)

---

## Why this exists

N-ATLaS is open, but using it still means every team repeats the same setup work: fitting an 8-billion-parameter model onto
affordable hardware, getting the chat template right, writing an API around it and testing output in each language.
The DevKit does that once, so Nigerian developers can spend their time on the product, not the plumbing.

| Without the DevKit | With the DevKit |
|---|---|
| ~16 GB GPU for full-precision weights | Runs in **~6 GB** (4-bit) on a free Colab T4 or a consumer GPU |
| Hand-written prompt formatting | Model's own chat template applied automatically |
| Custom API for every project | **OpenAI-compatible** endpoint: change one URL in existing apps |
| No quick way to try the model | Web playground with examples in four languages |

## What's included

| Component | What it does | Run it |
|---|---|---|
| **Python SDK** (`natlas.NAtlas`) | `ask`, `chat`, `stream_chat`, `translate`; 4-bit / 8-bit / full precision | `from natlas import NAtlas` |
| **API server** (`natlas.server`) | `/v1/chat/completions` (incl. streaming), `/v1/models`, `/health`; optional API key; CORS | `natlas serve` |
| **Playground** (`natlas.playground`) | Browser chat with language selector, temperature and length controls | `natlas playground` |
| **CLI** | Ask a question or chat in the terminal | `natlas chat "..."` |
| **Colab notebook** | End-to-end demo on a free GPU | `notebooks/` |
| **Examples** | Python, OpenAI client, JavaScript, curl | `examples/` |

## Architecture

```
            ┌──────────────────────── your apps ────────────────────────┐
            │  Python code   OpenAI SDK apps   JS / web   curl / SMS bots │
            └──────┬───────────────┬───────────────┬──────────────────────┘
                   │               └──── HTTP ─────┤
                   │                               ▼
                   │                ┌──────────────────────────┐   ┌───────────────┐
                   │                │ API server (FastAPI)     │   │ Playground    │
                   │                │ OpenAI-compatible, SSE   │   │ (Gradio)      │
                   │                └────────────┬─────────────┘   └──────┬────────┘
                   ▼                             ▼                        ▼
            ┌──────────────────────────────────────────────────────────────────┐
            │ natlas SDK: chat template · generation defaults · streaming ·     │
            │ language control · 4/8-bit loading · thread-safe GPU access      │
            └──────────────────────────────┬───────────────────────────────────┘
                                           ▼
                     N-ATLaS weights (NCAIR1/N-ATLaS, Llama-3 8B base)
                     downloaded once from Hugging Face, then run locally
```

Everything runs on your own hardware. After the one-time model download there are no calls to any third-party service.

## Quick start

### Option A: Google Colab (free, no setup)
Click **Open in Colab** above and follow the notebook.

### Option B: Your own machine (NVIDIA GPU)

1. Accept the model terms at <https://huggingface.co/NCAIR1/N-ATLaS> and create a Hugging Face read token.
2. Install and run:

```bash
pip install "natlas-devkit[all] @ git+https://github.com/codeunitdev/natlas-devkit.git"
export HF_TOKEN=hf_your_token

natlas chat "menene ake nufi da gwagwarmaya"   # one question
natlas serve --port 8000                        # API server
natlas playground --share                       # web playground
```

### Hardware

| Mode | GPU memory (approx.) | Good for |
|---|---|---|
| `4bit` (default) | ~6 GB | Colab T4, RTX 3060/4060, development |
| `8bit` | ~9 GB | Better fidelity on mid-range GPUs |
| `none` (float16) | ~16 GB | A10/A100/L4 servers, best quality |

CPU-only use is possible with `quantization="none"` but is slow; a llama.cpp/GGUF path for CPUs is on the roadmap.

## Python SDK

```python
from natlas import NAtlas

llm = NAtlas()                                         # loads NCAIR1/N-ATLaS in 4-bit

llm.ask("menene ake nufi da gwagwarmaya")              # replies in the user's language
llm.ask("What are the early signs of malaria?", language="yo")   # force Yorùbá
llm.translate("Wash your hands before eating.", to="Igbo")

result = llm.chat([
    {"role": "system", "content": "You are a patient maths tutor."},
    {"role": "user",   "content": "Explain fractions with an example about sharing suya."},
], max_new_tokens=300, temperature=0.2)
print(result.text, result.completion_tokens)

for piece in llm.stream_chat([{"role": "user", "content": "Tell me a short story."}]):
    print(piece, end="")
```

Generation defaults follow the model card (temperature 0.1, repetition penalty 1.12). Language codes: `en`, `ha`, `yo`, `ig` (or full names).

## API server

```bash
natlas serve --port 8000
# optional: NATLAS_API_KEY=secret natlas serve   → clients must send "Authorization: Bearer secret"
```

Use it with the official OpenAI client, unchanged except for `base_url`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
client.chat.completions.create(model="n-atlas", messages=[{"role": "user", "content": "Sannu!"}])
```

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions format. Supports `max_tokens`, `temperature`, `top_p`, `repetition_penalty`, `stream` |
| `GET /v1/models` | Lists `n-atlas` |
| `GET /health` | Status, whether the model is loaded, attribution |
| `GET /docs` | Interactive API documentation (Swagger UI) |

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face token (model is gated behind accepting terms) |
| `NATLAS_MODEL_ID` | `NCAIR1/N-ATLaS` | Model id or local path (e.g. a fine-tuned derivative) |
| `NATLAS_QUANT` | `4bit` | `4bit`, `8bit` or `none` |
| `NATLAS_API_KEY` | unset | If set, requests need `Authorization: Bearer <key>` |
| `NATLAS_LAZY_LOAD` | unset | `1` = load the model on first request instead of at startup |

## Tests

The API is tested with a fake model, so tests run in seconds without a GPU:

```bash
pip install -e ".[dev]" && pytest -q
```

## Roadmap

- **Starter templates:** SMS assistant (via CodeUnit's Flowritext SMS API), WhatsApp bot, document Q&A.
- **JavaScript/TypeScript SDK** published to npm, and **DialectScript** integration so learners can call N-ATLaS from code written in Hausa, Yorùbá, Igbo or Pidgin.
- **Voice template** using the N-ATLaS ASR models (speech in → text answer → speech out).
- **Evaluation toolkit:** repeatable quality checks per language, so teams can measure fine-tunes against the base model.
- **Fine-tuning starter:** LoRA recipe for sector data (health, agriculture, education).
- **CPU / low-cost deployment** via GGUF + llama.cpp.

## Known limitations (from the N-ATLaS model card)

- Human-evaluation average scores vary by language: English 4.21/5, Hausa 3.98, Igbo 3.87, Yorùbá 2.69. Yorùbá output needs extra review.
- Optimal context length is about 8K tokens.
- The model card reports some bias issues. Check important outputs, especially in health, legal and financial use.

## Licence and attribution

- **DevKit code:** MIT licence (see `LICENSE`).
- **N-ATLaS model:** *not* included in this repository. It is downloaded from Hugging Face and governed by the
  [N-ATLaS Terms of Use](https://huggingface.co/NCAIR1/N-ATLaS), including required attribution, a 1,000 active end-user cap
  for non-commercial use (larger deployments need a commercial licence from Awarri Technologies / FMCIDE) and a list of
  prohibited uses. Anyone deploying with this DevKit must follow those terms. See `NOTICE`.

## Team

**CodeUnit Technologies Limited** — builders of [DialectScript](https://dialectscript.github.io/core/) (code in Hausa, Yorùbá, Igbo and Pidgin),
Flowritext (SMS for apps), ErrorCatcher, and the Function Navigator and Database Navigator VS Code extensions.

- Yunusa Ibrahim
- Abubakar Usman

Issues and pull requests are welcome.
