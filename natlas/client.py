"""Python SDK for N-ATLaS, Nigeria's open multilingual language model.

Loads NCAIR1/N-ATLaS from Hugging Face (optionally 4-bit quantised so it runs
on a single consumer GPU or a free Google Colab T4) and exposes a small,
friendly API:

    from natlas import NAtlas
    llm = NAtlas()
    print(llm.ask("menene ake nufi da gwagwarmaya"))
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_MODEL_ID = "NCAIR1/N-ATLaS"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, friendly assistant built on N-ATLaS. "
    "Reply in the same language as the user's message (Hausa, Yoruba, Igbo "
    "or English) unless the user asks for a different language."
)

LANGUAGES: Dict[str, str] = {
    "en": "English",
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",
}

ATTRIBUTION = (
    "N-ATLaS is an initiative of the Federal Ministry of Communications, "
    "Innovation and Digital Economy, and powered by Awarri Technologies."
)

Message = Dict[str, str]


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling settings. Defaults follow the N-ATLaS model card."""

    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.12


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"


def resolve_language(language: Optional[str]) -> Optional[str]:
    """Accept 'ha', 'Hausa', 'hausa' etc. Returns the display name or None."""
    if not language:
        return None
    key = language.strip().lower()
    if key in LANGUAGES:
        return LANGUAGES[key]
    for name in LANGUAGES.values():
        if name.lower() == key:
            return name
    raise ValueError(
        f"Unsupported language {language!r}. Use one of: "
        + ", ".join(f"{k} ({v})" for k, v in LANGUAGES.items())
    )


def normalize_messages(
    messages: List[Message], system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT
) -> List[Message]:
    """Validate chat messages and add a system prompt if none was given."""
    if not messages:
        raise ValueError("messages must contain at least one message")
    cleaned: List[Message] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"invalid role {role!r}; use system, user or assistant")
        if not isinstance(content, str):
            raise ValueError("message content must be a string")
        cleaned.append({"role": role, "content": content})
    if system_prompt and cleaned[0]["role"] != "system":
        cleaned.insert(0, {"role": "system", "content": system_prompt})
    return cleaned


class NAtlas:
    """A loaded N-ATLaS model with chat, streaming, ask and translate helpers.

    Args:
        model_id: Hugging Face model id (or a local path to the weights).
        quantization: "4bit" (default, ~6 GB VRAM), "8bit" (~9 GB) or "none"
            (full float16, ~16 GB). 4/8-bit need an NVIDIA GPU + bitsandbytes.
        device_map: passed to transformers ("auto" spreads over available GPUs).
        token: Hugging Face token. Defaults to the HF_TOKEN environment variable.
            You must accept the model's terms on huggingface.co first.
        system_prompt: default system prompt added when a chat has none.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        quantization: str = "4bit",
        device_map: str = "auto",
        token: Optional[str] = None,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
        generation: GenerationConfig = GenerationConfig(),
    ) -> None:
        if quantization not in ("4bit", "8bit", "none"):
            raise ValueError("quantization must be '4bit', '8bit' or 'none'")
        self.model_id = model_id
        self.quantization = quantization
        self.system_prompt = system_prompt
        self.generation = generation
        self._token = token or os.environ.get("HF_TOKEN")
        self._lock = threading.Lock()  # one GPU model: serialise generation
        self._load(device_map)

    # ------------------------------------------------------------------ setup
    def _load(self, device_map: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: Dict[str, Any] = {"device_map": device_map, "token": self._token}
        if self.quantization == "none":
            kwargs["torch_dtype"] = (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )
        else:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"{self.quantization} quantisation needs an NVIDIA GPU. "
                    "Use quantization='none' on CPU (slow, needs ~32 GB RAM), "
                    "or run on Google Colab with a T4 GPU."
                )
            from transformers import BitsAndBytesConfig

            if self.quantization == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self._token)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self.model.eval()

        # Llama-3 chat models end turns with <|eot_id|>; stop on it as well as EOS.
        eos_ids = [self.tokenizer.eos_token_id]
        eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot != self.tokenizer.unk_token_id and eot not in eos_ids:
            eos_ids.append(eot)
        self._eos_ids = [i for i in eos_ids if i is not None]

    # ---------------------------------------------------------------- prompts
    def build_prompt(self, messages: List[Message]) -> str:
        """Render messages with the model's own chat template (as on the model card)."""
        return self.tokenizer.apply_chat_template(
            normalize_messages(messages, self.system_prompt),
            add_generation_prompt=True,
            tokenize=False,
            date_string=datetime.now().strftime("%d %b %Y"),
        )

    def _inputs(self, messages: List[Message]):
        text = self.build_prompt(messages)
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
            self.model.device
        )

    def _gen_kwargs(self, cfg: GenerationConfig) -> Dict[str, Any]:
        kw: Dict[str, Any] = {
            "max_new_tokens": cfg.max_new_tokens,
            "repetition_penalty": cfg.repetition_penalty,
            "eos_token_id": self._eos_ids,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if cfg.temperature and cfg.temperature > 0:
            kw.update(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p)
        else:
            kw.update(do_sample=False)
        return kw

    def _config(self, **overrides: Any) -> GenerationConfig:
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self.generation, **clean)

    # ------------------------------------------------------------- generation
    def chat(self, messages: List[Message], **overrides: Any) -> ChatResult:
        """Generate a reply to an OpenAI-style list of messages.

        overrides: max_new_tokens, temperature, top_p, repetition_penalty.
        """
        import torch

        cfg = self._config(**overrides)
        with self._lock:
            inputs = self._inputs(messages)
            prompt_len = inputs["input_ids"].shape[-1]
            with torch.no_grad():
                out = self.model.generate(**inputs, **self._gen_kwargs(cfg))
        new_tokens = out[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        finish = "length" if len(new_tokens) >= cfg.max_new_tokens else "stop"
        return ChatResult(text, int(prompt_len), int(len(new_tokens)), finish)

    def stream_chat(self, messages: List[Message], **overrides: Any) -> Iterator[str]:
        """Like chat(), but yields text pieces as they are generated."""
        from transformers import TextIteratorStreamer

        cfg = self._config(**overrides)
        with self._lock:
            inputs = self._inputs(messages)
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            thread = threading.Thread(
                target=self.model.generate,
                kwargs={**inputs, **self._gen_kwargs(cfg), "streamer": streamer},
                daemon=True,
            )
            thread.start()
            for piece in streamer:
                if piece:
                    yield piece
            thread.join()

    # ------------------------------------------------------------ shortcuts
    def ask(
        self,
        prompt: str,
        language: Optional[str] = None,
        system: Optional[str] = None,
        **overrides: Any,
    ) -> str:
        """One-shot question. Optionally force the reply language ('ha', 'Yoruba'...)."""
        sys_prompt = system or self.system_prompt or DEFAULT_SYSTEM_PROMPT
        lang = resolve_language(language)
        if lang:
            sys_prompt += f" Always reply in {lang}."
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.chat(messages, **overrides).text

    def translate(self, text: str, to: str, **overrides: Any) -> str:
        """Translate text into English, Hausa, Yoruba or Igbo."""
        lang = resolve_language(to)
        prompt = (
            f"Translate the following text into {lang}. "
            f"Reply with only the translation.\n\n{text}"
        )
        overrides.setdefault("temperature", 0.0)
        return self.ask(prompt, **overrides)
