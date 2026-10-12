"""Quickstart: load N-ATLaS and talk to it in four languages.

Needs an NVIDIA GPU (~6 GB VRAM for 4-bit) and HF_TOKEN set, after accepting
the model terms at https://huggingface.co/NCAIR1/N-ATLaS
"""

from natlas import NAtlas

llm = NAtlas()  # 4-bit by default

# 1. One-shot question (replies in the language you ask in)
print(llm.ask("menene ake nufi da gwagwarmaya"))

# 2. Force the reply language
print(llm.ask("What are the early signs of malaria?", language="yo"))

# 3. Translation
print(llm.translate("Wash your hands with soap before eating.", to="Igbo"))

# 4. Multi-turn chat with streaming
history = [{"role": "user", "content": "Give me three tips for saving money as a student."}]
for piece in llm.stream_chat(history, max_new_tokens=300):
    print(piece, end="", flush=True)
print()
