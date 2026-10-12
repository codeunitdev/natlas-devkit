"""Web playground for trying N-ATLaS in Hausa, Yoruba, Igbo and English.

Run:  natlas playground            (local)
      natlas playground --share    (temporary public link, e.g. from Colab)
"""

from __future__ import annotations

from typing import Any, List, Optional

from .client import ATTRIBUTION, DEFAULT_SYSTEM_PROMPT, LANGUAGES

LANGUAGE_CHOICES = ["Auto (match my language)"] + list(LANGUAGES.values())

# Starter prompts. The Hausa one is the example from the N-ATLaS model card.
# Have a native speaker review/replace these before a public demo.
EXAMPLES = [
    "menene ake nufi da gwagwarmaya",
    "Báwo ni mo ṣe lè dáàbò bo ara mi lọ́wọ́ ibà?",
    "Kedu ka m ga-esi chebe onwe m pụọ n'ọrịa ịba?",
    "Explain in simple terms how a farmer can save money with a cooperative.",
    "Translate into Hausa: Wash your hands with soap before eating.",
]


def _system_prompt(language: str) -> str:
    if language in LANGUAGES.values():
        return DEFAULT_SYSTEM_PROMPT + f" Always reply in {language}."
    return DEFAULT_SYSTEM_PROMPT


def build_demo(engine: Optional[Any] = None):
    """Create the Gradio app. Pass an already-loaded NAtlas to reuse it."""
    import gradio as gr

    state = {"engine": engine}

    def get_engine():
        if state["engine"] is None:
            from .client import NAtlas

            state["engine"] = NAtlas()
        return state["engine"]

    def respond(message: str, history: List[dict], language: str, temperature: float, max_tokens: int):
        if not message or not message.strip():
            yield history, ""
            return
        history = list(history or []) + [{"role": "user", "content": message}]
        convo = [{"role": "system", "content": _system_prompt(language)}] + [
            {"role": h["role"], "content": h["content"]} for h in history
        ]
        history.append({"role": "assistant", "content": ""})
        for piece in get_engine().stream_chat(
            convo, temperature=temperature, max_new_tokens=int(max_tokens)
        ):
            history[-1]["content"] += piece
            yield history, ""

    with gr.Blocks(title="N-ATLaS Playground") as demo:
        gr.Markdown(
            "# N-ATLaS Playground\n"
            "Chat with Nigeria's open language model in **Hausa, Yorùbá, Igbo** "
            "or **English**. Part of the open-source N-ATLaS DevKit by CodeUnit."
        )
        try:  # Gradio 4/5
            chatbot = gr.Chatbot(type="messages", height=440, label="Conversation")
        except TypeError:  # Gradio 6+: messages format is the default
            chatbot = gr.Chatbot(height=440, label="Conversation")
        with gr.Row():
            box = gr.Textbox(
                placeholder="Type in Hausa, Yorùbá, Igbo or English…",
                show_label=False,
                scale=5,
            )
            send = gr.Button("Send", variant="primary", scale=1)
        with gr.Accordion("Settings", open=False):
            language = gr.Dropdown(LANGUAGE_CHOICES, value=LANGUAGE_CHOICES[0], label="Reply language")
            temperature = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="Temperature")
            max_tokens = gr.Slider(64, 1024, value=400, step=32, label="Max new tokens")
        gr.Examples(EXAMPLES, inputs=box, label="Try one")
        clear = gr.Button("Clear chat")
        gr.Markdown(f"<small>{ATTRIBUTION} Outputs can be wrong; check important answers.</small>")

        inputs = [box, chatbot, language, temperature, max_tokens]
        box.submit(respond, inputs, [chatbot, box])
        send.click(respond, inputs, [chatbot, box])
        clear.click(lambda: [], None, chatbot)

    return demo


def main(share: bool = False, port: int = 7860) -> None:
    build_demo().queue().launch(share=share, server_port=port)
