"""Command line: natlas chat | serve | playground"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="natlas", description="N-ATLaS DevKit command line")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chat", help="Ask N-ATLaS a question (or chat interactively)")
    c.add_argument("prompt", nargs="?", help="Question. Omit for interactive mode.")
    c.add_argument("--lang", help="Force reply language: en, ha, yo, ig")
    c.add_argument("--quant", default="4bit", choices=["4bit", "8bit", "none"])

    s = sub.add_parser("serve", help="Run the OpenAI-compatible API server")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--quant", default=None, choices=["4bit", "8bit", "none"])

    g = sub.add_parser("playground", help="Run the web playground")
    g.add_argument("--port", type=int, default=7860)
    g.add_argument("--share", action="store_true", help="Create a temporary public link")

    args = p.parse_args(argv)

    if args.cmd == "serve":
        if args.quant:
            os.environ["NATLAS_QUANT"] = args.quant
        from .server import main as serve

        serve(host=args.host, port=args.port)
        return 0

    if args.cmd == "playground":
        from .playground import main as play

        play(share=args.share, port=args.port)
        return 0

    from .client import NAtlas

    llm = NAtlas(quantization=args.quant)
    if args.prompt:
        print(llm.ask(args.prompt, language=args.lang))
        return 0

    print("N-ATLaS chat. Type 'exit' to quit.")
    history = []
    while True:
        try:
            line = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() in ("exit", "quit"):
            break
        if not line:
            continue
        history.append({"role": "user", "content": line})
        print("N-ATLaS: ", end="", flush=True)
        reply = ""
        for piece in llm.stream_chat(history):
            reply += piece
            print(piece, end="", flush=True)
        print()
        history.append({"role": "assistant", "content": reply})
    return 0


if __name__ == "__main__":
    sys.exit(main())
