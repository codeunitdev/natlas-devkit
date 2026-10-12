# Demo video script (about 4 minutes)

Record your screen (OBS Studio, Loom or the Windows Game Bar / macOS Cmd+Shift+5) with your voice.
Run the Colab notebook **before** recording so the model is already loaded; then re-run cells while recording.
Upload to YouTube as **Unlisted** and paste the link in the form.

| Time | On screen | What to say |
|---|---|---|
| 0:00–0:30 | README on GitHub | "We're CodeUnit Technologies. N-ATLaS is Nigeria's open language model, but every team has to repeat the same setup work before building anything. The N-ATLaS DevKit removes that work: an SDK, an OpenAI-compatible API and a playground, all open source." |
| 0:30–1:00 | Notebook cells 1–3 | "It installs with one command. With 4-bit loading, the 8-billion-parameter model fits in about 6 GB, so it runs on a free Colab GPU or an affordable laptop GPU, not only expensive servers." |
| 1:00–2:00 | Cell 4 output (four languages), then translate + streaming | "One call, four languages. Here it answers in Hausa, Yorùbá, Igbo and English. Developers can force the reply language, translate, and stream answers word by word." |
| 2:00–2:50 | Cell 5: API server + OpenAI client | "Most AI apps today are written for OpenAI's API. Our server speaks the same format, so an existing app switches to N-ATLaS by changing one line, the base URL. Streaming and API keys are supported." Briefly show `/docs`. |
| 2:50–3:30 | Playground link: ask something in Hausa, change reply language | "For non-coders and for testing, there's a web playground with example prompts in every language." |
| 3:30–4:00 | README roadmap | "Next, with NAIC support: SMS and WhatsApp starter templates using our Flowritext SMS API, a JavaScript SDK, DialectScript integration so students can call N-ATLaS from code written in their own language, a voice template using N-ATLaS speech recognition, and an evaluation toolkit. Our goal is that any Nigerian developer can ship an N-ATLaS product in an afternoon." |

Tips: show real outputs, don't edit them. Keep the browser zoomed so text is readable. Mention the team names at the start or end.
