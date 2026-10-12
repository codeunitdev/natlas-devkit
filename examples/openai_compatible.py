"""Use N-ATLaS through the official OpenAI Python client.

Start the server first:   natlas serve --port 8000
Then:                     pip install openai && python examples/openai_compatible.py

Existing OpenAI-based apps switch to N-ATLaS by changing only base_url.
"""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed-locally")

resp = client.chat.completions.create(
    model="n-atlas",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Reply in Hausa."},
        {"role": "user", "content": "How can a small farmer store grain so it does not spoil?"},
    ],
    max_tokens=300,
    temperature=0.1,
)
print(resp.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="n-atlas",
    messages=[{"role": "user", "content": "Kedu ka m ga-esi chebe onwe m pụọ n'ọrịa ịba?"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
