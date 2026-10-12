// Call the N-ATLaS DevKit API from JavaScript (Node 18+ or the browser).
// Start the server first:  natlas serve --port 8000
// Run:                      node examples/javascript_fetch.mjs

const BASE_URL = process.env.NATLAS_URL ?? "http://localhost:8000/v1";

const res = await fetch(`${BASE_URL}/chat/completions`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "n-atlas",
    messages: [{ role: "user", content: "Báwo ni mo ṣe lè dáàbò bo ara mi lọ́wọ́ ibà?" }],
    max_tokens: 300,
  }),
});

if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
const data = await res.json();
console.log(data.choices[0].message.content);
