#!/usr/bin/env bash
# Call the N-ATLaS DevKit API with curl. Start the server first: natlas serve
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "n-atlas",
        "messages": [{"role": "user", "content": "menene ake nufi da gwagwarmaya"}],
        "max_tokens": 200
      }'
echo
