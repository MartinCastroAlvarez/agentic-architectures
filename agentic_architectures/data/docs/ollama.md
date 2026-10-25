# ollama

ollama runs open-weight language models on your own machine. The server listens on
port 11434 by default (http://127.0.0.1:11434) and exposes a chat API at `/api/chat`.
Pull a model with `ollama pull qwen3:4b`. Set `format` to `json` to ask for a JSON reply.
Keep the server bound to 127.0.0.1 unless you mean to share it on the network.
