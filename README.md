# abliteration_node

ComfyUI node for [abliteration.ai](https://abliteration.ai) — an OpenAI-compatible, uncensored
inference API. Streams the model's reply token-by-token into the graph and broadcasts it live to
the ComfyUI frontend.

Built for prompt-writing in video workflows, where a silently truncated prompt costs a full GPU
render. **This node fails loudly instead.**

## Why not a generic OpenAI node

Most chat nodes return whatever text they got, even when the upstream call was cut short. A
half-written prompt then reaches the video model, which happily "fills in the blanks" — the subject
drifts, the scene smears, and you only find out after paying for the render.

`AbliterationChat` raises instead of forwarding damaged output:

| Condition | Behaviour |
|---|---|
| HTTP non-2xx | raise, with the upstream body attached |
| `finish_reason == "length"` | raise — and distinguishes *body truncated* from *reasoning ate the whole budget* (different fixes) |
| `finish_reason` not `stop` | raise (filtered / upstream error) |
| Empty body | raise |
| SSE stream ends with no `finish_reason` | raise — a half-received stream is indistinguishable from a complete one otherwise |

## Live text streaming

The node requests `stream: true` and accumulates `delta.content`. As text arrives it broadcasts a
`abliteration.text` websocket event (`{node, text, done}`), rate-limited to ≥120 new characters or
≥0.35 s:

- **Local runs** — the frontend can render the text into a display node while it generates.
- **Remote runs** — a bridge/worker watching the ComfyUI websocket can relay the same event back to
  a local UI, giving live text for graphs executed off-machine.

Broadcasting is strictly best-effort: any failure to emit is swallowed, never affecting the run.
If the upstream rejects `stream`, the node falls back to a single blocking request and emits one
final event.

The full reply is also printed to stdout between `===== GENERATED PROMPT BEGIN/END =====` markers,
so remote worker logs always retain what was actually generated.

## Inputs

| Input | Notes |
|---|---|
| `api_key` | abliteration.ai key. Sent as `Authorization: Bearer <key>` |
| `system_prompt` / `user_message_box` | Text. A connected `user_message_input` takes priority over the box |
| `model` | `abliterated-model` (multimodal, 256K) or `abliterated-model-large` (text-only, 1M) |
| `reasoning_effort` | Defaults to `none` — see below |
| `temperature`, `max_tokens`, `timeout_sec` | Standard |
| `image_1` (optional) | Reference image; JPEG-encoded to a data URI. Rejected by `-large` |

### `reasoning_effort` defaults to `none` on purpose

`max_tokens` is a **total** budget including reasoning tokens. Both models think by default, and
reasoning is spent first — so a low budget can be consumed entirely before a single output character
is produced. `include_reasoning: false` only suppresses the trace in the response; it does not save
tokens. Rewriting a prompt is a formatting task, so thinking is off by default.

Caching keys on image content as well as text, so swapping the reference image re-runs the node
instead of silently reusing the previous prompt.

## Install

Clone into `ComfyUI/custom_nodes/` and restart ComfyUI:

```bash
git clone git@github.com:AdamGoodApp/abliteration_node.git ComfyUI/custom_nodes/abliteration_node
```

No third-party dependencies beyond what ComfyUI already ships (`Pillow`, `numpy`); HTTP uses the
standard library only, which keeps container images one dependency lighter.

## License

MIT
