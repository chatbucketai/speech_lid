# lid9

Spoken language identification for nine languages:

**english · hindi · telugu · tamil · kannada · malayalam · marathi · bengali · gujarati**

Audio in, language name out. Feature extraction is compiled into the ONNX model,
so the API takes raw audio — there is no front end to configure and nothing that
can drift out of sync.

## Project Structure

```
lid9/
├── __init__.py      inference library
├── cli.py           command-line interface
└── models/
    └── lid.onnx     quantized model (13 MB)
server.py            FastAPI HTTP server
auth.py              bearer token check (same as the STT services)
Dockerfile           multi-stage container build
pyproject.toml       package metadata
requirements.txt     pip dependencies
```

## Quick Start

### Docker (recommended)

```bash
docker build -t lid9 .
docker run -p 8000:8000 --read-only --cap-drop=ALL lid9
```

### Local

```bash
pip install -r requirements.txt
python server.py
```

## API

### `POST /v1/identify`

Needs `Authorization: Bearer <token>` - the same tokens the STT services accept
(see auth.py). Deployed as Cloud Run service `stt-lid`, public at
`https://<stt domain>/stt-lid/v1/identify`.

Upload an audio file, get the language back.

```bash
curl -X POST http://localhost:8000/v1/identify \
  -F "audio=@sample.wav"
```

```json
{
  "language": "telugu",
  "confidence": 0.9997,
  "duration": 4.832,
  "top_k": [
    {"language": "telugu", "probability": 0.9997},
    {"language": "malayalam", "probability": 0.0002}
  ]
}
```

Query parameters: `min_confidence` (0.0–1.0), `top_k` (1–9).

### `GET /health`

```json
{"status": "ok", "languages": ["english", "hindi", ...]}
```

## Python Library

```python
from lid9 import LanguageID

lid = LanguageID()              # construct ONCE — reuse across requests
r = lid.identify("sample.wav")

r.language      # 'telugu'
r.confidence    # 0.9997
r.top_k         # [('telugu', 0.9997), ('malayalam', 0.0002), ...]
```

`identify()` accepts a file path **or a numpy array**:

```python
lid.identify(samples)                     # 1-D float32 @ 16 kHz
lid.identify(samples, sample_rate=8000)   # resampled internally
lid.identify(stereo_array)                # downmixed automatically
```

## CLI

```bash
lid9 --audio sample.wav
# telugu

lid9 --audio sample.wav --top-k 3
lid9 --audio a.wav b.wav --json
```

## Notes

- **Construct `LanguageID` once.** Session creation costs ~100x a single
  inference. The server and CLI already do this.
- Clips under 0.3 s raise `AudioTooShort`.
- `identify()` is thread-safe on one instance.
- The hard cluster is Indo-Aryan (Hindi, Marathi, Gujarati, Bengali) —
  Dravidian languages separate easily.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `S2T_PORT` | `8000` | listen port (Cloud Run: 8080) |
| `S2T_DECODE_WORKERS` | `2` | onnxruntime threads |
| `S2T_MAX_UPLOAD_MB` | `30` | upload limit |
| `S2T_CORS_ORIGIN_REGEX` | empty | allowed browser origins |
| `S2T_TOKEN_KEY`, `S2T_TOKEN_KEY_B2B` | empty | HS256 token keys |
| `S2T_APP_TOKEN_PUBLIC_KEY` | empty | cb-backend RS256 public key |

With no key set, every request is accepted (local use only).
