import asyncio
import io
import logging
import os

import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from auth import token_valid
from lid9 import AudioTooShort, LanguageID

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("lid9.server")

PORT = int(os.environ.get("S2T_PORT", "8000"))
THREADS = int(os.environ.get("S2T_DECODE_WORKERS", "2"))
MAX_UPLOAD_BYTES = int(os.environ.get("S2T_MAX_UPLOAD_MB", "30")) * 1024 * 1024
CORS_ORIGIN_REGEX = os.environ.get("S2T_CORS_ORIGIN_REGEX", "")

app = FastAPI(title="lid9", docs_url=None, redoc_url=None, openapi_url=None)

lid = LanguageID(threads=THREADS)


# Before the body is read, so an unauthenticated client can't make us receive
# the upload. Registered before CORS so browsers can still read the 401.
@app.middleware("http")
async def require_token(request: Request, call_next):
    if request.method == "POST" and request.url.path in ("/v1/identify", "/identify"):
        if not token_valid(request.headers.get("authorization", "").removeprefix("Bearer ")):
            return JSONResponse({"detail": "missing or invalid token"}, status_code=401,
                                headers={"WWW-Authenticate": "Bearer"})
    return await call_next(request)


if CORS_ORIGIN_REGEX:
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=CORS_ORIGIN_REGEX,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        max_age=600,
    )


def _identify(raw: bytes, min_confidence: float):
    samples, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    return lid.identify(samples, sample_rate=sample_rate, min_confidence=min_confidence)


@app.post("/v1/identify")
@app.post("/identify")
async def identify(
    audio: UploadFile = File(...),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    top_k: int = Query(9, ge=1, le=9),
):
    raw = await audio.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    try:
        result = await asyncio.to_thread(_identify, raw, min_confidence)
    except AudioTooShort as e:
        raise HTTPException(422, str(e))
    except (sf.LibsndfileError, ValueError, RuntimeError):
        raise HTTPException(400, "could not decode audio")
    except Exception:
        log.exception("identify failed")
        raise HTTPException(500, "internal error")

    return {
        "language": result.language,
        "confidence": round(result.confidence, 6),
        "duration": round(result.duration, 3),
        "top_k": [
            {"language": lang, "probability": round(prob, 6)}
            for lang, prob in result.top_k[:top_k]
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "languages": lid.languages}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, access_log=False, timeout_graceful_shutdown=30)
