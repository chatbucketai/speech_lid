import io
import tempfile
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from lid9 import AudioTooShort, LanguageID

app = FastAPI(title="lid9", description="Spoken language identification for 9 Indian languages")

lid = LanguageID()


@app.post("/identify")
async def identify(
    audio: UploadFile = File(...),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    top_k: int = Query(9, ge=1, le=9),
):
    suffix = Path(audio.filename or "audio.wav").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(await audio.read())
        tmp.flush()
        try:
            result = lid.identify(tmp.name, min_confidence=min_confidence)
        except AudioTooShort as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
