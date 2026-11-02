#!/usr/bin/env python3
# Copyright    2026  Pramiteeh
#
# Importable inference API for the 9-language LID model built by
# export-onnx-lid9.py.
#
#   from lid9 import LanguageID
#
#   lid = LanguageID("models/lid9")            # load once, reuse
#   print(lid.identify("sample.wav").language) # -> "hindi"
#
# Dependencies: onnxruntime, numpy. soundfile only if you pass file paths;
# passing a numpy array needs neither it nor any audio library.
#
# Feature extraction lives inside the ONNX graph, so there is nothing to keep
# in sync here — the model consumes a raw 16 kHz mono waveform.
#
# Thread safety: onnxruntime InferenceSession.run() is thread-safe, so one
# LanguageID instance can serve concurrent requests. Construct it once at
# process start; constructing per request will dominate your latency.

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "onnxruntime is required: pip install onnxruntime"
    ) from e

__all__ = ["LanguageID", "LidResult", "AudioTooShort", "ModelNotFound"]

MODEL_FILENAME = "lid.onnx"

# The STFT needs a full 25 ms analysis frame and the res2net stack needs a few
# frames of context before its output means anything.
MIN_SECONDS = 0.3

AudioInput = Union[str, os.PathLike, np.ndarray]

# Models ship inside the package, so the default works regardless of CWD.
_BUNDLED_MODEL_DIR = str(Path(__file__).parent / "models")


class ModelNotFound(FileNotFoundError):
    """Raised when no lid.onnx exists at the requested path."""


class AudioTooShort(ValueError):
    """Raised when a clip is below MIN_SECONDS of audio."""


@dataclass(frozen=True)
class LidResult:
    """Outcome of one identification.

    Attributes:
        language:   best-scoring language name, or None if `min_confidence`
                    was set and nothing cleared it
        confidence: probability of `language` (or of the top candidate when
                    `language` is None because the threshold was not met)
        scores:     every language mapped to its probability, descending
        duration:   seconds of audio actually fed to the model, after any
                    truncation to `max_seconds`
    """

    language: Optional[str]
    confidence: float
    scores: Dict[str, float] = field(repr=False)
    duration: float = 0.0

    @property
    def top_k(self) -> List[Tuple[str, float]]:
        return list(self.scores.items())

    def is_confident(self, threshold: float) -> bool:
        return self.confidence >= threshold


def _load_audio_file(path: Union[str, os.PathLike], target_sr: int) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "Reading audio files needs soundfile (pip install soundfile). "
            "Alternatively pass a numpy array of 16 kHz mono samples."
        ) from e

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = data.mean(axis=1)  # downmix
    return _resample(audio, sr, target_sr)


def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Linear resampling — adequate for LID, and keeps the dependency set small.

    If your inputs are routinely off-rate and accuracy matters, resample
    upstream with ffmpeg/sox and hand this module 16 kHz audio directly.
    """
    if sr == target_sr:
        return audio.astype(np.float32, copy=False)
    n_out = int(round(audio.shape[0] / sr * target_sr))
    if n_out <= 1:
        return np.zeros(0, dtype=np.float32)
    idx = np.linspace(0, audio.shape[0] - 1, n_out, dtype=np.float64)
    return np.interp(idx, np.arange(audio.shape[0]), audio).astype(np.float32)


class LanguageID:
    """9-language spoken language identifier backed by an ONNX model.

    Args:
        model: directory containing lid.onnx, or a path to the .onnx itself
        threads: onnxruntime intra-op threads
        max_seconds: truncate longer audio. LID accuracy saturates within a few
            seconds and attentive pooling is linear in length, so this bounds
            latency without costing accuracy. None uses the whole clip.
        providers: onnxruntime execution providers; defaults to CPU.
    """

    def __init__(
        self,
        model: Union[str, os.PathLike] = _BUNDLED_MODEL_DIR,
        threads: int = 4,
        max_seconds: Optional[float] = 10.0,
        providers: Optional[Sequence[str]] = None,
    ):
        path = Path(model)
        if path.is_dir():
            path = path / MODEL_FILENAME
        if not path.exists():
            raise ModelNotFound(
                f"No model at {path}. Build it with:\n"
                f"  python export-onnx-lid9.py --output-dir {Path(model)}"
            )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(path),
            sess_options=opts,
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )

        meta = self._session.get_modelmeta().custom_metadata_map
        # The other LID model in this repo (export-onnx-lid.py, built from the
        # Zipformer checkpoint) takes fbank features rather than a waveform, so
        # loading it here would fail obscurely at the first run() call.
        if meta.get("model_type") == "zipformer2-lid":
            raise ValueError(
                f"{path} is the Zipformer LID model, which takes fbank features "
                f"rather than a raw waveform. Run it with lid_infer.py, or pass "
                f"a model built by export-onnx-lid9.py."
            )
        if "languages" not in meta:
            raise ValueError(
                f"{path} has no 'languages' metadata — it was not produced by "
                f"export-onnx-lid9.py. Re-export it."
            )
        self.languages: List[str] = meta["languages"].split(",")
        self.sample_rate: int = int(meta.get("sample_rate", 16000))
        self.model_path: Path = path
        self.max_seconds = max_seconds

    # -- public API ------------------------------------------------------

    def identify(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
        min_confidence: float = 0.0,
    ) -> LidResult:
        """Identify the language of one clip.

        Args:
            audio: path to an audio file, or a 1-D numpy array of samples.
                A 2-D array is downmixed across its last axis.
            sample_rate: required when `audio` is an array recorded at
                something other than the model's rate; ignored for file paths.
            min_confidence: if the top probability is below this, `language`
                comes back None rather than a guess. Use it when "unknown" is a
                better answer than a wrong one.

        Raises:
            AudioTooShort: clip is under 0.3 s.
        """
        samples = self._to_samples(audio, sample_rate)
        probs = self._run(samples)

        order = np.argsort(-probs)
        scores = {self.languages[i]: float(probs[i]) for i in order}
        best = self.languages[order[0]]
        confidence = float(probs[order[0]])

        return LidResult(
            language=best if confidence >= min_confidence else None,
            confidence=confidence,
            scores=scores,
            duration=len(samples) / self.sample_rate,
        )

    def identify_batch(
        self,
        items: Iterable[AudioInput],
        sample_rate: Optional[int] = None,
        min_confidence: float = 0.0,
        on_error: str = "raise",
    ) -> List[Optional[LidResult]]:
        """Identify several clips.

        The exported graph has batch size 1, so this loops rather than batching
        on the accelerator — it exists for convenience and error handling, not
        throughput. For real throughput, run several LanguageID calls across
        threads; ORT sessions are thread-safe.

        Args:
            on_error: "raise" propagates, "skip" puts None in that slot.
        """
        if on_error not in ("raise", "skip"):
            raise ValueError("on_error must be 'raise' or 'skip'")

        out: List[Optional[LidResult]] = []
        for item in items:
            try:
                out.append(self.identify(item, sample_rate, min_confidence))
            except Exception:
                if on_error == "raise":
                    raise
                out.append(None)
        return out

    # -- internals -------------------------------------------------------

    def _to_samples(
        self, audio: AudioInput, sample_rate: Optional[int]
    ) -> np.ndarray:
        if isinstance(audio, (str, os.PathLike)):
            samples = _load_audio_file(audio, self.sample_rate)
        elif isinstance(audio, np.ndarray):
            samples = audio
            if samples.ndim == 2:
                samples = samples.mean(axis=-1)
            elif samples.ndim != 1:
                raise ValueError(f"expected 1-D or 2-D audio, got {samples.ndim}-D")
            samples = samples.astype(np.float32, copy=False)
            if sample_rate is not None and sample_rate != self.sample_rate:
                samples = _resample(samples, sample_rate, self.sample_rate)
        else:
            raise TypeError(
                f"audio must be a path or numpy array, got {type(audio).__name__}"
            )

        if self.max_seconds:
            samples = samples[: int(self.max_seconds * self.sample_rate)]

        if samples.shape[0] < int(MIN_SECONDS * self.sample_rate):
            raise AudioTooShort(
                f"{samples.shape[0] / self.sample_rate:.2f}s of audio; "
                f"need at least {MIN_SECONDS}s"
            )
        return samples

    def _run(self, samples: np.ndarray) -> np.ndarray:
        wav = np.ascontiguousarray(samples[None, :], dtype=np.float32)
        log_probs, _ = self._session.run(
            ["log_probs", "language_idx"], {"wav": wav}
        )
        return np.exp(log_probs[0])

    def __repr__(self) -> str:
        return f"LanguageID({self.model_path.name}, languages={len(self.languages)})"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python lid9.py <audio> [audio ...]", file=sys.stderr)
        raise SystemExit(2)

    lid = LanguageID()
    for path in sys.argv[1:]:
        r = lid.identify(path)
        print(f"{path}: {r.language} ({r.confidence:.3f})")
