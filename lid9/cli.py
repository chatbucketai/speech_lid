#!/usr/bin/env python3
# Copyright    2026  Pramiteeh
#
# CLI for the 9-language LID model. Audio file in, language name out.
#
#   $ ./lid9_infer.py --audio sample.wav
#   hindi
#
# All the actual inference lives in lid9.py; this is only argument parsing and
# output formatting, so the library and the CLI cannot drift apart.

import argparse
import json
import sys
from pathlib import Path

from lid9 import AudioTooShort, LanguageID, ModelNotFound


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Identify the spoken language of an audio file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--audio", type=Path, required=True, nargs="+",
                   help="One or more audio files.")
    # Default deliberately None rather than "models/lid9": when installed as a
    # package the model ships inside it, and LanguageID's own default resolves
    # to the bundled copy. Hardcoding a relative path here would break the
    # console script from any directory but the repo root.
    p.add_argument("--model", type=Path, default=None,
                   help="Directory with lid.onnx, or a path to the .onnx file. "
                        "Defaults to the bundled model.")
    p.add_argument("--max-seconds", type=float, default=10.0,
                   help="Truncate audio before classifying. 0 uses the whole file.")
    p.add_argument("--min-confidence", type=float, default=0.0,
                   help="Below this, report 'unknown' instead of guessing.")
    p.add_argument("--top-k", type=int, default=1,
                   help="With 1 (default) print only the language name.")
    p.add_argument("--json", action="store_true",
                   help="One JSON object per file.")
    p.add_argument("--threads", type=int, default=4)
    return p


def main() -> None:
    args = get_parser().parse_args()

    try:
        kwargs = {} if args.model is None else {"model": args.model}
        lid = LanguageID(
            **kwargs,
            threads=args.threads,
            max_seconds=args.max_seconds if args.max_seconds > 0 else None,
        )
    except (ModelNotFound, ValueError) as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)

    exit_code = 0
    show_prefix = len(args.audio) > 1

    for path in args.audio:
        try:
            r = lid.identify(path, min_confidence=args.min_confidence)
        except (AudioTooShort, FileNotFoundError, RuntimeError) as e:
            print(f"{path}: {e}", file=sys.stderr)
            exit_code = 1
            continue

        label = r.language or "unknown"

        if args.json:
            print(json.dumps({
                "file": str(path),
                "language": r.language,
                "confidence": round(r.confidence, 6),
                "duration": round(r.duration, 3),
                "top_k": [{"language": l, "probability": round(p, 6)}
                          for l, p in r.top_k[: max(args.top_k, 1)]],
            }))
        elif args.top_k <= 1:
            print(f"{path}: {label}" if show_prefix else label)
        else:
            body = "  ".join(f"{l} {p:.4f}" for l, p in r.top_k[: args.top_k])
            print(f"{path}: {body}" if show_prefix else body)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
