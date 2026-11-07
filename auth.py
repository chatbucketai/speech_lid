"""Bearer token check, identical to the CB_STT_<language> services.

Accepted tokens, each with an exp in the future (anything else -> 401):
- STT token: HS256, aud "stt", signed with S2T_TOKEN_KEY (issued by
  cb-backend-nest) or S2T_TOKEN_KEY_B2B (signed by the B2B servers).
- ChatBucket app login token: RS256, verified with cb-backend-nest's public
  key (S2T_APP_TOKEN_PUBLIC_KEY). Only the login access token (userId +
  deviceId) - not refresh, pre-OTP or admin tokens.
Always valid when no key is configured (local use).
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import time

log = logging.getLogger(__name__)

TOKEN_KEY = os.environ.get("S2T_TOKEN_KEY", "")
TOKEN_KEY_B2B = os.environ.get("S2T_TOKEN_KEY_B2B", "")

_SHA256_DIGEST_INFO = bytes.fromhex("3031300d060960864801650304020105000420")
_RSA_OID = bytes.fromhex("2a864886f70d010101")
_NOT_LOGIN_CLAIMS = ("type", "loginAttemptRecordId", "isAdmin")


def _b64decode(part: str) -> bytes:
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


def _der(buf: bytes, i: int) -> tuple[int, bytes, int]:
    tag, length = buf[i], buf[i + 1]
    i += 2
    if length & 0x80:
        size = length & 0x7F
        length = int.from_bytes(buf[i:i + size], "big")
        i += size
    return tag, buf[i:i + length], i + length


def _rsa_public_key(pem: str) -> tuple[int, int] | None:
    if not pem:
        return None
    pem = pem.replace("\\n", "\n")
    der = base64.b64decode("".join(line for line in pem.strip().splitlines() if not line.startswith("-----")))
    _, spki, _ = _der(der, 0)
    _, algorithm, i = _der(spki, 0)
    if _RSA_OID not in algorithm:
        raise ValueError("not an RSA public key")
    _, bits, _ = _der(spki, i)
    _, rsa_key, _ = _der(bits[1:], 0)
    _, n, j = _der(rsa_key, 0)
    _, e, _ = _der(rsa_key, j)
    return int.from_bytes(n, "big"), int.from_bytes(e, "big")


try:
    _APP_KEY = _rsa_public_key(os.environ.get("S2T_APP_TOKEN_PUBLIC_KEY", ""))
except Exception:
    log.exception("S2T_APP_TOKEN_PUBLIC_KEY is not a valid RSA public key - app login tokens are refused")
    _APP_KEY = None


def _rs256_valid(signed: bytes, signature: bytes, key: tuple[int, int]) -> bool:
    n, e = key
    k = (n.bit_length() + 7) // 8
    s = int.from_bytes(signature, "big")
    if len(signature) != k or s >= n:
        return False
    digest_info = _SHA256_DIGEST_INFO + hashlib.sha256(signed).digest()
    expected = b"\x00\x01" + b"\xff" * (k - len(digest_info) - 3) + b"\x00" + digest_info
    return hmac.compare_digest(pow(s, e, n).to_bytes(k, "big"), expected)


def token_valid(token: str | None) -> bool:
    keys = [k for k in (TOKEN_KEY, TOKEN_KEY_B2B) if k]
    if not keys and not _APP_KEY:
        return True
    try:
        header, payload, signature = token.split(".")
        signed = f"{header}.{payload}".encode()
        received = _b64decode(signature)
        alg = json.loads(_b64decode(header)).get("alg")
        claims = json.loads(_b64decode(payload))
        if not float(claims["exp"]) > time.time():
            return False
        if alg == "HS256":
            matches = [hmac.compare_digest(received, hmac.new(k.encode(), signed, hashlib.sha256).digest()) for k in keys]
            return any(matches) and claims.get("aud") == "stt"
        if alg == "RS256" and _APP_KEY:
            return (
                _rs256_valid(signed, received, _APP_KEY)
                and isinstance(claims.get("userId"), str) and bool(claims["userId"])
                and bool(claims.get("deviceId"))
                and not any(c in claims for c in _NOT_LOGIN_CLAIMS)
            )
        return False
    except Exception:
        return False
