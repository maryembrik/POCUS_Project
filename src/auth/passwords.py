"""Password hashing.

scrypt from the standard library, so no dependency is added for the one thing in this project
that must not be got wrong. The parameters below are the interactive-login profile from RFC
7914: n=2**14 costs roughly 100 ms and 16 MB per verification here, which is negligible for a
clinician signing in once and expensive for someone working through a stolen database.

What is stored is a single self-describing string:

    scrypt$16384$8$1$<salt hex>$<hash hex>

The parameters travel WITH the hash rather than being read from this module at verification
time. If n is raised later, existing passwords still verify against the value they were made
with, instead of every account silently failing on the next deploy.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

N = 1 << 14
R = 8
P = 1
SALT_BYTES = 16
KEY_LEN = 32


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("refusing to hash an empty password")
    salt = secrets.token_bytes(SALT_BYTES)
    key = hashlib.scrypt(password.encode("utf8"), salt=salt, n=N, r=R, p=P,
                         dklen=KEY_LEN, maxmem=64 * 1024 * 1024)
    return f"scrypt${N}${R}${P}${salt.hex()}${key.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time, and false rather than raising on anything malformed.

    A verification that throws on a corrupt row would turn one bad record into a 500 that
    distinguishes it from a wrong password -- which tells an attacker something about the
    account. Every failure path returns the same False.
    """
    try:
        scheme, n, r, p, salt_hex, key_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        key = hashlib.scrypt(password.encode("utf8"), salt=bytes.fromhex(salt_hex),
                             n=int(n), r=int(r), p=int(p),
                             dklen=len(key_hex) // 2, maxmem=64 * 1024 * 1024)
    except (ValueError, TypeError):
        return False
    # compare_digest, not ==, so the time taken does not depend on how many leading bytes
    # matched. Ordinary comparison of two hex strings leaks that through timing.
    return hmac.compare_digest(key.hex(), key_hex)
