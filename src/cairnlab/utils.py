from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


def safe_id_filename(object_id: str) -> str:
    safe = []
    for char in object_id:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe) + ".yaml"


def stable_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def utc_event_id(prefix: str, target: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    digest = hashlib.sha1(target.encode("utf-8")).hexdigest()[:8]
    return f"event:{prefix}:{timestamp}:{digest}"


def actor_from_string(value: str):
    from .models import Actor

    return Actor(id=value, role="unknown")


def enum_value(value: Any) -> Any:
    return getattr(value, "value", value)
