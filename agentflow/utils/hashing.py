from __future__ import annotations

import hashlib
import json
from typing import Any


def canonicalize_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_config_hash(normalized_config: dict[str, Any]) -> str:
    canonical_json = canonicalize_json(normalized_config)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
