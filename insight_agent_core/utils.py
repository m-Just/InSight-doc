from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    """Convert common non-JSON values into stable JSON-serializable values."""

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    module_name = type(value).__module__.split(".", 1)[0]
    if module_name == "numpy":
        if hasattr(value, "tolist"):
            return json_safe(value.tolist())
        if hasattr(value, "item"):
            return value.item()

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
