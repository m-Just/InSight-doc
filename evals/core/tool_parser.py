from __future__ import annotations

from insight_agent_core import tool_parser as _tool_parser


for _name in dir(_tool_parser):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_tool_parser, _name)

del _tool_parser
del _name
