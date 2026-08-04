from __future__ import annotations

from insight_agent_core import config as _config


for _name in dir(_config):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_config, _name)

del _config
del _name
