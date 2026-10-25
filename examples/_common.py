"""Shared setup for the examples: the offline toy model unless you ask for a real one.

python -m examples.react                          # offline, scripted toy model
AGENTIC_PROVIDER=ollama python -m examples.react  # your local ollama (qwen3:4b)
"""

from __future__ import annotations

import os
from typing import Any

from agentic_architectures import build_runtime
from agentic_architectures.core.providers import model_from_env
from agentic_architectures.core.runtime import Result
from agentic_architectures.core.runtime import Runtime
from agentic_architectures.toy import TASKS
from agentic_architectures.toy import ToyModel

SPEED, PORT, IDEMPOTENCY = (t.question for t in TASKS)


def runtime(**kwargs: Any) -> Runtime:
    model = model_from_env() if os.environ.get("AGENTIC_PROVIDER") else ToyModel()
    return build_runtime(model, echo=True, **kwargs)


def show(result: Result) -> Result:
    print(f"\n{result.answer}\n{result.summary()}\n")
    return result
