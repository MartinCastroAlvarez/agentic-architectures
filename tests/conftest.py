"""Shared helpers: a runtime on a scripted fake model with the standard tools."""

from __future__ import annotations

import json
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from agentic_architectures import build_runtime
from agentic_architectures.core.budget import Budget
from agentic_architectures.core.fake import ScriptedModel
from agentic_architectures.core.runtime import Runtime


def j(**data: Any) -> str:
    return json.dumps(data)


MakeRt = Callable[..., tuple[Runtime, ScriptedModel]]


@pytest.fixture
def make_rt(tmp_path: Path) -> MakeRt:
    def make(
        script: Mapping[str, Any] | None = None,
        *,
        handler: Any = None,
        budget: Budget | None = None,
        latency: float = 0.0,
    ) -> tuple[Runtime, ScriptedModel]:
        model = ScriptedModel(script, handler=handler, latency=latency)
        return build_runtime(model, workspace=tmp_path / "space", budget=budget), model

    return make
