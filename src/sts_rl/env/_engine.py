"""Single import point for the compiled Slay the Spire engine module.

The engine (originally ``daniel-ziegler/sts_lightspeed``, vendored via a
build-portability fork; see the README) is a C++ project vendored as a git
submodule and built out of tree; its Python binding compiles to a ``slaythespire``
shared module under ``engine/sts_lightspeed/build/``. That directory is not on
``sys.path`` by default, so every part of the environment adapter imports the
engine through this module rather than importing ``slaythespire`` directly::

    from sts_rl.env._engine import slaythespire as sts

If the engine has not been built, importing this module raises
:class:`EngineNotBuiltError` with the command to build it. Tests that need the
engine use ``pytest.importorskip("sts_rl.env._engine")`` so suites stay green on
machines without the native build (for example CI runners).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

# Repo layout: this file is src/sts_rl/env/_engine.py; the engine build dir is
# <repo>/engine/sts_lightspeed/build relative to it.
_REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_BUILD_DIR = _REPO_ROOT / "engine" / "sts_lightspeed" / "build"

_MODULE_NAME = "slaythespire"
_BUILD_COMMAND = "scripts/build_engine.sh"


class EngineNotBuiltError(ImportError):
    """Raised when the compiled ``slaythespire`` module cannot be located."""


def _load() -> ModuleType:
    try:
        if ENGINE_BUILD_DIR.is_dir():
            build_dir = str(ENGINE_BUILD_DIR)
            # Append rather than prepend: the engine module has a unique name, so
            # we need only make it importable, not shadow any earlier path entry.
            if build_dir not in sys.path:
                sys.path.append(build_dir)
        return importlib.import_module(_MODULE_NAME)
    except ImportError as exc:
        raise EngineNotBuiltError(
            f"could not import the {_MODULE_NAME!r} engine module "
            f"(looked in {ENGINE_BUILD_DIR}). Build it with: {_BUILD_COMMAND}"
        ) from exc


slaythespire: ModuleType = _load()

__all__ = ["slaythespire", "ENGINE_BUILD_DIR", "EngineNotBuiltError"]
