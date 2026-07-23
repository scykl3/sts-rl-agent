#!/usr/bin/env bash
#
# Build the Slay the Spire simulation engine's Python module (`slaythespire`).
#
# The engine is a C++ project vendored as a git submodule under
# engine/sts_lightspeed. This script initializes the two nested dependencies it
# needs, configures CMake, and builds only the Python binding target. The
# resulting shared module is loaded by src/sts_rl/env/_engine.py.
#
# Usage:
#   scripts/build_engine.sh
#
# Environment overrides:
#   PYTHON  Python interpreter the module is built for (default: .venv/bin/python).
#   CMAKE   CMake binary to use (default: .venv/bin/cmake, else `cmake` on PATH).
#   JOBS    Parallel build jobs (default: 8).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_DIR="$REPO_ROOT/engine/sts_lightspeed"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
JOBS="${JOBS:-8}"

if [[ -n "${CMAKE:-}" ]]; then
    :
elif [[ -x "$REPO_ROOT/.venv/bin/cmake" ]]; then
    CMAKE="$REPO_ROOT/.venv/bin/cmake"
else
    CMAKE="cmake"
fi

if [[ ! -d "$ENGINE_DIR" ]]; then
    echo "error: engine submodule missing at $ENGINE_DIR" >&2
    echo "run: git submodule update --init engine/sts_lightspeed" >&2
    exit 1
fi

# Nested dependencies the engine builds against: nlohmann/json and pybind11.
# CommunicationMod (the live-game bridge) is deliberately not initialized.
echo ">> initializing nested engine dependencies (json, pybind11)"
git -C "$ENGINE_DIR" submodule update --init json pybind11

# Configure. CMAKE_POLICY_VERSION_MINIMUM=3.5 lets the vendored nlohmann/json,
# which declares a very old cmake_minimum_required, configure under CMake >= 4.
#
# Pin the interpreter with every spelling CMake might consult: the legacy
# -DPYTHON_EXECUTABLE (old FindPythonInterp) plus the modern
# -DPython_EXECUTABLE / -DPython3_EXECUTABLE that current FindPython and pybind11
# actually honor. Passing only the legacy name lets a newer CMake fall back to
# autodetection and build against the wrong interpreter, producing an import or
# ABI mismatch when the module is loaded from "$PYTHON".
echo ">> configuring"
"$CMAKE" -B "$ENGINE_DIR/build" -S "$ENGINE_DIR" \
    -DPYTHON_EXECUTABLE="$PYTHON" \
    -DPython_EXECUTABLE="$PYTHON" \
    -DPython3_EXECUTABLE="$PYTHON" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5

# Build only the Python module target (not the console sim or benchmarks).
echo ">> building slaythespire module"
"$CMAKE" --build "$ENGINE_DIR/build" --target slaythespire -j"$JOBS"

echo ">> done; module at:"
ls "$ENGINE_DIR"/build/slaythespire*.so
