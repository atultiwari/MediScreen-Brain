#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# MediScreen-Brain — one-shot bootstrap & launcher (macOS / Linux)
#
# Default behaviour (no flags): create a local .venv if missing, install
# dependencies if missing, then launch the main GUI.
#
# Flags:
#   --setup            Create venv and install deps only. Do NOT run the app.
#   --run              Skip setup, launch the app from an existing venv.
#   --reinstall        Force-reinstall dependencies into the existing venv.
#   --clean            Remove the venv (and __pycache__). Cannot combine with --run.
#   --batch            Launch batch_compare_ui.py instead of the main UI.
#   --python <bin>     Use a specific python interpreter (default: python3.11
#                      → python3.10 → python3).
#   --venv <dir>       Override venv directory (default: .venv).
#   --requirements <f> Use a different requirements file
#                      (default: requirements-macos.txt on macOS/Linux).
#   --no-cache         Pass --no-cache-dir to pip.
#   --upgrade-pip      Upgrade pip/setuptools/wheel before installing deps.
#   -h | --help        Show this help.
#
# Exit codes: 0 success, 1 user error, 2 environment problem, 3 install fail.
# ---------------------------------------------------------------------------
set -euo pipefail

# -------- Defaults ---------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR=".venv"
REQ_FILE=""        # auto-pick below if empty
PYTHON_BIN=""      # auto-pick below if empty
APP_ENTRY="Brain_Tumor_detection_ui.py"
BATCH_ENTRY="batch_compare_ui.py"

DO_SETUP=1
DO_RUN=1
DO_REINSTALL=0
DO_CLEAN=0
USE_BATCH=0
PIP_NO_CACHE=0
DO_UPGRADE_PIP=0

# -------- Logging helpers --------------------------------------------------
COLOR_BLUE="\033[1;34m"
COLOR_GREEN="\033[1;32m"
COLOR_YELLOW="\033[1;33m"
COLOR_RED="\033[1;31m"
COLOR_RESET="\033[0m"
log()  { printf "${COLOR_BLUE}[run.sh]${COLOR_RESET} %s\n" "$*"; }
ok()   { printf "${COLOR_GREEN}[run.sh]${COLOR_RESET} %s\n" "$*"; }
warn() { printf "${COLOR_YELLOW}[run.sh]${COLOR_RESET} %s\n" "$*" >&2; }
die()  { printf "${COLOR_RED}[run.sh]${COLOR_RESET} %s\n" "$*" >&2; exit "${2:-2}"; }

# -------- Help -------------------------------------------------------------
print_help() {
  sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
}

# -------- Arg parsing ------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --setup)         DO_SETUP=1; DO_RUN=0; shift ;;
    --run)           DO_SETUP=0; DO_RUN=1; shift ;;
    --reinstall)     DO_REINSTALL=1; shift ;;
    --clean)         DO_CLEAN=1; shift ;;
    --batch)         USE_BATCH=1; shift ;;
    --python)        PYTHON_BIN="${2:?--python needs a path}"; shift 2 ;;
    --venv)          VENV_DIR="${2:?--venv needs a path}"; shift 2 ;;
    --requirements)  REQ_FILE="${2:?--requirements needs a path}"; shift 2 ;;
    --no-cache)      PIP_NO_CACHE=1; shift ;;
    --upgrade-pip)   DO_UPGRADE_PIP=1; shift ;;
    -h|--help)       print_help; exit 0 ;;
    *) die "Unknown flag: $1 (use --help)" 1 ;;
  esac
done

# -------- Pick Python ------------------------------------------------------
pick_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python not found: $PYTHON_BIN"
    echo "$PYTHON_BIN"; return
  fi
  for candidate in python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"; return
    fi
  done
  die "No suitable python found. Install Python 3.10 or 3.11 (e.g. 'brew install python@3.11')."
}

# -------- Pick requirements file ------------------------------------------
pick_requirements() {
  if [[ -n "$REQ_FILE" ]]; then
    [[ -f "$REQ_FILE" ]] || die "Requirements file not found: $REQ_FILE"
    echo "$REQ_FILE"; return
  fi
  local os; os="$(uname -s)"
  if [[ "$os" == "Darwin" || "$os" == "Linux" ]]; then
    [[ -f requirements-macos.txt ]] && { echo "requirements-macos.txt"; return; }
  fi
  [[ -f requirements.txt ]] || die "No requirements file found."
  echo "requirements.txt"
}

# -------- Clean ------------------------------------------------------------
if [[ $DO_CLEAN -eq 1 ]]; then
  if [[ -d "$VENV_DIR" ]]; then
    log "Removing venv: $VENV_DIR"
    rm -rf "$VENV_DIR"
    ok "Venv removed."
  else
    warn "No venv at $VENV_DIR — nothing to clean."
  fi
  find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
  ok "__pycache__ cleared."
  # If user asked --clean and nothing else, exit. Otherwise fall through to setup/run.
  if [[ $DO_RUN -eq 0 && $DO_SETUP -eq 0 ]]; then exit 0; fi
fi

# -------- Setup ------------------------------------------------------------
if [[ $DO_SETUP -eq 1 ]]; then
  PY="$(pick_python)"
  log "Using interpreter: $PY ($("$PY" -V 2>&1))"

  if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating venv at $VENV_DIR"
    "$PY" -m venv "$VENV_DIR" || die "Failed to create venv" 2
  else
    log "Venv exists at $VENV_DIR (reusing)"
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  PIP_FLAGS=()
  [[ $PIP_NO_CACHE -eq 1 ]] && PIP_FLAGS+=("--no-cache-dir")

  if [[ $DO_UPGRADE_PIP -eq 1 ]]; then
    log "Upgrading pip/setuptools/wheel"
    python -m pip install --upgrade ${PIP_FLAGS[@]+"${PIP_FLAGS[@]}"} pip setuptools wheel || die "pip upgrade failed" 3
  fi

  REQ="$(pick_requirements)"
  log "Installing dependencies from: $REQ"

  if [[ $DO_REINSTALL -eq 1 ]]; then
    PIP_FLAGS+=("--force-reinstall")
  fi

  # Check if deps already present (cheap probe). Skip install when not reinstalling.
  if [[ $DO_REINSTALL -eq 0 ]] \
      && python -c "import PySide6, ultralytics, torch, nibabel, cv2" >/dev/null 2>&1; then
    ok "Dependencies already satisfied (skip install). Use --reinstall to force."
  else
    python -m pip install ${PIP_FLAGS[@]+"${PIP_FLAGS[@]}"} -r "$REQ" || die "Dependency install failed" 3
    ok "Dependencies installed."
  fi
fi

# -------- Run --------------------------------------------------------------
if [[ $DO_RUN -eq 1 ]]; then
  [[ -d "$VENV_DIR" ]] || die "No venv at $VENV_DIR. Run with --setup first."
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  ENTRY="$APP_ENTRY"
  [[ $USE_BATCH -eq 1 ]] && ENTRY="$BATCH_ENTRY"
  [[ -f "$ENTRY" ]] || die "Entry script missing: $ENTRY" 2

  log "Launching: python $ENTRY"
  exec python "$ENTRY"
fi
