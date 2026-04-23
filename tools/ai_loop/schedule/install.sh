#!/bin/bash
# Install the Julie AI-loop daily-maintenance schedule.
#
# What this does:
#   - Copies com.julie.ai_loop.plist to ~/Library/LaunchAgents/
#   - Loads it into launchd
#   - Verifies it's registered
#
# What you get:
#   Every weekday at 14:30 PT (17:30 ET = 30 min into CME maintenance),
#   launchd runs:  python3 -m tools.ai_loop.run_daily
#   Output goes to ai_loop_data/run_daily.log + launchd.{stdout,stderr}.log
#
# To stop:  bash tools/ai_loop/schedule/uninstall.sh

set -euo pipefail

PLIST_NAME="com.julie.ai_loop"
SRC="$(cd "$(dirname "$0")" && pwd)/${PLIST_NAME}.plist"
DEST="${HOME}/Library/LaunchAgents/${PLIST_NAME}.plist"

if [[ ! -f "$SRC" ]]; then
    echo "ERROR: plist not found at $SRC"
    exit 1
fi

mkdir -p "${HOME}/Library/LaunchAgents"
cp -f "$SRC" "$DEST"
echo "[install] copied plist → $DEST"

# Unload if it was already loaded (idempotent)
launchctl unload "$DEST" 2>/dev/null || true

launchctl load -w "$DEST"
echo "[install] launchd loaded"

# Verify
if launchctl list | grep -q "${PLIST_NAME}"; then
    echo "[install] ✅ registered. launchctl list | grep julie:"
    launchctl list | grep julie
    echo ""
    echo "Next scheduled run: weekday 14:30 local time (17:30 ET)."
    echo "Observe logs:"
    echo "    tail -f $(pwd)/../../ai_loop_data/run_daily.log"
    echo ""
    echo "Dry-run it now without waiting (doesn't apply changes):"
    echo "    python3 -m tools.ai_loop.run_daily --dry-run"
    echo ""
    echo "Kill switch (halt all auto-applies):"
    echo "    export JULIE_FREEZE_AUTO_CONFIG=1"
else
    echo "[install] ❌ plist not registered, check launchd logs"
    exit 1
fi
