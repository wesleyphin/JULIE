#!/bin/bash
# Uninstall the Julie AI-loop daily-maintenance schedule.
set -euo pipefail

PLIST_NAME="com.julie.ai_loop"
DEST="${HOME}/Library/LaunchAgents/${PLIST_NAME}.plist"

if [[ -f "$DEST" ]]; then
    launchctl unload "$DEST" 2>/dev/null || true
    rm -f "$DEST"
    echo "[uninstall] removed $DEST"
else
    echo "[uninstall] no plist found at $DEST (already uninstalled?)"
fi

if launchctl list | grep -q "${PLIST_NAME}"; then
    echo "[uninstall] ⚠ still registered. Try: launchctl bootout gui/$(id -u)/${PLIST_NAME}"
else
    echo "[uninstall] ✅ no longer registered"
fi
