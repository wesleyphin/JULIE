#!/usr/bin/env python3
"""Quick diagnostic to test Truth Social API connectivity via truthbrush.

Run from the project root with your julie_bot_11 conda env active:
    python tools/diagnose_truth_social.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from config_secrets import SECRETS as CONFIG_SECRETS
except Exception:
    CONFIG_SECRETS = {}


def _secret(key: str) -> str:
    return str(CONFIG_SECRETS.get(key, "") or os.environ.get(key, "") or "").strip()


def main() -> int:
    token = _secret("TRUTHSOCIAL_TOKEN")
    username = _secret("TRUTHSOCIAL_USERNAME")
    password = _secret("TRUTHSOCIAL_PASSWORD")

    print("=== Truth Social API Diagnostic ===\n")

    if token:
        print(f"  Token:    {'*' * 8}...{token[-6:]}")
    else:
        print("  Token:    (not set)")
    print(f"  Username: {username or '(not set)'}")
    print(f"  Password: {'***' if password else '(not set)'}")
    print()

    if not token and (not username or not password):
        print("ERROR: No credentials found. Add TRUTHSOCIAL_TOKEN or USERNAME+PASSWORD to config_secrets.py")
        return 1

    try:
        from truthbrush import Api
    except ImportError:
        print("ERROR: truthbrush is not installed. Run: pip install truthbrush")
        return 1

    api = Api(username=username, password=password, token=token)
    target = "realDonaldTrump"

    # --- Step 1: lookup ---
    print(f"Step 1: Looking up @{target} ...")
    try:
        user_info = api.lookup(target)
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        print("\n  This means Cloudflare is blocking the lookup endpoint.")
        print("  Your token may be expired or Cloudflare has flagged this IP.")
        print("\n  To fix:")
        print("  1. Open https://truthsocial.com in Chrome, log in")
        print("  2. Open DevTools > Application > Local Storage > https://truthsocial.com")
        print("  3. Copy the value of the 'access_token' key")
        print("  4. Paste it as TRUTHSOCIAL_TOKEN in config_secrets.py")
        return 1

    if not isinstance(user_info, dict) or "id" not in user_info:
        print(f"  FAILED: lookup returned {type(user_info).__name__} instead of user dict")
        print("  This usually means Cloudflare returned an HTML block page.")
        print("\n  Same fix as above: refresh your token from the browser.")
        return 1

    user_id = user_info["id"]
    print(f"  OK: user_id={user_id}, display_name={user_info.get('display_name', '?')}")

    # --- Step 2: pull one page of statuses ---
    print(f"\nStep 2: Fetching recent posts for @{target} ...")
    try:
        posts = list(api.pull_statuses(target, replies=False, pinned=False))
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        print("\n  Lookup worked but status fetch is blocked.")
        print("  Cloudflare may be rate-limiting after too many requests.")
        print("  Wait 15-30 minutes then try again, or refresh your token.")
        return 1

    if not posts:
        print("  OK but 0 posts returned (may be genuine or a silent block).")
    else:
        print(f"  OK: got {len(posts)} posts")
        latest = posts[0]
        print(f"  Latest: id={latest.get('id')}, created={latest.get('created_at')}")
        content = str(latest.get("content", ""))[:120]
        print(f"  Preview: {content}...")

    print("\n=== Diagnosis: Truth Social API is reachable! ===")
    print("The bot should work. If you still see Cloudflare errors,")
    print("increase poll_interval in config.py (currently set to 120s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
