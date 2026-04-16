#!/usr/bin/env python3
"""Quick diagnostic to test Truth Social RSS feed connectivity.

Run from the project root with your julie_bot_11 conda env active:
    python tools/diagnose_truth_social.py
"""
from __future__ import annotations

import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime

RSS_URL = "https://trumpstruth.org/feed"
RSS_NS = {"truth": "https://truthsocial.com/ns"}
HTML_TAG_RE = re.compile(r"<[^>]+>")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_2_1) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
)


def main() -> int:
    print("=== Truth Social RSS Feed Diagnostic ===\n")

    # --- Step 1: fetch feed ---
    print(f"Step 1: Fetching RSS feed from {RSS_URL} ...")
    try:
        req = urllib.request.Request(RSS_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
            root = ET.fromstring(raw)
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        print("\n  The RSS feed is unreachable. Check your internet connection.")
        return 1

    all_items = root.findall(".//item")
    print(f"  OK: feed has {len(all_items)} items ({len(raw):,} bytes)")

    # --- Step 2: filter for today ---
    today = date.today()
    today_posts = []
    for item in all_items:
        pub_date_str = item.findtext("pubDate", "").strip()
        if not pub_date_str:
            continue
        try:
            dt = parsedate_to_datetime(pub_date_str)
        except Exception:
            continue
        if dt.date() != today:
            continue

        original_url = item.findtext("truth:originalUrl", "", RSS_NS)
        post_id = original_url.rsplit("/", 1)[-1] if original_url else "?"
        title = (item.findtext("title", "") or "").strip()
        description = (item.findtext("description", "") or "").strip()
        content = title if title and title.lower() != "[no title]" else description
        content = HTML_TAG_RE.sub(" ", content).strip()

        today_posts.append({
            "id": post_id,
            "time": dt.astimezone(timezone.utc).strftime("%H:%M:%S UTC"),
            "text": content[:140],
            "url": original_url,
        })

    print(f"\nStep 2: Found {len(today_posts)} posts from today ({today.isoformat()}):\n")
    if not today_posts:
        print("  (no posts found for today -- @realDonaldTrump may not have posted yet)")
    else:
        for i, p in enumerate(today_posts, 1):
            print(f"  {i}. [{p['time']}] id={p['id']}")
            print(f"     {p['text']}{'...' if len(p['text']) >= 140 else ''}")
            print()

    print("=== Diagnosis: RSS feed is working! ===")
    print("The bot will use this feed instead of truthbrush (no Cloudflare).")
    print("Posts are fetched from trumpstruth.org which mirrors Truth Social")
    print("with ~5 minute delay. No authentication needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
