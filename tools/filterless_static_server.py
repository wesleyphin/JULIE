from __future__ import annotations

import argparse
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "montecarlo" / "Backtest-Simulator-main" / "dist"
TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


class QuietStaticHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the built filterless Monte Carlo app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not DIST_DIR.exists():
        raise SystemExit(f"Built app directory does not exist: {DIST_DIR}")

    handler = partial(QuietStaticHandler, directory=str(DIST_DIR))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    verbose = str(os.environ.get("FILTERLESS_STATIC_SERVER_VERBOSE", "")).strip().lower()
    if verbose in TRUTHY_ENV_VALUES:
        print(f"Serving filterless dist at http://{args.host}:{args.port}/ from {DIST_DIR}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
