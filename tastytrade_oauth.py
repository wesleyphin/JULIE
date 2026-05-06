"""TastyTrade OAuth2 helper — handles authorization code flow + refresh tokens.

Run this module directly ONCE to do the initial browser authorization:

    python3 tastytrade_oauth.py

The script:
  1. Opens https://my.tastytrade.com/auth.html in your browser with the
     OAuth params (client_id, redirect_uri, response_type=code, scope=read).
  2. You sign in to TastyTrade and authorize the app.
  3. Browser redirects to https://localhost:8080/callback?code=XXX&state=YYY
  4. A local HTTPS server (self-signed cert, generated on the fly) catches it.
  5. Exchanges code for refresh_token + access_token.
  6. Saves refresh_token to TASTYTRADE_REFRESH_TOKEN_PATH (per config_secrets).

After that, programmatic use is just:

    from tastytrade_oauth import get_access_token
    token = get_access_token()  # auto-refreshes from saved refresh_token

Notes:
  - Self-signed cert means your browser will warn "not secure" — accept once.
  - access_token lifetime ~15 min; the helper caches and refreshes lazily.
  - refresh_token is long-lived but can be revoked at my.tastytrade.com →
    Manage → API → Manage OAuth Grants.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import ssl
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import requests

from config_secrets import SECRETS

CLIENT_ID = SECRETS["TASTYTRADE_CLIENT_ID"]
CLIENT_SECRET = SECRETS["TASTYTRADE_CLIENT_SECRET"]
REDIRECT_URI = SECRETS["TASTYTRADE_REDIRECT_URI"]
REFRESH_TOKEN_PATH = Path(SECRETS["TASTYTRADE_REFRESH_TOKEN_PATH"])
AUTHORIZE_URL = SECRETS["TASTYTRADE_OAUTH_AUTHORIZE"]
TOKEN_URL = SECRETS["TASTYTRADE_OAUTH_TOKEN"]

# In-process cache of {access_token, expires_at_unix}
_TOKEN_CACHE: dict = {}


# ─── one-time authorization (run interactively) ──────────────────────────────

class _CallbackHandler(BaseHTTPRequestHandler):
    """Tiny HTTPS handler that captures the OAuth ?code=... redirect."""

    captured: dict = {}

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        if "code" in params:
            _CallbackHandler.captured["code"] = params["code"][0]
            _CallbackHandler.captured["state"] = params.get("state", [""])[0]
            body = b"<h2>OAuth callback received. You can close this tab.</h2>"
            self.send_response(200)
        elif "error" in params:
            _CallbackHandler.captured["error"] = params["error"][0]
            body = f"<h2>OAuth error: {params['error'][0]}</h2>".encode()
            self.send_response(400)
        else:
            body = b"<h2>Waiting for OAuth callback...</h2>"
            self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kwargs):
        return  # silence the default access log


def _generate_self_signed_cert() -> tuple[Path, Path]:
    """Generate a one-shot self-signed cert for localhost HTTPS callback.
    Returns (cert_path, key_path)."""
    import datetime as _dt
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    tmp_dir = Path("/tmp")
    cert_path = tmp_dir / "tt_oauth_cert.pem"
    key_path = tmp_dir / "tt_oauth_key.pem"

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = _dt.datetime.utcnow()
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now).not_valid_after(now + _dt.timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]),
                       critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    return cert_path, key_path


def authorize_interactive(scope: str = "read", port: int = 8080) -> str:
    """Run the one-time browser authorization flow and return the refresh_token.

    Side effects:
      - Opens browser
      - Runs local HTTPS server on localhost:{port} until callback received
      - Writes refresh_token to REFRESH_TOKEN_PATH
    """
    state = secrets.token_urlsafe(16)
    auth_params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": scope,
        "state": state,
    }
    auth_url = f"{AUTHORIZE_URL}?{urllib.parse.urlencode(auth_params)}"

    print(f"\n[tastytrade-oauth] Opening browser to authorize:")
    print(f"  {auth_url}")
    print(f"\n[tastytrade-oauth] Browser may warn 'not secure' for localhost — accept once.")
    print(f"[tastytrade-oauth] Waiting for callback on https://localhost:{port}/callback ...\n")

    cert_path, key_path = _generate_self_signed_cert()
    server = HTTPServer(("localhost", port), _CallbackHandler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    server.socket = ctx.wrap_socket(server.socket, server_side=True)

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    webbrowser.open(auth_url)

    # Wait up to 5 minutes for the callback
    deadline = time.time() + 300
    while time.time() < deadline:
        if "code" in _CallbackHandler.captured or "error" in _CallbackHandler.captured:
            break
        time.sleep(0.2)
    server.shutdown()

    if "error" in _CallbackHandler.captured:
        raise RuntimeError(f"OAuth error: {_CallbackHandler.captured['error']}")
    if "code" not in _CallbackHandler.captured:
        raise TimeoutError("OAuth callback not received within 5 minutes")
    if _CallbackHandler.captured.get("state") != state:
        raise RuntimeError("OAuth state mismatch — possible CSRF; aborting")

    code = _CallbackHandler.captured["code"]
    print(f"[tastytrade-oauth] Got authorization code; exchanging for tokens...")

    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Token exchange failed: {resp.status_code} {resp.text}")
    tokens = resp.json()

    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise RuntimeError(f"No refresh_token in response: {tokens}")

    REFRESH_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFRESH_TOKEN_PATH.write_text(refresh_token)
    REFRESH_TOKEN_PATH.chmod(0o600)
    print(f"[tastytrade-oauth] ✅ refresh_token saved to {REFRESH_TOKEN_PATH}")

    # Cache the access_token for immediate use
    _TOKEN_CACHE["access_token"] = tokens["access_token"]
    _TOKEN_CACHE["expires_at"] = time.time() + int(tokens.get("expires_in", 900)) - 30
    print(f"[tastytrade-oauth] ✅ access_token cached (expires in {tokens.get('expires_in', 900)}s)")
    return refresh_token


# ─── programmatic access (called by other modules) ──────────────────────────

def _refresh_access_token() -> str:
    """Use the saved refresh_token to mint a new access_token."""
    if not REFRESH_TOKEN_PATH.exists():
        raise RuntimeError(
            f"No refresh_token at {REFRESH_TOKEN_PATH}. "
            f"Run `python3 tastytrade_oauth.py` once to authorize."
        )
    refresh_token = REFRESH_TOKEN_PATH.read_text().strip()
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Refresh failed: {resp.status_code} {resp.text}. "
            f"You may need to re-run `python3 tastytrade_oauth.py` to re-authorize."
        )
    tokens = resp.json()
    # Some OAuth providers rotate the refresh_token on each use; persist if so
    new_refresh = tokens.get("refresh_token")
    if new_refresh and new_refresh != refresh_token:
        REFRESH_TOKEN_PATH.write_text(new_refresh)
    _TOKEN_CACHE["access_token"] = tokens["access_token"]
    _TOKEN_CACHE["expires_at"] = time.time() + int(tokens.get("expires_in", 900)) - 30
    logging.info("TastyTrade access_token refreshed (expires_in=%ss)",
                 tokens.get("expires_in"))
    return _TOKEN_CACHE["access_token"]


def get_access_token() -> str:
    """Return a valid access_token, refreshing if needed.
    Safe to call from hot paths — caches in-process; refreshes only when
    within 30s of expiry."""
    now = time.time()
    if (_TOKEN_CACHE.get("access_token")
            and _TOKEN_CACHE.get("expires_at", 0) > now):
        return _TOKEN_CACHE["access_token"]
    return _refresh_access_token()


def auth_headers() -> dict:
    """Convenience: return {'Authorization': 'Bearer <token>'} for requests."""
    return {"Authorization": f"Bearer {get_access_token()}"}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Testing access token...")
        print(f"access_token: {get_access_token()[:20]}...")
        print("OK")
    else:
        print("[tastytrade-oauth] Running one-time authorization flow.")
        print("[tastytrade-oauth] Make sure you're logged into TastyTrade in your browser first.")
        try:
            authorize_interactive(scope="read")
        except Exception as exc:
            print(f"[tastytrade-oauth] ❌ {exc}")
            sys.exit(1)
        print("[tastytrade-oauth] ✅ Done. You can now use get_access_token() programmatically.")
