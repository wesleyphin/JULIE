import os


# Copy this file to config_secrets.py for local use only.
# Do not commit real secrets to GitHub.
SECRETS = {
    "USERNAME": os.environ.get("TOPSTEPX_USERNAME", ""),
    "API_KEY": os.environ.get("TOPSTEPX_API_KEY", ""),
    "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
    "KALSHI_KEY_ID": os.environ.get("KALSHI_KEY_ID", "your-key-id-here"),
    "KALSHI_PRIVATE_KEY_PATH": os.environ.get("KALSHI_PRIVATE_KEY_PATH", "path/to/your/kalshi_key.pem"),
}
