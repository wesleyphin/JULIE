import os


# Copy this file to config_secrets.py for local use only.
# Do not commit real secrets to GitHub.
SECRETS = {
    "USERNAME": os.environ.get("TOPSTEPX_USERNAME", ""),
    "API_KEY": os.environ.get("TOPSTEPX_API_KEY", ""),
    "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
}
