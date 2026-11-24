"""The container healthcheck: readiness, which means the database answered."""

import sys
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:8000/health/ready", timeout=4) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
