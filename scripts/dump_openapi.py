"""Write the OpenAPI schema to stdout without starting a server.

`openapi-typescript` can read a URL, but that would make type generation depend
on a running API and a reachable database. The schema is a pure function of the
route definitions, so it is dumped straight from the app object — which is what
lets `make types` and the CI drift check run with nothing else up.

Keys are sorted so the output is byte-stable. The drift check diffs generated
types against committed ones, and a dictionary that reordered between runs would
make that check fail for no reason and teach everyone to ignore it.
"""

import json
import sys

from memoryos.api.app import create_app
from memoryos.config import Settings


def main() -> int:
    # No database is touched: `create_app` builds the container inside its
    # lifespan, and nothing here runs that.
    app = create_app(Settings())
    json.dump(app.openapi(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
