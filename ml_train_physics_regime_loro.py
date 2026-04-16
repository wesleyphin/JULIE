from __future__ import annotations

from typing import List

import ml_train_physics as base_train


def main(argv: List[str] | None = None) -> int:
    # Legacy LORO entrypoint now shares the same dist_bracket_ml pipeline.
    # We preserve argv passthrough so existing automation still works.
    return int(base_train.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
