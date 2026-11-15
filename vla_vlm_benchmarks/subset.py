from __future__ import annotations

import random


def choose_indices(n: int, limit: int | None, seed: int) -> list[int]:
    """Stable subset of ``range(n)``.

    ``limit`` of None or <= 0 keeps every index. The chosen indices are sorted
    so example order follows the dataset, and the choice depends only on
    ``n``, ``limit``, and ``seed``.
    """

    if n < 0:
        raise ValueError(n)
    if limit is None or limit <= 0 or limit >= n:
        return list(range(n))
    rng = random.Random(seed)
    picked = rng.sample(range(n), limit)
    picked.sort()
    return picked
