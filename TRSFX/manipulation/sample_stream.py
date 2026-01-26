import random
from typing import Optional

from .._utils import Chunk, Stream


def sample_crystals(
    stream: Stream,
    count: Optional[int] = None,
    percent: Optional[float] = None,
    seed: int = 2026,
) -> tuple[list[Chunk], int]:
    """
    Sample crystals from the stream file

    :param stream: Stream object containing data
    :type stream: Stream
    :param count: Number of crystals to sample
    :type count: Optional[int]
    :param percent: Percentage of total crystals to sample (0-100)
    :type percent: Optional[float]
    :param seed: Random seed for reproducibility
    :type seed: int
    :return: Tuple of sampled chunks and total number of crystals sampled
    :rtype: tuple[list[Chunk], int]
    """

    chunks = [c for c in stream.chunks if c.crystals]
    total = sum(len(c.crystals) for c in chunks)

    if not chunks:
        return [], 0

    random.seed(seed)

    if percent is not None:
        n_select = max(1, int(len(chunks) * (percent / 100)))
    elif count is not None:
        n_select = min(count, len(chunks))
    else:
        n_select = len(chunks)

    selected = random.sample(chunks, n_select)
    return selected, total
