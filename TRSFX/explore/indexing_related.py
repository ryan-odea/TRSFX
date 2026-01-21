from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .._utils.stream_io import Chunk, Stream


def get_consistent_crystals(stream: Stream) -> list[str]:
    """
    Find files where every frame was successfully indexed (has crystals).

    :param stream: Parsed Stream object
    :type stream: Stream
    :return: List of filenames where all frames have crystals
    :rtype: list[str]
    """
    from collections import defaultdict

    chunks_by_file: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in stream.chunks:
        chunks_by_file[chunk.filename].append(chunk)

    consistent_files = []
    for filename, chunks in chunks_by_file.items():
        if all(len(c.crystals) > 0 for c in chunks):
            consistent_files.append(filename)

    return sorted(consistent_files)


def indexing_stats(stream: Stream) -> dict[str, dict]:
    """
    Get indexing statistics for each file in the stream.

    :param stream: Parsed Stream object
    :type stream: Stream
    :return: Dict mapping filename to stats dict with keys:
             'total_frames', 'indexed_frames', 'fraction_indexed'
    :rtype: dict[str, dict]
    """
    from collections import defaultdict

    chunks: dict[str, list[Chunk]] = defaultdict(list)
    for c in stream.chunks:
        chunks[c.filename].append(c)

    stats = {}
    for filename, chunks in chunks.items():
        total = len(chunks)
        indexed = sum(1 for c in chunks if c.crystals)
        stats[filename] = {
            "total_frames": total,
            "indexed_frames": indexed,
            "fraction_indexed": indexed / total if total > 0 else 0.0,
        }

    return stats


def plot_stats(
    stream: Stream,
    output: Optional[str] = None,
    bins: int = 20,
) -> plt.Figure:
    """
    Plot distribution of indexed frame counts (or fractions) per file.
    """
    stats = indexing_stats(stream)

    values = [s["indexed_frames"] for s in stats.values()]
    xlabel = "Number of Indexed Frames"
    title = "Distribution of Indexed Frame Counts per File"
    bin_range = (0, max(values) + 1) if values else (0, 10)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        values,
        bins=bins,
        range=bin_range,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Number of Files", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    n_files = len(values)
    n_fully_indexed = sum(1 for s in stats.values() if s["fraction_indexed"] == 1.0)
    avg_fraction = (
        np.mean([s["fraction_indexed"] for s in stats.values()]) if stats else 0
    )

    stats_text = f"Total files: {n_files}\nFully indexed: {n_fully_indexed}\nMean rate: {avg_fraction:.1%}"
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")

    return fig
