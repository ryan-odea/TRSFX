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


def consecutive_stats(stream: Stream) -> list[int]:
    """
    Get lengths of consecutive indexed frame sequences across all files.

    For each file, frames are sorted by event number. Consecutive runs of
    indexed frames are counted. E.g., if frames 0,1,2,3 are indexed, then
    frame 4 is not, then frames 5,6 are indexed, this yields [4, 2].

    :param stream: Parsed Stream object
    :type stream: Stream
    :return: List of consecutive run lengths across all files
    :rtype: list[int]
    """
    from collections import defaultdict

    chunks_by_file: dict[str, list[Chunk]] = defaultdict(list)
    for c in stream.chunks:
        chunks_by_file[c.filename].append(c)

    all_runs = []

    for _, chunks in chunks_by_file.items():
        sorted_chunks = sorted(chunks, key=lambda c: c.event_number)

        current_run = 0
        for chunk in sorted_chunks:
            if chunk.crystals:
                current_run += 1
            else:
                if current_run > 0:
                    all_runs.append(current_run)
                    current_run = 0

        if current_run > 0:
            all_runs.append(current_run)

    return all_runs


def plot_consecutive_stats(
    stream: Stream,
    output: Optional[str] = None,
    bins: int = 20,
) -> plt.Figure:
    """
    Plot distribution of consecutive indexed frame run lengths.

    :param stream: Parsed Stream object
    :type stream: Stream
    :param output: Optional path to save the figure
    :type output: Optional[str]
    :param bins: Number of histogram bins
    :type bins: int
    :return: matplotlib Figure object
    :rtype: plt.Figure
    """
    runs = consecutive_stats(stream)

    xlabel = "Consecutive Indexed Frames"
    title = "Distribution of Consecutive Indexed Frame Runs"
    bin_range = (0, max(runs) + 1) if runs else (0, 10)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        runs,
        bins=bins,
        range=bin_range,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    n_runs = len(runs)
    avg_run = np.mean(runs) if runs else 0
    max_run = max(runs) if runs else 0

    stats_text = (
        f"Total runs: {n_runs}\nMean length: {avg_run:.1f}\nMax length: {max_run}"
    )
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
