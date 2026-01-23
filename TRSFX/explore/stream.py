from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted

from .._utils import Chunk, Stream


def get_time_series(stream: Stream) -> tuple[list[str], list[int], list[Chunk]]:
    """
    Generate time series data of indexed crystals over events

    :param stream: Stream object containing data
    :type stream: Stream
    :return: Tuple of (labels, peak_counts, sortedChunks)
    :rtype: tuple(list[str], list[int], list[Chunk])
    """
    sortedChunks = natsorted(stream.chunks, key=lambda c: c.sort_key)
    labels = [f"{Path(c.filename).name}:{c.event_number}" for c in sortedChunks]
    peaks = [c.num_peaks for c in sortedChunks]

    return labels, peaks, sortedChunks


def plot_time_series(
    stream: Stream,
    output: Optional[str] = None,
    show_labels: bool = False,
    fig_size: tuple[int, int] = (12, 6),
) -> plt.figure:
    """
    Plot time series of peaks per crystal

    :param stream: Stream object containing data
    :type stream: Stream
    :param output: Output file path for saving the plot
    :type output: Optional[str]
    :param fig_size: Dimension of the figure
    :type fig_size: tuple[int, int]
    :return: Matplotlib figure object
    :rtype: Any
    """

    labels, peaks, sorted = get_time_series(stream)
    fig, ax = plt.subplots(figsize=fig_size)
    x = np.arange(len(peaks))

    colors = ["coral" if c.hit else "steelblue" for c in sorted]
    ax.scatter(x, peaks, c=colors, alpha=0.1)
    ax.set_xlabel("Frame (Image:Event Number)", fontsize=12)
    ax.set_ylabel("Number of Peaks", fontsize=12)
    ax.set_title("Time Series of Peaks per Crystal (Ordered)", fontsize=14)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="coral", edgecolor="black", label="Hit"),
        Patch(facecolor="steelblue", edgecolor="black", label="Non-hit"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    if show_labels:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
    else:
        ax.set_xlim(-1, len(peaks))

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")

    return fig


def plot_peak_dist(
    stream: Stream,
    output: Optional[str] = None,
    bins: int = 50,
    fig_size: tuple[int, int] = (10, 6),
) -> plt.figure:
    """
    Plot histogram of peak counts, split by hits and non-hits

    :param stream: Stream object containing data
    :type stream: Stream
    :param output: Output file path for saving the plot
    :type output: Optional[str]
    :param bins: Number of bins for histogram
    :type bins: int
    :param fig_size: Figure size for the plot
    :type fig_size: tuple
    :return: Matplotlib figure object
    :rtype: Any
    """
    hit_peaks = [chunk.num_peaks for chunk in stream.hits]
    non_hit_peaks = [chunk.num_peaks for chunk in stream.non_hits]
    all_peaks = hit_peaks + non_hit_peaks

    fig, ax = plt.subplots(figsize=fig_size)

    bin_range = (0, max(all_peaks) + 1)

    if non_hit_peaks:
        ax.hist(
            non_hit_peaks,
            bins=bins,
            range=bin_range,
            alpha=0.7,
            label=f"Non-hits (n={len(non_hit_peaks)})",
            color="steelblue",
            edgecolor="black",
        )

    if hit_peaks:
        ax.hist(
            hit_peaks,
            bins=bins,
            range=bin_range,
            alpha=0.7,
            label=f"Hits (n={len(hit_peaks)})",
            color="coral",
            edgecolor="black",
        )

    ax.set_xlabel("Number of Peaks", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Peak Distribution: Hits vs Non-Hits", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")

    return fig
