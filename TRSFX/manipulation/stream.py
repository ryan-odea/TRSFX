import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted


@dataclass
class Chunk:
    """
    Represents a data chunk from within a stream file
    """

    filename: str
    event: str
    serial_number: int
    hit: bool
    indexed_by: str
    num_peaks: int
    raw: str
    crystals: list = field(default_factory=list)

    @property
    def event_number(self) -> int:
        """
        Extract event number from event string
        """
        match = re.search(r"//(\d+)", self.event)
        return int(match.group(1)) if match else 0

    @property
    def sort_key(self) -> str:
        """
        Key for natural sorting of chunks
        """
        return (self.filename, self.event_number)


@dataclass
class Stream:
    """
    Represents a full stream file
    """

    preamble: str
    chunks: list[Chunk]

    @property
    def hits(self) -> list[Chunk]:
        """
        List of chunks that are hits
        """
        return [chunk for chunk in self.chunks if chunk.hit]

    @property
    def non_hits(self) -> list[Chunk]:
        """
        List of chunks that are non-hits
        """
        return [chunk for chunk in self.chunks if not chunk.hit]

    @property
    def n_crystals(self) -> int:
        """
        Total number of crystals across all chunks
        """
        return sum(len(chunk.crystals) for chunk in self.chunks)


def read_stream(filepath: str | Path) -> Stream:
    """
    Parse stream file into structured data

    :param filepath: Filepath for stream file
    :type filepath: str | Path
    :return: Stream object containing parsed data
    :rtype: Stream
    """

    content = Path(filepath).read_text()

    chunk_pattern = r"----- Begin chunk -----"
    parts = re.split(chunk_pattern, content)
    preamble = parts[0].rstrip()
    chunks = []

    for t in parts[1:]:
        end_match = re.search(r"----- End chunk -----", t)
        if end_match:
            chunk_content = t[: end_match.end()]
        else:
            chunk_content = t

        filename = re.search(r"Image filename: (.+)", chunk_content)
        event = re.search(r"Event: (.+)", chunk_content)
        serial = re.search(r"Image serial number: (\d+)", chunk_content)
        hit = re.search(r"hit = (\d+)", chunk_content)
        indexed = re.search(r"indexed_by = (.+)", chunk_content)
        peaks = re.search(r"num_peaks = (\d+)", chunk_content)

        crystals = []
        crystal_blocks = re.findall(
            r"--- Begin crystal.*?--- End crystal", chunk_content, re.DOTALL
        )

        for c in crystal_blocks:
            crystals.append(c)

        chunk = Chunk(
            filename=filename.group(1) if filename else "",
            event=event.group(1) if event else "",
            serial_number=int(serial.group(1)) if serial else 0,
            hit=hit.group(1) == "1" if hit else False,
            indexed_by=indexed.group(1) if indexed else "none",
            num_peaks=int(peaks.group(1)) if peaks else 0,
            raw=chunk_content,
            crystals=crystals,
        )

        chunks.append(chunk)

        return Stream(preamble=preamble, chunks=chunks)


def write_stream(preamble: str, chunks: list[Chunk], output: str | Path) -> None:
    """
    Write structured stream data back to file

    :param preamble: Preamble text for stream file
    :type preamble: str
    :param chunks: List of Chunk objects to write
    :type chunks: list[Chunk]
    :param output: Output file path
    :type output: str | Path
    """
    output = Path(output)
    with output.open("w") as f:
        f.write(preamble + "\n")
        for c in chunks:
            f.write(c.raw)
            if not c.raw.endswith("\n"):
                f.write("\n")


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
    ax.scatter(x, peaks, c=colors, alpha=0.7, edgecolor="black")
    ax.plot(x, peaks, linestyle="--", alpha=0.5, color="gray")
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
