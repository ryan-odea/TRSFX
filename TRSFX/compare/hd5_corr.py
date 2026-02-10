import csv
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from .._utils import read_h5


def trace(
    pattern: str,
    output_csv: Union[str, Path],
    start_idx: int = 0,
    end_idx: int = None,
    log_space: bool = False,
    plot: bool = False,
) -> None:
    """
    Globs HDF5 files matching a pattern, computes frame-to-frame correlations for each,
    and streams results to a CSV.

    :param pattern: Glob pattern (e.g., "data/*.h5" or "/path/to/**/*.h5")
    :param output_csv: Path for output CSV file
    :param start_idx: Start frame index for correlation calculation
    :param end_idx: End frame index for correlation calculation
    :param log_space: If True, compute correlation on log10(data)
    :param plot: If True, generate plots (saved next to -o output as .png)
    """
    input_path = Path(pattern)

    if input_path.exists() and input_path.is_file():
        files = [input_path]
    elif input_path.is_absolute():
        files = sorted(Path("/").glob(pattern.lstrip("/")))
    else:
        files = sorted(Path.cwd().glob(pattern))

    if not files:
        return

    output_path = Path(output_csv)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "index", "corr_coef"])
        writer.writeheader()

        for filepath in files:
            frames = read_h5(filename=str(filepath))
            indices, correlations = _compute_correlations(
                frames, start_idx, end_idx, log_space
            )

            for idx, corr in zip(indices, correlations):
                writer.writerow(
                    {
                        "filename": str(filepath),
                        "index": idx,
                        "corr_coef": corr,
                    }
                )

            if plot:
                plot_path = Path(output_csv).with_suffix(".png")
                _plot_trace(indices, correlations, log_space, filepath.stem, plot_path)


def _compute_correlations(
    frames: list,
    start_idx: int = 0,
    end_idx: int = None,
    log_space: bool = False,
) -> Tuple[List[int], List[float]]:
    """Core correlation computation logic."""
    n_total = len(frames)
    if end_idx is None or end_idx > n_total:
        end_idx = n_total

    num_pairs = end_idx - start_idx - 1
    if num_pairs < 1:
        return [], []

    correlations = []
    indices = []

    for i in range(num_pairs):
        curr_idx = start_idx + i
        next_idx = curr_idx + 1
        f1 = frames[curr_idx]
        f2 = frames[next_idx]

        if log_space:
            valid_mask = (f1 > 0) & (f2 > 0)
            if np.sum(valid_mask) < 2:
                correlations.append(0.0)
                indices.append(curr_idx)
                continue
            vals1 = np.log10(f1[valid_mask])
            vals2 = np.log10(f2[valid_mask])
        else:
            vals1 = f1.ravel()
            vals2 = f2.ravel()

        v1 = vals1 - vals1.mean()
        v2 = vals2 - vals2.mean()
        den = np.sqrt((v1**2).sum() * (v2**2).sum())
        r = (v1 * v2).sum() / den if den > 0 else 0.0

        correlations.append(0.0 if np.isnan(r) else float(r))
        indices.append(curr_idx)

    return indices, correlations


def _plot_trace(
    indices: List[int],
    correlations: List[float],
    log_space: bool,
    title_suffix: str,
    output_path: Path,
):
    """Generate and save correlation trace plot."""
    plt.figure(figsize=(12, 5))
    plt.plot(
        indices,
        correlations,
        marker="o",
        markersize=3,
        linestyle="-",
        linewidth=1,
        color="royalblue",
    )
    mode = "Log" if log_space else "Linear"
    plt.title(f"Frame-to-Frame Correlation Trace ({mode}) - {title_suffix}")
    plt.xlabel("Frame Index (n vs n+1)")
    plt.ylabel("Pearson Correlation Coefficient")
    plt.grid(True, alpha=0.3)
    if correlations:
        plt.ylim(max(0, min(correlations) - 0.0001), min(1, max(correlations) + 0.0001))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
