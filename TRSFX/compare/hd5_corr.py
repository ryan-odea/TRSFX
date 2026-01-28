from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np


def trace(
    frames: list,
    start_idx: int = 0,
    end_idx: int = None,
    log_space: bool = False,
    filename: Union[str, Path] = None,
):
    """
    Calculates the scalar Pearson correlation coefficient for every consecutive pair of frames
    and plots it as a time series - primarily for finding a crystal size as a number of frames.

    :param frames: List of numpy arrays.
    :type frames: list
    :param start_idx: Start frame index. Defaults to 0.
    :type start_idx: int, optional
    :param end_idx: End frame index. Defaults to None.
    :type end_idx: int, optional
    :param log_space: If True, computes correlation on log10(data) to weight weak signals equally. Defaults to False.
    :type log_space: bool, optional
    :param filename: File path to save the output/plot. Defaults to None.
    :type filename: str or Path, optional
    :return: A tuple containing:

        * **indices** (*list*): The frame indices [0, 1, 2...]
        * **correlations** (*list*): The correlation values [r(0,1), r(1,2)...]

    :rtype: tuple
    """
    n_total = len(frames)
    if end_idx is None or end_idx > n_total:
        end_idx = n_total

    num_pairs = end_idx - start_idx - 1
    if num_pairs < 1:
        print("Error: Need at least 2 frames.")
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

        r = np.corrcoef(vals1, vals2)[0, 1]
        if np.isnan(r):
            r = 0.0

        correlations.append(r)
        indices.append(curr_idx)

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
    plt.title(f"Frame-to-Frame Correlation Trace ({'Log' if log_space else 'Linear'})")
    plt.xlabel("Frame Index (n vs n+1)")
    plt.ylabel("Pearson Correlation Coefficient")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Normally correlations are very small due to the noise
    plt.ylim(max(0, min(correlations) - 0.0001), min(1, max(correlations) + 0.0001))

    plt.tight_layout()
    plt.show()

    if filename is not None:
        plt.savefig(Path(filename))

    return indices, correlations
