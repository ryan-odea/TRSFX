from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter


def stability(
    frames: list,
    start_idx=0,
    end_idx=None,
    kernel_size=5,
    filename: Union[str, Path] = None,
):
    """
    Docstring for plot_stability

    :param frames: Description
    :type frames: list
    :param start_idx: Description
    :param end_idx: Description
    :param kernel_size: Description
    :param filename: Description
    :type filename: Union[str, Path]
    """
    n_total = len(frames)
    if end_idx is None or end_idx > n_total:
        end_idx = n_total

    num_pairs = end_idx - start_idx - 1
    if num_pairs < 1:
        print("Error: Need at least 2 frames to compute correlation.")
        return None

    h, w = frames[0].shape
    sum_corr_map = np.zeros((h, w), dtype=np.float32)

    for i in range(num_pairs):
        idx1 = start_idx + i
        idx2 = idx1 + 1

        img1 = frames[idx1].astype(np.float32)
        img2 = frames[idx2].astype(np.float32)

        mean1 = uniform_filter(img1, size=kernel_size, mode="reflect")
        mean2 = uniform_filter(img2, size=kernel_size, mode="reflect")

        mean12 = uniform_filter(img1 * img2, size=kernel_size, mode="reflect")
        mean1_sq = uniform_filter(img1**2, size=kernel_size, mode="reflect")
        mean2_sq = uniform_filter(img2**2, size=kernel_size, mode="reflect")

        numerator = mean12 - mean1 * mean2
        var1 = mean1_sq - mean1**2
        var2 = mean2_sq - mean2**2

        denominator = np.sqrt(np.maximum(var1, 0) * np.maximum(var2, 0))
        denominator[denominator == 0] = 1e-9

        local_corr = numerator / denominator
        sum_corr_map += np.clip(local_corr, -1, 1)

    avg_corr_map = sum_corr_map / num_pairs
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_corr_map, cmap="inferno", vmin=0, vmax=1)

    plt.title(f"Stability Map (Avg Correlation, Frames {start_idx}-{end_idx})")
    cbar = plt.colorbar()
    cbar.set_label("Mean Pearson")
    plt.axis("off")

    plt.tight_layout()
    if filename is not None:
        plt.savefig("test.png")

    return avg_corr_map
