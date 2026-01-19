import re
from itertools import combinations
from pathlib import Path
from typing import Sequence, Tuple

import gemmi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

COLUMN_PRESETS = {
    "meteor": ["F", "PHI"],
    "phenix": ["FoFo", "PHFc"],
}


def _detect_cols(mtz_path: str | Path) -> Tuple[str, str]:
    """
    Detects the column names for the map correlation function.

    :param mtz_path: Path to the MTZ file
    :type mtz_path: str | Path
    :return: Tuple of column names (F, PHI)
    :rtype: Tuple[str, str]
    """

    mtz = gemmi.read_mtz_file(str(mtz_path))
    col_names = [col.label for col in mtz.columns]
    for preset, names in COLUMN_PRESETS.items():
        if all(name in col_names for name in names):
            return names[0], names[1]
    raise ValueError(
        f"Could not detect column names in {mtz_path}. Please specify manually."
    )


def _load(
    mtz_path: str | Path,
    f: str | None = None,
    phi: str | None = None,
    grid_size: tuple[int, int, int] | None = None,
    d_min: float | None = None,
) -> gemmi.FloatGrid:
    """
    Load map coefficients using gemmi with optional resolution cutoff.

    :param mtz_path: Path to the MTZ file
    :type mtz_path: str | Path
    :param f: Column name for structure factors (F)
    :type f: str | None
    :param phi: Column name for phases (PHI)
    :type phi: str | None
    :param grid_size: Description
    :type grid_size: tuple[int, int, int] | None
    :param d_min: Description
    :type d_min: float | None
    :return: Description
    :rtype: FloatGrid
    """

    mtz = gemmi.read_mtz_file(str(mtz_path))
    if d_min:
        d = mtz.make_d_array()
        mask = d >= d_min
        mtz.set_data(mtz.array[mask])

    if grid_size:
        return mtz.transform_f_phi_to_map(f, phi, exact_size=grid_size)
    return mtz.transform_f_phi_to_map(f, phi, sample_rate=3)


def _map_cc(map1: gemmi.FloatGrid, map2: gemmi.FloatGrid) -> float:
    """
    Compute correlation between two gemmi grids.

    :param map1: First map grid
    :type map1: FloatGrid
    :param map2: Second map grid
    :type map2: FloatGrid
    :return: Correlation coefficient
    :rtype: float
    """

    arr1 = np.array(map1, copy=False).flatten()
    arr2 = np.array(map2, copy=False).flatten()
    return np.corrcoef(arr1, arr2)[0, 1]


def _label(
    n: int, time_step: str | None = None, ranges: bool = False
) -> list[str] | None:
    """
    Generate time-based labels for correlation matrix axes.

    :param n: Number of labels to generate
    :type n: int
    :param time_step: Step size with unit, e.g. "5ms", "100us", "1s". If None, returns None
    :type time_step: str | None
    :param ranges: If True, generate range labels (0-5ms, 5-10ms). If False, generate point labels (0ms, 5ms)
    :type ranges: bool
    :return: List of formatted time labels, or None if time_step not provided
    :rtype: list[str] | None
    """
    if time_step is None:
        return None

    match = re.match(r"^(\d+(?:\.\d+)?)\s*([a-zA-Zμµ]+)?$", time_step)
    if not match:
        raise ValueError(f"Invalid time_step format: {time_step}")

    value = float(match.group(1))
    unit = match.group(2)

    if ranges:
        return [f"{value * i:.4g}-{value * (i + 1):.4g}{unit}" for i in range(n)]
    else:
        return [f"{value * i:.4g}{unit}" for i in range(n)]


def map_correlation(
    mtz_files: Sequence[str | Path],
    labels: Sequence[str] | None = None,
    time_step: str | None = None,
    time_ranges: bool = False,
    f: str | None = None,
    phi: str | None = None,
    d_min: float | None = None,
) -> pd.DataFrame:
    """
    Compute pairwise map-map correlation matrix from MTZ files.

    :param mtz_files: Sequence of paths to MTZ files containing map coefficients
    :type mtz_files: Sequence[str | Path]
    :param labels: Labels for each file. If None, uses filenames as labels
    :type labels: Sequence[str] | None
    :param f: F column name. If None, auto-detected from first file
    :type f: str | None
    :param phi: Phase column name. If None, auto-detected from first file
    :type phi: str | None
    :param d_min: Resolution cutoff in Angstroms. If None, uses all reflections
    :type d_min: float | None
    :return: Square correlation matrix with labels as row/column indices
    :rtype: pd.DataFrame
    """

    mtz_files = list(mtz_files)
    n_files = len(mtz_files)

    if labels is None:
        labels = _label(n_files, time_step=time_step, ranges=time_ranges)
    if labels is None:
        labels = [Path(mtz).stem for mtz in mtz_files]

    if f is None or phi is None:
        f_detected, phi_detected = _detect_cols(mtz_files[0])
        f = f or f_detected
        phi = phi or phi_detected

    temp = [_load(mtz, f=f, phi=phi, d_min=d_min) for mtz in mtz_files]
    grid_size = tuple(min(m.shape[i] for m in temp) for i in range(3))

    maps = [
        _load(mtz, f=f, phi=phi, grid_size=grid_size, d_min=d_min) for mtz in mtz_files
    ]

    corr_mat = np.eye(n_files)
    for i, j in combinations(range(n_files), 2):
        corr_mat[i, j] = corr_mat[j, i] = _map_cc(maps[i], maps[j])

    return pd.DataFrame(corr_mat, index=labels, columns=labels)


def corr_heatmap(
    df: pd.DataFrame,
    output: str | Path | None = None,
    title: str | None = None,
    vmin: float = 0.2,
    vmax: float = 1.0,
    cmap: str = "RdYlBu_r",
    figsize: Tuple[int, int] = (10, 8),
    mask: bool = True,
    **kwargs,
) -> plt.Figure:
    """
    Plot and save a correlation matrix as a heatmap.

    :param df: Square correlation matrix with matching row/column labels
    :type df: pd.DataFrame
    :param output: Output file path. Defaults to ./correlation_heatmap.png
    :type output: str | Path | None
    :param title: Plot title
    :type title: str | None
    :param vmin: Minimum value for color scale
    :type vmin: float
    :param vmax: Maximum value for color scale
    :type vmax: float
    :param cmap: Matplotlib/seaborn colormap name
    :type cmap: str
    :param figsize: Figure dimensions as (width, height) in inches
    :type figsize: Tuple[int, int]
    :param mask: If True, mask values outside [vmin, vmax] range
    :type mask: bool
    :param kwargs: Additional arguments passed to sns.heatmap
    :return: Matplotlib figure object
    :rtype: plt.Figure
    """

    plot_data = df.copy()
    if mask:
        plot_data[(plot_data < vmin) | (plot_data > vmax)] = np.nan

    plt.figure(figsize=figsize)
    sns.heatmap(
        plot_data,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={"label": "Correlation"},
        **kwargs,
    )
    if title:
        plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output is None:
        output = Path.cwd() / "correlation_heatmap.png"
    plt.savefig(str(output), dpi=150)

    plt.close()
