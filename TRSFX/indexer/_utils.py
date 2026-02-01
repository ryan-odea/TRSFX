import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Union


def detect_n_frames(h5_path: Union[str, Path], dataset: str = "/data/data") -> int:
    """Detect number of frames in an HDF5 file."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        if dataset in f:
            return f[dataset].shape[0]
        for key in f.keys():
            if "data" in f[key]:
                return f[key]["data"].shape[0]
    raise ValueError(f"Could not detect frames in {h5_path}")


def read_geometry_clen(geom_path: Union[str, Path]) -> float:
    """
    Read the camera length from a geometry file.

    Looks for panel-specific clen (e.g., p0/clen) or global clen.
    Returns the first clen value found.
    """
    geom_path = Path(geom_path)
    content = geom_path.read_text()

    match = re.search(r"^\s*\w+/clen\s*=\s*([\d.eE+-]+)", content, re.MULTILINE)
    if match:
        return float(match.group(1))

    match = re.search(r"^\s*clen\s*=\s*([\d.eE+-]+)", content, re.MULTILINE)
    if match:
        return float(match.group(1))

    raise ValueError(f"No clen found in {geom_path}")


def edit_geometry_clen(
    geom_path: Union[str, Path],
    output_path: Union[str, Path],
    new_clen: float,
) -> Path:
    """
    Create a copy of a geometry file with modified camera length.

    Replaces all clen values (both panel-specific and global) with the new value.

    Parameters
    ----------
    geom_path : path
        Input geometry file
    output_path : path
        Output geometry file (can be same as input to modify in place)
    new_clen : float
        New camera length in meters

    Returns
    -------
    Path
        Path to the output geometry file
    """
    geom_path = Path(geom_path)
    output_path = Path(output_path)

    content = geom_path.read_text()

    content = re.sub(
        r"^(\s*\w+/clen\s*=\s*)[\d.eE+-]+",
        rf"\g<1>{new_clen}",
        content,
        flags=re.MULTILINE,
    )
    content = re.sub(
        r"^(\s*clen\s*=\s*)[\d.eE+-]+",
        rf"\g<1>{new_clen}",
        content,
        flags=re.MULTILINE,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)

    return output_path


def generate_clen_geometries(
    geom_path: Union[str, Path],
    output_dir: Union[str, Path],
    clen_values: List[float],
) -> Dict[float, Path]:
    """
    Generate multiple geometry files with different camera lengths.

    Parameters
    ----------
    geom_path : path
        Template geometry file
    output_dir : path
        Directory for output geometry files
    clen_values : list of float
        Camera length values to generate

    Returns
    -------
    dict
        Mapping of clen value to geometry file path
    """
    geom_path = Path(geom_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geometries = {}
    for clen in clen_values:
        output_path = output_dir / f"clen_{clen:.6f}.geom"
        edit_geometry_clen(geom_path, output_path, clen)
        geometries[clen] = output_path

    return geometries


def expand_event_list(
    source_list: Union[str, Path],
    output_list: Union[str, Path],
    n_frames: int | None = None,
    entry_prefix: str = "//",
    start_index: int = 0,
) -> Path:
    """
    Expand a file list into an event list for CrystFEL.

    Output format: file entry frame_number
    Example: /path/file.h5 //1 1

    If n_frames is None, detects automatically from the first HDF5 file.
    """
    source_list = Path(source_list).resolve()
    output_list = Path(output_list).resolve()

    if not source_list.exists():
        raise FileNotFoundError(f"Source list not found: {source_list}")

    files = [
        ln.strip().split()[0]
        for ln in source_list.read_text().splitlines()
        if ln.strip()
    ]

    if n_frames is None:
        n_frames = detect_n_frames(files[0])

    output_list.parent.mkdir(parents=True, exist_ok=True)

    with open(output_list, "w") as f:
        for filepath in files:
            for i in range(start_index, start_index + n_frames):
                f.write(f"{filepath} {entry_prefix}{i} {i}\n")

    return output_list


def split_list(
    source_list: Union[str, Path],
    output_dir: Union[str, Path],
    n_chunks: int,
) -> List[Path]:
    """
    Split an event list into n_chunks roughly equal parts.

    Returns list of paths to chunk files.

    Raises
    ------
    FileNotFoundError
        If source_list doesn't exist
    ValueError
        If source_list is empty or n_chunks < 1
    """
    source_list = Path(source_list)
    output_dir = Path(output_dir)

    if not source_list.exists():
        raise FileNotFoundError(f"List file not found: {source_list}")

    if n_chunks < 1:
        raise ValueError(f"n_chunks must be >= 1, got {n_chunks}")

    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [ln for ln in source_list.read_text().splitlines() if ln.strip()]
    n_lines = len(lines)

    if n_lines == 0:
        raise ValueError(f"List file is empty: {source_list}")

    n_chunks = min(n_chunks, n_lines)

    chunk_size = n_lines // n_chunks
    remainder = n_lines % n_chunks

    chunks = []
    start = 0

    for i in range(n_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunk_lines = lines[start:end]
        start = end

        if not chunk_lines:
            continue

        chunk_path = output_dir / f"chunk_{i:04d}.lst"
        chunk_path.write_text("\n".join(chunk_lines) + "\n")
        chunks.append(chunk_path)

    return chunks


def subsample_list(
    source_list: Union[str, Path],
    output_list: Union[str, Path],
    n_samples: int,
    seed: int | None = None,
) -> Path:
    """
    Randomly subsample events from a list file.

    If n_samples exceeds the number of events, returns all events.

    Raises
    ------
    FileNotFoundError
        If source_list doesn't exist
    ValueError
        If source_list is empty or n_samples < 1
    """
    source_list = Path(source_list)
    output_list = Path(output_list)

    if not source_list.exists():
        raise FileNotFoundError(f"List file not found: {source_list}")

    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    lines = [ln for ln in source_list.read_text().splitlines() if ln.strip()]

    if len(lines) == 0:
        raise ValueError(f"List file is empty: {source_list}")

    if n_samples >= len(lines):
        sampled = lines
    else:
        if seed is not None:
            random.seed(seed)
        sampled = random.sample(lines, n_samples)

    output_list.parent.mkdir(parents=True, exist_ok=True)
    output_list.write_text("\n".join(sampled) + "\n")

    return output_list


def concat_streams(source_dir: Union[str, Path], output_file: Union[str, Path]) -> Path:
    """Concatenate all stream files in a directory."""
    source_dir = Path(source_dir)
    output_file = Path(output_file)

    streams = sorted(source_dir.glob("*.stream"))
    if not streams:
        raise FileNotFoundError(f"No streams in {source_dir}")

    with open(output_file, "wb") as out:
        for stream in streams:
            if stream.stat().st_size > 0:
                with open(stream, "rb") as src:
                    shutil.copyfileobj(src, out)

    return output_file


def parse_stream_stats(stream_path: Union[str, Path]) -> Dict[str, int]:
    """Count indexed crystals and chunks in a stream file."""
    chunks = 0
    crystals = 0

    with open(stream_path) as f:
        for line in f:
            if line.startswith("----- Begin chunk -----"):
                chunks += 1
            elif line.startswith("--- Begin crystal"):
                crystals += 1

    return {"chunks": chunks, "crystals": crystals}
