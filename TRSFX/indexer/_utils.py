import random
import shutil
from pathlib import Path
from typing import Dict, List, Union

import h5py


def detect_n_frames(h5_path: Union[str, Path], dataset: str = "/data/data") -> int:
    """Detect number of frames in an HDF5 file by searching for the first valid dataset."""
    with h5py.File(h5_path, "r") as f:
        if dataset in f and isinstance(f[dataset], h5py.Dataset):
            return f[dataset].shape[0]

        def find_dataset(obj):
            if isinstance(obj, h5py.Dataset):
                if len(obj.shape) >= 2:
                    return obj.shape[0]
            elif isinstance(obj, h5py.Group):
                for key in obj.keys():
                    res = find_dataset(obj[key])
                    if res is not None:
                        return res
            return None

        n_frames = find_dataset(f)
        if n_frames is not None:
            return n_frames

    raise ValueError(
        f"Could not find a valid image dataset in {h5_path}. "
        f"Keys found: {list(f.keys())}"
    )


def expand_event_list(
    source_list: Union[str, Path],
    output_list: Union[str, Path],
    n_frames: int | None = None,
    event_pattern: str = "//",
    start_index: int = 0,
) -> Path:
    """
    Expand a file list into an event list for CrystFEL.

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
                f.write(f"{filepath} {event_pattern}{i} {i}\n")

    return output_list


def split_list(
    source_list: Union[str, Path],
    output_dir: Union[str, Path],
    n_chunks: int,
) -> List[Path]:
    """
    Split an event list into n_chunks roughly equal parts.

    Returns list of paths to chunk files.
    """
    source_list = Path(source_list)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [ln for ln in source_list.read_text().splitlines() if ln.strip()]
    n_lines = len(lines)

    if n_chunks > n_lines:
        n_chunks = n_lines

    chunk_size = n_lines // n_chunks
    remainder = n_lines % n_chunks

    chunks = []
    start = 0

    for i in range(n_chunks):
        # Distribute remainder across first chunks
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
    """
    source_list = Path(source_list)
    output_list = Path(output_list)

    lines = [ln for ln in source_list.read_text().splitlines() if ln.strip()]

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
