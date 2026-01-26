import shutil
from pathlib import Path
from typing import Dict, Union

import h5py


def detect_n_frames(h5_path: Union[str, Path], dataset: str = "/data/data") -> int:
    """Detect number of frames in an HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        if dataset in f:
            return f[dataset].shape[0]
        for key in f.keys():
            if "data" in f[key]:
                return f[key]["data"].shape[0]
    raise ValueError(f"Could not detect frames in {h5_path}")


def expand_event_list(
    source_list: Union[str, Path],
    output_list: Union[str, Path],
    n_frames: int | None = None,
    event_pattern: str = "//{i}",
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
                tag = event_pattern.format(i=i)
                f.write(f"{filepath} {tag}\n")

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
