import random
from pathlib import Path
from typing import List, Union


def write_list(filenames: list[str], output: str | Path) -> None:
    """
    Write a list of filenames.

    :param filenames: List of filenames to write
    :type filenames: list[str]
    :param output: Output file path
    :type output: str | Path
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for filename in filenames:
            f.write(filename + "\n")


def subsample_list(
    master_list: Union[str, Path], output_path: Union[str, Path], n: int
) -> None:
    """
    Subsamples a list, writing it out

    :param master_list: List to sample from
    :type master_list: Path
    :param output_path: Description
    :type output_path: Path
    :param n: Description
    :type n: int
    """
    with open(Path(master_list), "r") as f:
        lines = [x.strip() for x in f if x.strip()]

    selection = random.sample(lines, n) if len(lines) > n else lines
    write_list(selection, output_path)
    return


def split_list(
    master_list: Union[str, Path], output_dir: Union[str, Path], n_chunks: int
) -> List[Path]:
    """
    Docstring for split_list

    :param master_list: Description
    :type master_list: Union[str, Path]
    :param output_dir: Description
    :type output_dir: Union[str, Path]
    :param n_chunks: Description
    :type n_chunks: int
    :return: Description
    :rtype: List[Path]
    """
    master_list = Path(master_list)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(master_list, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) == 0:
        raise ValueError(f"Input list {master_list} is empty.")
    chunk_size = (len(lines) + n_chunks - 1) // n_chunks

    created_files = []

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk_lines = lines[start_idx:end_idx]
        if not chunk_lines:
            break

        chunk_path = output_dir / f"chunk_{i:04d}.lst"
        write_list(chunk_lines, chunk_path)
        created_files.append(chunk_path)

    return created_files
