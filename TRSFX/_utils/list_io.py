from pathlib import Path


def write_list(filenames: list[str], output: str | Path) -> None:
    """
    Write a list of filenames.

    :param filenames: List of filenames to write
    :type filenames: list[str]
    :param output: Output file path
    :type output: str | Path
    """
    output = Path(output)
    with output.open("w") as f:
        for filename in filenames:
            f.write(filename + "\n")
