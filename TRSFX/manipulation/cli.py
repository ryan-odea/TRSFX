from pathlib import Path

import click
import numpy as np

from .._utils import read_h5, read_stream, write_h5, write_stream
from .crystfel_to_meteor import crystfel_to_meteor as func
from .sample_stream import sample_crystals as sample_func


@click.group()
def cli():
    """
    PSI crystallographic manipulation utilities
    """
    pass


@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file path. If not provided, appends '_meteor.mtz' to input filename.",
)
def crystfel_to_meteor(input, output):
    """
    Command-line interface to convert CrystFEL-processed MTZ files to Meteor-ready MTZ files.

    Example usage:

    \b
        sfx.manip crystfel-to-meteor input_file.mtz --output output_file.mtz
    """
    func(input=input, output=output)


@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option(
    "--count", "-n", type=int, default=100, help="Number of crystals to sample."
)
@click.option(
    "--percent",
    "-p",
    type=float,
    default=None,
    help="Percentage of crystals to sample (0-100).",
)
@click.option(
    "--seed", "-s", type=int, default=2026, help="Random seed for reproducibility."
)
def sample_crystals(input, output, count, percent, seed):
    """
    Command-line interface to sample crystals from a stream file.

    Example usage:

    \b
        sfx.manip sample-crystals input.stream output.stream --count 100 --seed 2026
    """
    stream = read_stream(input)

    if count is None and percent is None:
        click.echo("No downsampling parameters provided. Use --count or --percent.")
        return

    selected, _ = sample_func(stream, count=count, percent=percent, seed=seed)
    write_stream(stream.preamble, selected, output)


@cli.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to the input HDF5 file containing the frames.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path where the result HDF5 file will be written.",
)
def subtract_med(input_path: Path, output_path: Path):
    """
    Subtracts median from crystallographic data
    """
    frames_list = read_h5(input_path)
    data = np.array(frames_list)
    med = np.median(data, axis=0)

    corrected = data.astype(np.float32) - med.astype(np.float32)
    write_h5(output_path, corrected)


if __name__ == "__main__":
    cli()
