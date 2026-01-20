import click

from typing import Optional
import matplotlib.pyplot as plt

from .crystfel_to_meteor import crystfel_to_meteor as func
from .stream import (
    read_stream,
    write_stream,
    plot_peak_dist,
    plot_time_series as ts_func,
    sample_crystals as sample_func,
)

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
        psi-crystfel-to-meteor input_file.mtz --output output_file.mtz
    """
    func(input=input, output=output)

@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path.")
@click.option("--bins", "-b", type=int, default=50, help="Number of bins for histogram.")
def peak_dist(input, output, bins):
    """
    Command-line interface to plot peak distribution from a stream file.

    Example usage:

    \b
        sfx.manip peak-dist input.stream --output peak_distribution.png --bins 50
    """
    stream = read_stream(input)
    fig = plot_peak_dist(stream, output=output, bins=bins)
    if output:
        fig.savefig(output)
    else:
        plt.show()

@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("--count", "-n", type=int, default=100, help="Number of crystals to sample.")
@click.option("--percent", "-p", type=float, default=None, help="Percentage of crystals to sample (0-100).")
@click.option("--seed", "-s", type=int, default=2026, help="Random seed for reproducibility.")
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
@click.argument("input", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path for saving the plot.")
@click.option("--show-labels", is_flag=True, help="Show labels on x-axis.")
def peak_time_series(input, output, show_labels):
    """
    Command-line interface to plot time series of peaks per crystal from a stream file.

    Example usage:

    \b
        sfx.manip peak-time-series input.stream --output time_series.png --show-labels
    """
    stream = read_stream(input)
    fig = ts_func(stream, output=output, show_labels=show_labels)
    if output:
        fig.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    cli()
