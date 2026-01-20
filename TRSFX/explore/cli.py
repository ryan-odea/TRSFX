import click
from matplotlib import pyplot as plt

from .._utils import read_stream
from .stream import plot_peak_dist
from .stream import plot_time_series as ts_func


@click.group()
def cli():
    """
    PSI crystallographic data exploration utilities
    """
    pass


@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file path for saving the plot.",
)
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


@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.option(
    "--output", "-o", type=click.Path(), default=None, help="Output file path."
)
@click.option(
    "--bins", "-b", type=int, default=50, help="Number of bins for histogram."
)
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
