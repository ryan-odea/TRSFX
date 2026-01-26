import click
from matplotlib import pyplot as plt

from .._utils import read_stream, write_list
from .indexing_related import get_consistent_crystals, plot_consecutive_stats
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


@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option(
    "--plot", "-p", type=click.Path(), default=None, help="Output plot file path."
)
@click.option(
    "--bins", "-b", type=int, default=20, help="Number of bins for histogram."
)
def consistent_crystals(input, output, plot, bins):
    """
    Command-line interface to find files with consistently indexed crystals from a stream file.

    Optionally plots the distribution of consecutive indexed frame run lengths.

    Example usage:

    \b
        sfx.explore consistent-crystals input.stream output.lst
        sfx.explore consistent-crystals input.stream output.lst --plot dist.png
    """

    stream = read_stream(input)
    consistent_files = get_consistent_crystals(stream)

    if consistent_files:
        write_list(consistent_files, output)
    else:
        click.echo("No consistently indexed files found.")

    if plot:
        fig = plot_consecutive_stats(stream, output=plot, bins=bins)
        fig.savefig(plot)
    return
