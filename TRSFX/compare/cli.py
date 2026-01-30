from pathlib import Path

import click
from natsort import natsorted

from .hd5_corr import trace
from .map_corr import COLUMN_PRESETS
from .map_corr import corr_heatmap as heatmap_func
from .map_corr import map_correlation as corr_func


@click.group()
def cli():
    """
    PSI crystallographic manipulation utilities
    """
    pass


@cli.command()
@click.argument("inputs", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="corr_matrix.csv",
    help="Output CSV file for the correlation matrix.",
)
@click.option(
    "--glob",
    "-g",
    "glob_pattern",
    type=str,
    default=None,
    help="Glob pattern to select input MTZ files.",
)
@click.option(
    "--plot",
    "-p",
    type=click.Path(),
    default=None,
    help="Output file for the correlation heatmap plot.",
)
@click.option(
    "--columns",
    "-c",
    type=click.Choice(list(COLUMN_PRESETS.keys())),
    default=None,
    help="Column preset (auto-detected if not specified).",
)
@click.option(
    "--f-col",
    type=str,
    default=None,
    help="Column name for F values if not auto-detected.",
)
@click.option(
    "--phi-col",
    type=str,
    default=None,
    help="Column name for PHI values if not auto-detected.",
)
@click.option(
    "--d-min", type=float, default=2.5, help="Resolution cutoff in Angstroms."
)
@click.option(
    "--vmin", type=float, default=0.2, help="Minimum correlation value for heatmap."
)
@click.option(
    "--vmax", type=float, default=1.0, help="Maximum correlation value for heatmap."
)
@click.option(
    "--limit",
    "-n",
    type=float,
    default=None,
    help="Limit first n files for correlation calculation.",
)
@click.option(
    "--time-step",
    "-t",
    type=str,
    default=None,
    help="Time step for labeling (e.g., '5ms').",
)
@click.option(
    "--time-ranges", is_flag=True, default=False, help="Use time ranges for labeling."
)
@click.option("--no-sort", is_flag=True, default=False, help="Do not sort input files.")
def map_cc(
    inputs,
    glob_pattern,
    output,
    plot,
    d_min,
    columns,
    f_col,
    phi_col,
    vmin,
    vmax,
    limit,
    time_step,
    time_ranges,
    no_sort,
):
    """
    Command-line interface to compute map correlation matrix from MTZ files.

    \b
    Examples:
        # Basic usage with glob
        sfx.compare map-cc -g "data/*.mtz" -p heatmap.png

        # With time labels and resolution cutoff
        sfx.compare map-cc -g "*.mtz" -t 5ms --d-min 2.5 -n 21

        # Range-style time labels
        sfx.compare map-cc -g "*.mtz" -t 5ms --time-ranges

        # Explicit files, preserve order
        sfx.compare map-cc a.mtz b.mtz c.mtz --no-sort

        # Force column preset
        sfx.compare map-cc -g "*.mtz" --columns phenix
    """

    if glob_pattern:
        mtz_files = list(Path(".").glob(glob_pattern))
    elif inputs:
        mtz_files = [Path(f) for f in inputs]
    else:
        raise click.UsageError("Either input files or a glob pattern must be provided.")

    if not no_sort:
        mtz_files = natsorted(mtz_files)

    if limit is not None:
        mtz_files = mtz_files[: int(limit)]

    if len(mtz_files) < 2:
        raise click.UsageError(
            "At least two MTZ files are required for correlation calculation."
        )

    if columns and not (f_col and phi_col):
        f_col, phi_col = COLUMN_PRESETS.get(columns)

    df = corr_func(
        mtz_files,
        time_step=time_step,
        time_ranges=time_ranges,
        f=f_col,
        phi=phi_col,
        d_min=d_min,
    )

    df.to_csv(output)

    if plot:
        title = "Map-Map Correlation"
        if d_min:
            title += f" ({d_min} Å cutoff)"
        heatmap_func(
            df,
            output=plot,
            title=title,
            vmin=vmin,
            vmax=vmax,
        )


@cli.command()
@click.argument("input", type=str)
@click.option(
    "--start", type=int, default=0, help="Start index for the correlation trace"
)
@click.option(
    "--stop", type=int, default=None, help="Stop index for the correlation trace"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(exists=False),
    required=True,
    help="Output filename for the CSV",
)
@click.option(
    "--plot", "-p", is_flag=True, help="Save trace plots as <input>_trace.png"
)
@click.option(
    "--log-space/--linear", default=True, help="Use log10 transform for correlation"
)
def hd5_trace(input, start, stop, output, plot, log_space):
    """
    Calculates frame-to-frame Pearson correlation coefficients for HDF5 files.
    INPUT can be a single HDF5 file or a glob pattern (e.g., "data/*.h5").
    Results are written to a CSV with columns: filename, index, corr_coef
    """
    trace(
        pattern=input,
        output_csv=output,
        start_idx=start,
        end_idx=stop,
        log_space=log_space,
        plot=plot,
    )


if __name__ == "__main__":
    cli()
