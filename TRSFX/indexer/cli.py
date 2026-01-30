import csv
import json
from pathlib import Path

import click
import submitit

from ._configs import AlignDetectorConfig, GridSearchConfig, SlurmConfig
from ._utils import concat_streams, expand_event_list
from .crystfel_gridsearch import GridSearch
from .crystfel_indexing import Indexamajig
from .crystfel_merging import Ambigator, Partialator


@click.group()
@click.version_option()
def cli():
    """CrystFEL processing pipeline."""
    pass


@cli.command()
@click.option("--file-list", "-i", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True, type=click.Path())
@click.option("--pattern", default="//")
def expand(file_list, output, pattern):
    """Expand file list to event list."""
    expand_event_list(
        source_list=file_list,
        output_list=output,
        event_pattern=pattern,
    )
    click.echo(f"Expanded to {output}")


@cli.command()
@click.option("--directory", "-d", required=True, type=click.Path())
@click.option("--list-file", "-i", required=True, type=click.Path(exists=True))
@click.option("--geometry", "-g", required=True, type=click.Path(exists=True))
@click.option("--cell", "-p", type=click.Path(exists=True))
@click.option("--modules", "-m", multiple=True, default=["crystfel/0.12.0"])
@click.option("--n-subsample", default=1000, help="Events to subsample for testing")
@click.option("--n-jobs", default=4, help="Jobs per parameter combination")
@click.option("--time", default=60, help="Minutes per job")
@click.option("--mem", default=8, help="Memory in GB")
@click.option("--partition", default=None, help="SLURM partition")
@click.option(
    "--base-params",
    required=True,
    type=click.Path(exists=True),
    help="JSON file with base indexamajig params",
)
@click.option(
    "--grid-params",
    required=True,
    type=click.Path(exists=True),
    help="JSON file with grid search params (values must be lists)",
)
def grid_search(
    directory,
    list_file,
    geometry,
    cell,
    modules,
    n_subsample,
    n_jobs,
    time,
    mem,
    partition,
    base_params,
    grid_params,
):
    """
    Run parameter grid search for indexamajig.

    Requires two JSON files:

    \b
    --base-params: Base parameters applied to all runs
      {"indexing": "xgandalf", "peaks": "peakfinder8", "int_radius": "3,4,7"}

    \b
    --grid-params: Parameters to search (values must be lists)
      {"threshold": [6, 8, 10], "min_snr": [4.0, 5.0]}
    """
    base = json.loads(Path(base_params).read_text())
    grid = json.loads(Path(grid_params).read_text())

    list_path = Path(list_file)
    n_events = sum(1 for ln in list_path.read_text().splitlines() if ln.strip())
    if n_events == 0:
        raise click.ClickException(f"Input list file is empty: {list_file}")

    if n_subsample > n_events:
        click.echo(
            f"Warning: --n-subsample ({n_subsample}) > events in file ({n_events}), using all events"
        )
        n_subsample = n_events

    invalid_keys = [k for k, v in grid.items() if not isinstance(v, list)]
    if invalid_keys:
        raise click.ClickException(
            f"Grid params must be lists. Invalid keys: {', '.join(invalid_keys)}\n"
            f"Example: {{'threshold': [6, 8, 10]}} not {{'threshold': 6}}"
        )

    empty_keys = [k for k, v in grid.items() if isinstance(v, list) and len(v) == 0]
    if empty_keys:
        raise click.ClickException(
            f"Grid param lists cannot be empty: {', '.join(empty_keys)}"
        )

    try:
        config = GridSearchConfig(
            base_params=base,
            grid_params=grid,
            n_subsample=n_subsample,
            n_jobs_per_run=n_jobs,
        )
    except ValueError as e:
        raise click.ClickException(str(e))

    slurm = SlurmConfig(time=time, mem_gb=mem, partition=partition)

    try:
        gs = GridSearch(
            directory=directory,
            list_file=list_file,
            geometry=geometry,
            config=config,
            cell_file=cell,
            modules=list(modules),
            slurm=slurm,
            verbose=True,
        )
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))

    click.echo(f"Grid search: {config.n_combinations} parameter combinations")
    click.echo(f"Total jobs: {gs.n_total_jobs}")
    click.echo(f"Parameters: {', '.join(config.parameter_names)}")

    gs.submit()

    click.echo(f"\nSubmitted to {directory}")
    click.echo("Run 'crystflow grid-analyze -d <directory>' after jobs complete")


@cli.command()
@click.option("--directory", "-d", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default="best_params.json")
def grid_analyze(directory, output):
    """Analyze grid search results and save best parameters."""
    directory = Path(directory)

    try:
        analysis = GridSearch.analyze(directory)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))

    results = analysis["results"]

    if not results:
        raise click.ClickException("No results found. Jobs may still be running.")

    flat = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "params"}
        row.update(r["params"])
        flat.append(row)

    csv_path = directory / "grid_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat[0].keys())
        writer.writeheader()
        writer.writerows(flat)

    output_path = directory / output
    with open(output_path, "w") as f:
        json.dump(analysis["best_params"], f, indent=2)

    click.echo(analysis["summary"])
    click.echo(f"\nResults saved to: {csv_path}")
    click.echo(f"Best params saved to: {output_path}")


@cli.command()
@click.option("--directory", "-d", required=True, type=click.Path())
@click.option("--list-file", "-i", required=True, type=click.Path(exists=True))
@click.option("--geometry", "-g", required=True, type=click.Path(exists=True))
@click.option("--cell", "-p", required=True, type=click.Path(exists=True))
@click.option(
    "--params", required=True, type=click.Path(exists=True), help="JSON params file"
)
@click.option("--modules", "-m", multiple=True, default=["crystfel/0.12.0"])
@click.option("--n-jobs", default=10)
@click.option("--level", default=2, help="Millepede hierarchy level")
@click.option("--time", default=60)
@click.option("--mem", default=16)
@click.option("--partition", default=None)
@click.option("--camera-length/--no-camera-length", default=True)
@click.option("--out-of-plane/--no-out-of-plane", default=False)
def refine(
    directory,
    list_file,
    geometry,
    cell,
    params,
    modules,
    n_jobs,
    level,
    time,
    mem,
    partition,
    camera_length,
    out_of_plane,
):
    """Generate Millepede calibration data for detector refinement."""
    params_dict = json.loads(Path(params).read_text())
    slurm = SlurmConfig(time=time, mem_gb=mem, partition=partition)

    ref = Indexamajig.refine_detector(
        directory=directory,
        list_file=list_file,
        geometry=geometry,
        cell_file=cell,
        params=params_dict,
        modules=list(modules),
        n_jobs=n_jobs,
        mille_level=level,
        slurm=slurm,
        align_flags={"camera_length": camera_length, "out_of_plane": out_of_plane},
    )
    ref.submit()

    click.echo(f"Submitted {n_jobs} mille generation jobs to {directory}")
    click.echo("Run 'crystflow align' after jobs complete")


@cli.command()
@click.option("--directory", "-d", required=True, type=click.Path(exists=True))
@click.option("--geometry", "-g", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default="refined.geom")
@click.option("--modules", "-m", multiple=True, default=["crystfel/0.12.0"])
@click.option("--level", default=2)
@click.option("--time", default=30)
@click.option("--mem", default=64)
@click.option("--partition", default=None)
@click.option("--camera-length/--no-camera-length", default=True)
@click.option("--out-of-plane/--no-out-of-plane", default=False)
def align(
    directory,
    geometry,
    output,
    modules,
    level,
    time,
    mem,
    partition,
    camera_length,
    out_of_plane,
):
    """Run detector alignment on existing mille data."""
    directory = Path(directory)
    mille_dir = directory / "mille_bins"

    bins = list(mille_dir.glob("*.bin"))
    if not bins:
        raise click.ClickException(f"No .bin files in {mille_dir}")

    config = AlignDetectorConfig(
        geometry_in=Path(geometry).resolve(),
        geometry_out=directory / output,
        mille_dir=mille_dir,
        level=level,
        camera_length=camera_length,
        out_of_plane=out_of_plane,
    )
    config.to_cli(list(modules))

    logs_dir = directory / "align_logs"
    logs_dir.mkdir(exist_ok=True)

    slurm = SlurmConfig(
        time=time, mem_gb=mem, partition=partition, job_name="align_detector"
    )
    executor = submitit.AutoExecutor(folder=logs_dir)
    executor.update_parameters(**slurm.to_dict())

    func = submitit.helpers.CommandFunction(["bash", "-c", config._cmd_str])
    job = executor.submit(func)

    click.echo(f"Alignment submitted: {job.job_id}")
    click.echo(f"Output will be: {directory / output}")


@cli.command()
@click.option("--directory", "-d", required=True, type=click.Path())
@click.option("--list-file", "-i", required=True, type=click.Path(exists=True))
@click.option("--geometry", "-g", required=True, type=click.Path(exists=True))
@click.option("--cell", "-p", type=click.Path(exists=True))
@click.option("--params", required=True, type=click.Path(exists=True))
@click.option("--modules", "-m", multiple=True, default=["crystfel/0.12.0"])
@click.option("--n-jobs", default=100)
@click.option("--time", default=360)
@click.option("--mem", default=32)
@click.option("--partition", default=None)
def index(
    directory, list_file, geometry, cell, params, modules, n_jobs, time, mem, partition
):
    """Run production indexing."""
    params_dict = json.loads(Path(params).read_text())
    slurm = SlurmConfig(time=time, mem_gb=mem, partition=partition)

    idx = Indexamajig(
        directory=directory,
        list_file=list_file,
        geometry=geometry,
        params=params_dict,
        cell_file=cell,
        modules=list(modules),
        n_jobs=n_jobs,
        slurm=slurm,
    )
    idx.submit()

    click.echo(f"Submitted {n_jobs} indexing jobs to {directory}")


@cli.command()
@click.option("--directory", "-d", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True, type=click.Path())
def merge_streams(directory, output):
    """Concatenate stream files."""
    directory = Path(directory)
    streams_dir = directory / "streams"

    if not streams_dir.exists():
        streams_dir = directory

    concat_streams(streams_dir, output)
    click.echo(f"Merged to {output}")


@cli.command()
@click.option("--directory", "-d", required=True, type=click.Path())
@click.option("--input-stream", "-i", required=True, type=click.Path(exists=True))
@click.option("--true-symmetry", "-y", required=True, help="True point group symmetry")
@click.option(
    "--apparent-symmetry", "-w", required=True, help="Apparent lattice symmetry"
)
@click.option("--modules", "-m", multiple=True, default=["crystfel/0.12.0"])
@click.option("--ncorr", default=1000)
@click.option("--jobs", "-j", default=16)
@click.option("--time", default=240)
@click.option("--mem", default=32)
@click.option("--partition", default=None)
def ambigator(
    directory,
    input_stream,
    true_symmetry,
    apparent_symmetry,
    modules,
    ncorr,
    jobs,
    time,
    mem,
    partition,
):
    """
    Resolve indexing ambiguities.

    Requires both true symmetry (-y) and apparent symmetry (-w) to determine
    the ambiguity operator.
    """
    slurm = SlurmConfig(time=time, mem_gb=mem, partition=partition)

    ambig = Ambigator(
        directory=directory,
        input_stream=input_stream,
        true_symmetry=true_symmetry,
        apparent_symmetry=apparent_symmetry,
        params={"ncorr": ncorr, "j": jobs},
        modules=list(modules),
        slurm=slurm,
    )
    ambig.submit()

    click.echo("Submitted ambigator")
    click.echo(f"Output: {ambig.output_stream}")


@cli.command()
@click.option("--directory", "-d", required=True, type=click.Path())
@click.option("--input-stream", "-i", required=True, type=click.Path(exists=True))
@click.option("--symmetry", "-y", required=True)
@click.option("--output-name", "-o", default="merged")
@click.option("--modules", "-m", multiple=True, default=["crystfel/0.12.0"])
@click.option("--model", default="xsphere")
@click.option("--iterations", default=1)
@click.option("--push-res", default=1.5)
@click.option(
    "--unmerged-output",
    help="Output unmerged reflections (.unmerged)",
)
@click.option("--jobs", "-j", default=32)
@click.option("--time", default=1440)
@click.option("--mem", default=128)
@click.option("--partition", default=None)
@click.option("--custom-split", type=click.Path(exists=True))
@click.option("--no-logs/--logs", default=True)
def partialator(
    directory,
    input_stream,
    symmetry,
    output_name,
    modules,
    model,
    iterations,
    push_res,
    unmerged_output,
    jobs,
    time,
    mem,
    partition,
    custom_split,
    no_logs,
):
    """Merge and scale reflections."""
    params = {
        "model": model,
        "iterations": iterations,
        "push_res": push_res,
        "j": jobs,
        "unmerged-output": unmerged_output,
    }
    if custom_split:
        params["custom_split"] = custom_split
    if no_logs:
        params["no_logs"] = True

    slurm = SlurmConfig(time=time, mem_gb=mem, partition=partition)

    partial = Partialator(
        directory=directory,
        input_stream=input_stream,
        symmetry=symmetry,
        output_name=output_name,
        params=params,
        modules=list(modules),
        slurm=slurm,
    )
    partial.submit()

    click.echo("Submitted partialator")
    click.echo(f"Output: {partial.output_hkl}")


@cli.command()
@click.option("--output", "-o", default="crystflow_config.json")
def init(output):
    """Generate template config files for grid search."""
    base_template = {
        "indexing": "xgandalf,asdf,mosflm",
        "peaks": "peakfinder8",
        "int_radius": "3,4,7",
        "multi": True,
        "no_check_peaks": True,
    }

    grid_template = {
        "threshold": [6, 8, 10, 12],
        "min_snr": [4.0, 4.5, 5.0],
        "min_peaks": [8, 10, 12],
    }

    base_path = Path(output).stem + "_base.json"
    grid_path = Path(output).stem + "_grid.json"

    with open(base_path, "w") as f:
        json.dump(base_template, f, indent=2)

    with open(grid_path, "w") as f:
        json.dump(grid_template, f, indent=2)

    click.echo(f"Created {base_path} (base indexamajig params)")
    click.echo(f"Created {grid_path} (grid search params)")
    click.echo("\nUsage:")
    click.echo(
        f"  crystflow grid-search --base-params {base_path} --grid-params {grid_path} ..."
    )


@cli.command()
@click.option("--directory", "-d", required=True, type=click.Path(exists=True))
def status(directory):
    """Check job status for a directory."""
    directory = Path(directory)

    logs = list(directory.rglob("*.out"))
    if not logs:
        click.echo("No log files found")
        return

    stats = Indexamajig._parse_logs(directory)
    metrics = Indexamajig._compute_metrics(stats)

    click.echo(f"Processed: {stats['processed']}")
    click.echo(f"Hits: {stats['hits']} ({metrics['hit_rate']:.2f}%)")
    click.echo(f"Indexed: {stats['indexable']} ({metrics['indexing_rate']:.2f}%)")


if __name__ == "__main__":
    cli()
