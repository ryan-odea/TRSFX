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


def _parse_extra_args(args: tuple) -> dict:
    """
    Parse extra CLI arguments into a params dict.

    Handles:
      --flag=value  -> {"flag": "value"}
      --flag value  -> {"flag": "value"}
      --flag        -> {"flag": True}
      -j 4          -> {"j": "4"}
    """
    result = {}
    args = list(args)
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            if "=" in arg:
                # --flag=value
                key, value = arg[2:].split("=", 1)
                result[key.replace("-", "_")] = value
            elif i + 1 < len(args) and not args[i + 1].startswith("-"):
                # --flag value
                key = arg[2:].replace("-", "_")
                result[key] = args[i + 1]
                i += 1
            else:
                # --flag (boolean)
                key = arg[2:].replace("-", "_")
                result[key] = True
        elif arg.startswith("-") and len(arg) == 2:
            # -j 4 style short flags
            key = arg[1:]
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                result[key] = args[i + 1]
                i += 1
            else:
                result[key] = True
        i += 1
    return result


@click.group()
@click.version_option()
def cli():
    """CrystFEL processing pipeline."""
    pass


@cli.command()
@click.option("--file-list", "-i", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True, type=click.Path())
@click.option("--entry-prefix", default="//", help="Entry prefix (default: //)")
@click.option("--n-frames", type=int, default=None, help="Number of frames (auto-detect if not set)")
@click.option("--start-index", default=0, help="Starting frame index")
def expand(file_list, output, entry_prefix, n_frames, start_index):
    """Expand file list to event list (3 columns: file //N N)."""
    expand_event_list(
        source_list=file_list,
        output_list=output,
        entry_prefix=entry_prefix,
        n_frames=n_frames,
        start_index=start_index,
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
@click.option("--cores", default=1, help="CPU cores per job")
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
    cores,
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
        click.echo(f"Warning: --n-subsample ({n_subsample}) > events in file ({n_events}), using all events")
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

    slurm = SlurmConfig(time=time, mem_gb=mem, cores=cores, partition=partition)

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
@click.option("--cores", default=1, help="CPU cores per job")
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
    cores,
    partition,
    camera_length,
    out_of_plane,
):
    """Generate Millepede calibration data for detector refinement."""
    params_dict = json.loads(Path(params).read_text())
    slurm = SlurmConfig(time=time, mem_gb=mem, cores=cores, partition=partition)

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
@click.option("--mille-dir", type=click.Path(exists=True), help="Custom mille directory (auto-detects mille/ or mille_bins/ if not set)")
@click.option("--modules", "-m", multiple=True, default=["crystfel/0.12.0"])
@click.option("--level", default=2)
@click.option("--time", default=30)
@click.option("--mem", default=64)
@click.option("--cores", default=1, help="CPU cores per job")
@click.option("--partition", default=None)
@click.option("--camera-length/--no-camera-length", default=True)
@click.option("--out-of-plane/--no-out-of-plane", default=False)
def align(
    directory, geometry, output, mille_dir, modules, level, time, mem, cores, partition, camera_length, out_of_plane
):
    """Run detector alignment on existing mille data."""
    directory = Path(directory)

    if mille_dir:
        mille_path = Path(mille_dir)
    else:
        mille_path = None
        for subdir in ["mille_bins", "mille"]:
            candidate = directory / subdir
            if candidate.exists() and list(candidate.glob("*.bin")):
                mille_path = candidate
                break

        if mille_path is None:
            raise click.ClickException(
                f"No mille .bin files found in {directory}/mille_bins or {directory}/mille\n"
                "Use --mille-dir to specify a custom location"
            )

    bins = list(mille_path.glob("*.bin"))
    if not bins:
        raise click.ClickException(f"No .bin files in {mille_path}")

    click.echo(f"Found {len(bins)} mille files in {mille_path}")

    config = AlignDetectorConfig(
        geometry_in=Path(geometry).resolve(),
        geometry_out=directory / output,
        mille_dir=mille_path,
        level=level,
        camera_length=camera_length,
        out_of_plane=out_of_plane,
    )
    config.to_cli(list(modules))

    logs_dir = directory / "align_logs"
    logs_dir.mkdir(exist_ok=True)

    slurm = SlurmConfig(time=time, mem_gb=mem, cores=cores, partition=partition, job_name="align_detector")
    executor = submitit.AutoExecutor(folder=logs_dir)
    executor.update_parameters(**slurm.to_dict())

    func = submitit.helpers.CommandFunction(["bash", "-c", config._cmd_str])
    job = executor.submit(func)

    click.echo(f"Alignment submitted: {job.job_id}")
    click.echo(f"Output will be: {directory / output}")


@cli.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
@click.option("--directory", "-d", required=True, type=click.Path())
@click.option("--list-file", "-i", required=True, type=click.Path(exists=True))
@click.option("--geometry", "-g", required=True, type=click.Path(exists=True))
@click.option("--cell", "-p", type=click.Path(exists=True))
@click.option("--params", type=click.Path(exists=True), help="JSON params file (optional if using passthrough args)")
@click.option("--modules", "-m", multiple=True, default=["crystfel/0.12.0"])
@click.option("--n-jobs", default=100)
@click.option("--time", default=360)
@click.option("--mem", default=32)
@click.option("--cores", default=1, help="CPU cores per job")
@click.option("--partition", default=None)
@click.option("--mille/--no-mille", default=False, help="Generate Millepede calibration data")
@click.option("--mille-level", default=2, help="Millepede hierarchy level (1-3)")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def index(ctx, directory, list_file, geometry, cell, params, modules, n_jobs, time, mem, cores, partition, mille, mille_level, extra_args):
    """
    Run production indexing.

    Extra CrystFEL arguments can be passed after --:

    \b
      sfx.index index -d out/ -i events.lst -g det.geom -- --indexing=xgandalf --peaks=peakfinder8
    """
    params_dict = {}
    if params:
        params_dict = json.loads(Path(params).read_text())

    params_dict.update(_parse_extra_args(extra_args))
    slurm = SlurmConfig(time=time, mem_gb=mem, cores=cores, partition=partition)

    idx = Indexamajig(
        directory=directory,
        list_file=list_file,
        geometry=geometry,
        params=params_dict,
        cell_file=cell,
        modules=list(modules),
        n_jobs=n_jobs,
        slurm=slurm,
        mille=mille,
        mille_level=mille_level,
    )
    idx.submit()

    click.echo(f"Submitted {n_jobs} indexing jobs to {directory}")
    if mille:
        click.echo(f"Mille data will be written to {directory}/mille/")


@cli.command()
@click.option("--directory", "-d", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True, type=click.Path())
def concat(directory, output):
    """Concatenate stream files (no merging)."""
    directory = Path(directory)
    streams_dir = directory / "streams"

    if not streams_dir.exists():
        streams_dir = directory

    concat_streams(streams_dir, output)
    click.echo(f"Concatenated to {output}")


@cli.command(
    context_settings=dict(ignore_unknown_options=True)
)
@click.option("--directory", "-d", required=True, type=click.Path(exists=True))
@click.option("--symmetry", "-w", required=True, help="Point group symmetry for merging")
@click.option("--true-symmetry", "-y", default=None, help="True symmetry for ambiguity resolution (triggers ambigator)")
@click.option("--output-name", "-o", default="merged")
@click.option("--input-stream", "-i", type=click.Path(exists=True), help="Input stream (default: auto-concat from directory/streams/)")
@click.option("--modules", "-m", multiple=True, default=["crystfel/0.12.0"])
@click.option("--model", default="xsphere")
@click.option("--iterations", default=1)
@click.option("--push-res", default=1.5)
@click.option("--ncorr", default=1000, help="Correlations for ambigator")
@click.option("--time", default=1440)
@click.option("--mem", default=128)
@click.option("--cores", "-j", default=1, help="CPU cores (used for both SLURM and -j flag)")
@click.option("--partition", default=None)
@click.option("--custom-split", type=click.Path(exists=True))
@click.option("--no-logs/--logs", default=True)
@click.option("--no-wait", is_flag=True, help="Don't wait for ambigator before submitting partialator")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def merge(
    directory,
    symmetry,
    true_symmetry,
    output_name,
    input_stream,
    modules,
    model,
    iterations,
    push_res,
    ncorr,
    time,
    mem,
    cores,
    partition,
    custom_split,
    no_logs,
    no_wait,
    extra_args,
):
    """
    Merge reflections: concat streams → [ambigator] → partialator.

    If --true-symmetry/-y is provided, runs ambigator first to resolve
    indexing ambiguity, then partialator.

    \b
    Examples:
      # Simple merge (no ambiguity)
      sfx.index merge -d indexing/ -w mmm -j 16

      # With ambiguity resolution
      sfx.index merge -d indexing/ -w mmm -y 222 -j 16

      # Extra partialator args after --
      sfx.index merge -d indexing/ -w mmm -- --min-res=2.0
    """
    directory = Path(directory)

    if input_stream is None:
        streams_dir = directory / "streams"
        if not streams_dir.exists():
            raise click.ClickException(f"No streams directory found: {streams_dir}")

        stream_files = list(streams_dir.glob("*.stream"))
        if not stream_files:
            raise click.ClickException(f"No .stream files in {streams_dir}")

        input_stream = directory / "all.stream"
        click.echo(f"Concatenating {len(stream_files)} stream files...")
        concat_streams(streams_dir, input_stream)
        click.echo(f"Created {input_stream}")
    else:
        input_stream = Path(input_stream)

    slurm = SlurmConfig(time=time, mem_gb=mem, cores=cores, partition=partition)

    if true_symmetry:
        click.echo(f"Running ambigator: -y {true_symmetry} -w {symmetry}")

        ambig_params = {"ncorr": ncorr, "j": cores}
        ambig = Ambigator(
            directory=directory,
            input_stream=input_stream,
            true_symmetry=true_symmetry,
            apparent_symmetry=symmetry,
            params=ambig_params,
            modules=list(modules),
            slurm=slurm,
        )
        ambig.submit()
        click.echo(f"Submitted ambigator: {ambig.job.job_id}")

        if not no_wait:
            click.echo("Waiting for ambigator to complete...")
            ambig.job.wait()
            result = ambig.job.result()
            click.echo("Ambigator complete")

        # Use ambigator output for partialator
        input_stream = ambig.output_stream

    # Step 2: Partialator
    params = {
        "model": model,
        "iterations": iterations,
        "push_res": push_res,
        "j": cores,
    }
    if custom_split:
        params["custom_split"] = custom_split
    if no_logs:
        params["no_logs"] = True

    params.update(_parse_extra_args(extra_args))

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

    click.echo(f"Submitted partialator: {partial.job.job_id}")
    click.echo(f"Output: {partial.output_hkl}")


@cli.command(
    context_settings=dict(ignore_unknown_options=True)
)
@click.option("--directory", "-d", required=True, type=click.Path())
@click.option("--input-stream", "-i", required=True, type=click.Path(exists=True))
@click.option("--true-symmetry", "-y", required=True, help="True point group symmetry")
@click.option("--apparent-symmetry", "-w", required=True, help="Apparent lattice symmetry")
@click.option("--modules", "-m", multiple=True, default=["crystfel/0.12.0"])
@click.option("--ncorr", default=1000)
@click.option("--unmerged-output", type=click.Path(), help="Write unmerged reflection list to file")
@click.option("--time", default=240)
@click.option("--mem", default=32)
@click.option("--cores", "-j", default=1, help="CPU cores (used for both SLURM and -j flag)")
@click.option("--partition", default=None)
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def ambigator(directory, input_stream, true_symmetry, apparent_symmetry, modules, ncorr, unmerged_output, time, mem, cores, partition, extra_args):
    """
    Resolve indexing ambiguities.

    Requires both true symmetry (-y) and apparent symmetry (-w) to determine
    the ambiguity operator.

    Extra ambigator arguments can be passed after --:

    \b
      sfx.index ambigator -d out/ -i data.stream -y mmm -w 6/mmm -j 16 -- --corr-matrix
    """
    slurm = SlurmConfig(time=time, mem_gb=mem, cores=cores, partition=partition)

    params = {"ncorr": ncorr, "j": cores}
    if unmerged_output:
        params["unmerged_output"] = unmerged_output

    params.update(_parse_extra_args(extra_args))

    ambig = Ambigator(
        directory=directory,
        input_stream=input_stream,
        true_symmetry=true_symmetry,
        apparent_symmetry=apparent_symmetry,
        params=params,
        modules=list(modules),
        slurm=slurm,
    )
    ambig.submit()

    click.echo("Submitted ambigator")
    click.echo(f"Output: {ambig.output_stream}")


@cli.command(
    context_settings=dict(ignore_unknown_options=True)
)
@click.option("--directory", "-d", required=True, type=click.Path())
@click.option("--input-stream", "-i", required=True, type=click.Path(exists=True))
@click.option("--symmetry", "-w", required=True)
@click.option("--output-name", "-o", default="merged")
@click.option("--modules", "-m", multiple=True, default=["crystfel/0.12.0"])
@click.option("--model", default="xsphere")
@click.option("--iterations", default=1)
@click.option("--push-res", default=1.5)
@click.option("--time", default=1440)
@click.option("--mem", default=128)
@click.option("--cores", "-j", default=1, help="CPU cores (used for both SLURM and -j flag)")
@click.option("--partition", default=None)
@click.option("--custom-split", type=click.Path(exists=True))
@click.option("--no-logs/--logs", default=True)
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def partialator(
    directory,
    input_stream,
    symmetry,
    output_name,
    modules,
    model,
    iterations,
    push_res,
    time,
    mem,
    cores,
    partition,
    custom_split,
    no_logs,
    extra_args,
):
    """
    Merge and scale reflections.

    Extra partialator arguments can be passed after --:

    \b
      sfx.index partialator -d out/ -i data.stream -w mmm -j 16 -- --min-res=2.0
    """
    params = {
        "model": model,
        "iterations": iterations,
        "push_res": push_res,
        "j": cores,
    }
    if custom_split:
        params["custom_split"] = custom_split
    if no_logs:
        params["no_logs"] = True

    params.update(_parse_extra_args(extra_args))
    slurm = SlurmConfig(time=time, mem_gb=mem, cores=cores, partition=partition)

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
    click.echo(f"  crystflow grid-search --base-params {base_path} --grid-params {grid_path} ...")


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
