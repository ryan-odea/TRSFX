import csv
import json
from pathlib import Path

import click
import submitit

from ._configs import AlignDetectorConfig
from ._utils import concat_streams, expand_event_list
from .crystfel_indexing import Indexamajig
from .crystfel_merging import Ambigator, Partialator


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="JSON config file")
@click.pass_context
def cli(ctx, config):
    """CrystFEL processing pipeline."""
    ctx.ensure_object(dict)
    if config:
        ctx.obj = json.loads(Path(config).read_text())


@cli.command()
@click.option("--file-list", "-i", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True, type=click.Path())
@click.option("--pattern", default="//{i}")
@click.pass_context
def expand(ctx, file_list, output, pattern):
    """Expand file list to event list."""
    expand_event_list(
        source_list=file_list,
        output_list=output,
        event_pattern=pattern,
    )


@cli.command()
@click.option("--directory", "-d", required=True, type=click.Path())
@click.option("--list-file", "-i", required=True, type=click.Path(exists=True))
@click.option("--geometry", "-g", required=True, type=click.Path(exists=True))
@click.option("--cell", "-p", type=click.Path(exists=True))
@click.option("--modules", "-m", multiple=True, default=["crystfel/0.12.0"])
@click.option("--n-subsample", default=1000)
@click.option("--n-jobs", default=4)
@click.option("--time", default=60, help="Minutes")
@click.option("--mem", default=8, help="GB")
@click.option(
    "--base-params", type=click.Path(exists=True), help="JSON file with base params"
)
@click.option(
    "--grid-params", type=click.Path(exists=True), help="JSON file with grid params"
)
@click.pass_context
def grid_search(
    ctx,
    directory,
    list_file,
    geometry,
    cell,
    modules,
    n_subsample,
    n_jobs,
    time,
    mem,
    base_params,
    grid_params,
):
    """Run parameter grid search."""
    base = (
        json.loads(Path(base_params).read_text())
        if base_params
        else ctx.obj.get("indexing_params", {})
    )
    grid = (
        json.loads(Path(grid_params).read_text())
        if grid_params
        else ctx.obj.get("grid_params", {})
    )

    if not base or not grid:
        raise click.ClickException(
            "Provide --base-params and --grid-params or use --config"
        )

    gs = Indexamajig.grid_search(
        directory=directory,
        list_file=list_file,
        geometry=geometry,
        base_params=base,
        grid_params=grid,
        cell_file=cell,
        modules=list(modules),
        n_subsample=n_subsample,
        n_jobs_per_run=n_jobs,
        slurm_directives={"time": time, "mem_gb": mem},
    )
    gs.submit()

    click.echo(f"Submitted to {directory}")
    click.echo("Run 'crystflow grid-analyze' after jobs complete")


@cli.command()
@click.option("--directory", "-d", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default="best_params.json")
def grid_analyze(directory, output):
    """Analyze grid search results and save best parameters."""

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())

    results = []
    for entry in manifest:
        run_dir = Path(entry["directory"])
        stats = Indexamajig._parse_logs(run_dir / "logs")
        metrics = Indexamajig._compute_metrics(stats)
        results.append(
            {
                "run_id": entry["run_id"],
                **metrics,
                **stats,
                "params": entry["params"],
            }
        )

    results.sort(key=lambda x: x["indexing_rate"], reverse=True)

    # Save CSV
    flat = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "params"}
        row.update(r["params"])
        flat.append(row)

    with open(directory / "grid_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat[0].keys())
        writer.writeheader()
        writer.writerows(flat)

    # Save best params
    best = results[0]
    output_path = directory / output
    with open(output_path, "w") as f:
        json.dump(best["params"], f, indent=2)

    click.echo(f"Best: {best['run_id']} ({best['indexing_rate']:.2f}% indexing rate)")
    click.echo(f"Saved to {output_path}")


@cli.command()
@click.option("--directory", "-d", required=True, type=click.Path())
@click.option("--list-file", "-i", required=True, type=click.Path(exists=True))
@click.option("--geometry", "-g", required=True, type=click.Path(exists=True))
@click.option("--cell", "-p", required=True, type=click.Path(exists=True))
@click.option(
    "--params", required=True, type=click.Path(exists=True), help="JSON params file"
)
@click.option("--modules", "-m", multiple=True, default=["crystfel/0.12.0"])
@click.option("--n-jobs", default=20)
@click.option("--level", default=2)
@click.option("--time", default=60)
@click.option("--mem", default=16)
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
    camera_length,
    out_of_plane,
):
    """Generate Millepede calibration data."""
    params_dict = json.loads(Path(params).read_text())

    ref = Indexamajig.refine_detector(
        directory=directory,
        list_file=list_file,
        geometry=geometry,
        cell_file=cell,
        params=params_dict,
        modules=list(modules),
        n_jobs=n_jobs,
        mille_level=level,
        slurm_directives={"time": time, "mem_gb": mem},
        align_flags={"camera_length": camera_length, "out_of_plane": out_of_plane},
    )
    ref.submit()

    click.echo(f"Submitted mille generation to {directory}")
    click.echo("Run 'crystflow align' after jobs complete")


@cli.command()
@click.option("--directory", "-d", required=True, type=click.Path(exists=True))
@click.option("--geometry", "-g", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default="refined.geom")
@click.option("--modules", "-m", multiple=True, default=["crystfel/0.12.0"])
@click.option("--level", default=2)
@click.option("--time", default=30)
@click.option("--mem", default=64)
@click.option("--camera-length/--no-camera-length", default=True)
@click.option("--out-of-plane/--no-out-of-plane", default=False)
def align(
    directory, geometry, output, modules, level, time, mem, camera_length, out_of_plane
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

    executor = submitit.AutoExecutor(folder=logs_dir)
    executor.update_parameters(job_name="align_detector", time=time, mem_gb=mem)

    func = submitit.helpers.CommandFunction([config._cmd_str], shell=True)
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
def index(directory, list_file, geometry, cell, params, modules, n_jobs, time, mem):
    """Run production indexing."""
    params_dict = json.loads(Path(params).read_text())

    idx = Indexamajig(
        directory=directory,
        list_file=list_file,
        geometry=geometry,
        params=params_dict,
        cell_file=cell,
        modules=list(modules),
        n_jobs=n_jobs,
        slurm_directives={"time": time, "mem_gb": mem},
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
@click.option("--symmetry", "-w", required=True, help="Apparent symmetry")
@click.option("--modules", "-m", multiple=True, default=["crystfel/0.12.0"])
@click.option("--ncorr", default=1000)
@click.option("--jobs", "-j", default=16)
@click.option("--time", default=240)
@click.option("--mem", default=32)
def ambigator(directory, input_stream, symmetry, modules, ncorr, jobs, time, mem):
    """Resolve indexing ambiguities."""
    ambig = Ambigator(
        directory=directory,
        input_stream=input_stream,
        symmetry=symmetry,
        params={"ncorr": ncorr, "j": jobs},
        modules=list(modules),
        slurm_directives={"time": time, "mem_gb": mem},
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
@click.option("--jobs", "-j", default=32)
@click.option("--time", default=1440)
@click.option("--mem", default=128)
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
    jobs,
    time,
    mem,
    custom_split,
    no_logs,
):
    """Merge and scale reflections."""
    params = {
        "model": model,
        "iterations": iterations,
        "push_res": push_res,
        "j": jobs,
    }
    if custom_split:
        params["custom_split"] = custom_split
    if no_logs:
        params["no_logs"] = True

    partial = Partialator(
        directory=directory,
        input_stream=input_stream,
        symmetry=symmetry,
        output_name=output_name,
        params=params,
        modules=list(modules),
        slurm_directives={"time": time, "mem_gb": mem},
    )
    partial.submit()

    click.echo("Submitted partialator")
    click.echo(f"Output: {partial.output_hkl}")


@cli.command()
@click.option("--output", "-o", default="crystflow_config.json")
def init(output):
    """Generate a template config file."""
    template = {
        "modules": ["crystfel/0.12.0"],
        "indexing_params": {
            "indexing": "xgandalf,asdf,mosflm",
            "peaks": "peakfinder8",
            "int_radius": "3,4,7",
            "multi": True,
            "no_check_peaks": True,
        },
        "grid_params": {
            "threshold": [6, 8, 10, 12],
            "min_snr": [4.0, 4.5, 5.0],
            "min_peaks": [8, 10, 12],
        },
    }

    with open(output, "w") as f:
        json.dump(template, f, indent=2)

    click.echo(f"Created {output}")


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
