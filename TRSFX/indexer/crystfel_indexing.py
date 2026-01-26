import csv
import itertools
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import submitit

from .._utils import split_list, subsample_list
from ._configs import IndexamajigConfig
from .crystfel_detector import DetectorRefinement


class Indexamajig:
    """
    Manages parallel indexamajig execution over chunked input lists.

    Splits input into chunks and prepares jobs. Call `submit()` to launch.
    Use the `grid_search()` classmethod to optimize parameters first.
    """

    _STATS_PATTERN = re.compile(
        r"Final: (?P<processed>\d+) images processed, "
        r"(?P<hits>\d+) hits \(.+?\), "
        r"(?P<indexable>\d+) indexable"
    )

    def __init__(
        self,
        directory: Union[str, Path],
        list_file: Union[str, Path],
        geometry: Union[str, Path],
        params: Dict[str, Any],
        cell_file: Optional[Union[str, Path]] = None,
        modules: List[str] | None = None,
        n_jobs: int = 100,
        slurm_directives: Dict[str, Any] | None = None,
        verbose: bool = False,
    ):
        self.directory = Path(directory)
        self.lists_dir = self.directory / "lists"
        self.streams_dir = self.directory / "streams"
        self.logs_dir = self.directory / "logs"
        self.verbose = verbose

        for d in [self.lists_dir, self.streams_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.params = params
        self.modules = modules
        self.slurm_directives = slurm_directives or {}
        self.jobs: List[submitit.Job] = []
        self.configs: List[IndexamajigConfig] = []

        chunks = split_list(list_file, self.lists_dir, n_jobs)

        for i, chunk in enumerate(chunks):
            stream = self.streams_dir / f"idx_{i:04d}.stream"
            config = IndexamajigConfig(
                geometry=geometry,
                input_list=chunk,
                output_stream=stream,
                cell_file=cell_file,
                params=params,
            )
            config.to_cli(modules)
            self.configs.append(config)

    def submit(self) -> List[submitit.Job]:
        """Submit all prepared jobs to the cluster."""
        executor = submitit.AutoExecutor(folder=self.logs_dir)

        for i, config in enumerate(self.configs):
            directives = dict(self.slurm_directives)
            directives.setdefault("job_name", f"idx_{i:04d}")
            executor.update_parameters(**directives)

            func = submitit.helpers.CommandFunction([config._cmd_str], shell=True)
            job = executor.submit(func)
            self.jobs.append(job)

        return self.jobs

    @property
    def job_ids(self) -> List[str]:
        return [j.job_id for j in self.jobs]

    def wait(self) -> None:
        for job in self.jobs:
            job.wait()

    @property
    def results(self) -> Dict[str, Any]:
        """Wait for jobs and return aggregated statistics."""
        self.wait()
        stats = self._parse_logs(self.logs_dir)
        metrics = self._compute_metrics(stats)
        return {**stats, **metrics, "params": self.params}

    @classmethod
    def grid_search(
        cls,
        directory: Union[str, Path],
        list_file: Union[str, Path],
        geometry: Union[str, Path],
        base_params: Dict[str, Any],
        grid_params: Dict[str, List[Any]],
        cell_file: Optional[Union[str, Path]] = None,
        modules: List[str] | None = None,
        n_subsample: int = 1000,
        n_jobs_per_run: int = 4,
        slurm_directives: Dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> "_Grid":
        """
        Create a grid search over indexamajig parameters.

        Subsamples the input list and runs indexamajig with every combination
        of parameters in `grid_params`.
        """
        return _Grid(
            directory=directory,
            list_file=list_file,
            geometry=geometry,
            base_params=base_params,
            grid_params=grid_params,
            cell_file=cell_file,
            modules=modules,
            n_subsample=n_subsample,
            n_jobs_per_run=n_jobs_per_run,
            slurm_directives=slurm_directives,
            verbose=verbose,
        )

    @classmethod
    def refine_detector(
        cls,
        directory: Union[str, Path],
        list_file: Union[str, Path],
        geometry: Union[str, Path],
        cell_file: Union[str, Path],
        params: Dict[str, Any],
        modules: List[str] | None = None,
        n_jobs: int = 10,
        mille_level: int = 2,
        slurm_directives: Dict[str, Any] | None = None,
        align_flags: Dict[str, bool] | None = None,
        verbose: bool = False,
    ) -> "DetectorRefinement":
        """
        Create a detector refinement workflow using Millepede.
        """
        return DetectorRefinement(
            directory=directory,
            list_file=list_file,
            geometry=geometry,
            cell_file=cell_file,
            params=params,
            modules=modules,
            n_jobs=n_jobs,
            mille_level=mille_level,
            slurm_directives=slurm_directives,
            align_flags=align_flags,
            verbose=verbose,
        )

    @classmethod
    def _parse_logs(cls, directory: Path) -> Dict[str, int]:
        totals = {"processed": 0, "hits": 0, "indexable": 0}

        for pattern in ("*.out", "*.err", "*.log"):
            for log in directory.rglob(pattern):
                try:
                    match = cls._STATS_PATTERN.search(log.read_text())
                    if match:
                        totals["processed"] += int(match.group("processed"))
                        totals["hits"] += int(match.group("hits"))
                        totals["indexable"] += int(match.group("indexable"))
                except Exception:
                    continue

        return totals

    @staticmethod
    def _compute_metrics(stats: Dict[str, int]) -> Dict[str, float]:
        total = stats["processed"]
        if total == 0:
            return {"hit_rate": 0.0, "indexing_rate": 0.0}

        return {
            "hit_rate": round(100 * stats["hits"] / total, 2),
            "indexing_rate": round(100 * stats["indexable"] / total, 2),
        }

    from pathlib import Path


class _Grid:
    """
    Grid search over indexamajig parameters.
    """

    def __init__(
        self,
        directory: Union[str, Path],
        list_file: Union[str, Path],
        geometry: Union[str, Path],
        base_params: Dict[str, Any],
        grid_params: Dict[str, List[Any]],
        cell_file: Optional[Union[str, Path]] = None,
        modules: List[str] | None = None,
        n_subsample: int = 1000,
        n_jobs_per_run: int = 4,
        slurm_directives: Dict[str, Any] | None = None,
        verbose: bool = False,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        self.base_params = base_params
        self.grid_params = grid_params
        self.modules = modules
        self.slurm_directives = slurm_directives or {}
        self.runs: List[Indexamajig] = []
        self._results: Optional[List[Dict[str, Any]]] = None

        subsample_path = self.directory / "subsample.lst"
        subsample_list(list_file, subsample_path, n_subsample)

        keys = list(grid_params.keys())
        combinations = list(itertools.product(*grid_params.values()))

        for i, values in enumerate(combinations):
            grid_vals = dict(zip(keys, values))
            run_params = {**base_params, **grid_vals}

            run = Indexamajig(
                directory=self.directory / f"run_{i:03d}",
                list_file=subsample_path,
                geometry=geometry,
                params=run_params,
                cell_file=cell_file,
                modules=modules,
                n_jobs=n_jobs_per_run,
                slurm_directives=slurm_directives,
                verbose=verbose,
            )
            self.runs.append(run)

    def submit(self) -> List[submitit.Job]:
        """Submit all grid search jobs to the cluster."""
        all_jobs = []
        for run in self.runs:
            all_jobs.extend(run.submit())

        self._save_manifest()
        return all_jobs

    @property
    def results(self) -> List[Dict[str, Any]]:
        """Wait for all jobs and return ranked results."""
        if self._results is not None:
            return self._results

        results = []
        for i, run in enumerate(self.runs):
            run_result = run.results
            run_result["run_id"] = f"run_{i:03d}"
            results.append(run_result)

        results.sort(key=lambda x: x["indexing_rate"], reverse=True)
        self._results = results
        self._save_csv(results)

        return results

    @property
    def best_params(self) -> Dict[str, Any]:
        """Parameters from the best-performing run."""
        if not self.results:
            return {}
        return self.results[0]["params"]

    def _save_manifest(self) -> None:
        manifest = [
            {
                "run_id": f"run_{i:03d}",
                "directory": str(run.directory),
                "params": run.params,
            }
            for i, run in enumerate(self.runs)
        ]
        with open(self.directory / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def _save_csv(self, results: List[Dict[str, Any]]) -> None:
        if not results:
            return

        flat = []
        for r in results:
            row = {k: v for k, v in r.items() if k != "params"}
            row.update(r["params"])
            flat.append(row)

        with open(self.directory / "grid_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat[0].keys())
            writer.writeheader()
            writer.writerows(flat)
