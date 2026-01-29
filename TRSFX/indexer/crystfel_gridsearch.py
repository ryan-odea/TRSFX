import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import submitit

from ._configs import GridSearchConfig, SlurmConfig
from ._utils import subsample_list


class GridSearch:
    """
    Grid search over indexamajig parameters.

    Subsamples the input list and runs indexamajig with every combination
    of parameters specified in the grid configuration.

    Example
    -------
    >>> config = GridSearchConfig(
    ...     base_params={"indexing": "xgandalf", "peaks": "peakfinder8"},
    ...     grid_params={"threshold": [6, 8, 10], "min_snr": [4.0, 5.0]},
    ...     n_subsample=1000,
    ... )
    >>> gs = GridSearch(
    ...     directory="grid_output",
    ...     list_file="events.lst",
    ...     geometry="detector.geom",
    ...     config=config,
    ... )
    >>> gs.submit()
    """

    def __init__(
        self,
        directory: Union[str, Path],
        list_file: Union[str, Path],
        geometry: Union[str, Path],
        config: GridSearchConfig,
        cell_file: Optional[Union[str, Path]] = None,
        modules: List[str] | None = None,
        slurm: SlurmConfig | None = None,
        verbose: bool = False,
    ):
        from .crystfel_indexing import Indexamajig

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        self.config = config
        self.modules = modules
        self.slurm = slurm or SlurmConfig()
        self.runs: List[Indexamajig] = []
        self._results: Optional[List[Dict[str, Any]]] = None

        subsample_path = self.directory / "subsample.lst"
        subsample_list(list_file, subsample_path, config.n_subsample)

        for i, run_params in enumerate(config.iter_combinations()):
            run = Indexamajig(
                directory=self.directory / f"run_{i:03d}",
                list_file=subsample_path,
                geometry=geometry,
                params=run_params,
                cell_file=cell_file,
                modules=modules,
                n_jobs=config.n_jobs_per_run,
                slurm=self.slurm,
                verbose=verbose,
            )
            self.runs.append(run)

    @property
    def n_runs(self) -> int:
        """Number of parameter combinations to test."""
        return len(self.runs)

    @property
    def n_total_jobs(self) -> int:
        """Total number of SLURM jobs that will be submitted."""
        return self.n_runs * self.config.n_jobs_per_run

    def submit(self) -> List[submitit.Job]:
        """Submit all grid search jobs to the cluster."""
        all_jobs = []
        for run in self.runs:
            all_jobs.extend(run.submit())

        self._save_manifest()

        if self.verbose:
            print(
                f"Submitted {len(all_jobs)} jobs across {self.n_runs} parameter combinations"
            )

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

    @property
    def best_run(self) -> Optional[Dict[str, Any]]:
        """Full result dict from the best-performing run."""
        if not self.results:
            return None
        return self.results[0]

    def summary(self) -> str:
        """Return a formatted summary of results."""
        if not self.results:
            return "No results available. Call submit() and wait for jobs to complete."

        lines = [
            f"Grid Search Results ({self.n_runs} combinations)",
            "=" * 50,
            "",
            "Top 5 results:",
        ]

        for i, r in enumerate(self.results[:5], 1):
            lines.append(
                f"  {i}. {r['run_id']}: "
                f"{r['indexing_rate']:.2f}% indexing, "
                f"{r['hit_rate']:.2f}% hits"
            )

        lines.extend(
            [
                "",
                "Best parameters:",
            ]
        )
        for k, v in self.best_params.items():
            lines.append(f"  {k}: {v}")

        return "\n".join(lines)

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

    @classmethod
    def from_manifest(cls, directory: Union[str, Path]) -> "GridSearch":
        """
        Reconstruct a GridSearch from a previously saved manifest.

        Useful for analyzing results after jobs have completed.
        """
        from .crystfel_indexing import Indexamajig

        directory = Path(directory)
        manifest_path = directory / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest.json found in {directory}")

        manifest = json.loads(manifest_path.read_text())

        # Create a minimal instance for result analysis
        instance = object.__new__(cls)
        instance.directory = directory
        instance.verbose = False
        instance._results = None

        # Reconstruct run objects (minimal, for log parsing)
        instance.runs = []
        for entry in manifest:
            run = object.__new__(Indexamajig)
            run.directory = Path(entry["directory"])
            run.logs_dir = run.directory / "logs"
            run.params = entry["params"]
            run.jobs = []
            instance.runs.append(run)

        return instance

    @classmethod
    def analyze(cls, directory: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze completed grid search results.

        Returns dict with 'results', 'best_params', and 'summary'.
        """
        gs = cls.from_manifest(directory)
        return {
            "results": gs.results,
            "best_params": gs.best_params,
            "summary": gs.summary(),
        }
