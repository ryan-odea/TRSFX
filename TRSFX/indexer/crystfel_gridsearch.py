import csv
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import submitit

from ._configs import GridSearchConfig, SlurmConfig
from ._utils import generate_clen_geometries, subsample_list


class GridSearch:
    """
    Grid search over indexamajig parameters.

    Subsamples the input list and runs indexamajig with every combination
    of parameters specified in the grid configuration.

    Special handling for 'clen' parameter: instead of passing as a CLI flag,
    generates modified geometry files for each clen value.

    Example
    -------
    >>> config = GridSearchConfig(
    ...     base_params={"indexing": "xgandalf", "peaks": "peakfinder8"},
    ...     grid_params={
    ...         "threshold": [6, 8, 10],
    ...         "clen": [0.120, 0.125, 0.130],  # Special: modifies geometry
    ...     },
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
        try:
            subsample_list(list_file, subsample_path, config.n_subsample)
        except ValueError as e:
            raise ValueError(
                f"Cannot create grid search: {e}\n"
                f"Check that your input list file contains events."
            ) from e

        n_subsampled = sum(1 for ln in subsample_path.read_text().splitlines() if ln.strip())
        if n_subsampled == 0:
            raise ValueError(
                f"Subsampled list is empty. Input file may have no valid events: {list_file}"
            )

        if self.verbose:
            print(f"Subsampled {n_subsampled} events for grid search")

        clen_values = config.grid_params.get("clen")
        clen_geometries = {}
        if clen_values:
            geom_dir = self.directory / "geometries"
            clen_geometries = generate_clen_geometries(geometry, geom_dir, clen_values)
            if self.verbose:
                print(f"Generated {len(clen_geometries)} geometry files for clen scan")

        grid_params_no_clen = {k: v for k, v in config.grid_params.items() if k != "clen"}

        run_idx = 0
        for combo_params in self._iter_combinations_with_clen(config, clen_values):
            run_clen = combo_params.pop("clen", None)
            run_params = {**config.base_params, **combo_params}

            if run_clen is not None:
                run_geometry = clen_geometries[run_clen]
            else:
                run_geometry = geometry

            run = Indexamajig(
                directory=self.directory / f"run_{run_idx:03d}",
                list_file=subsample_path,
                geometry=run_geometry,
                params=run_params,
                cell_file=cell_file,
                modules=modules,
                n_jobs=config.n_jobs_per_run,
                slurm=self.slurm,
                verbose=verbose,
            )
            if run_clen is not None:
                run._grid_params = {**combo_params, "clen": run_clen}
            else:
                run._grid_params = combo_params

            self.runs.append(run)
            run_idx += 1

    def _iter_combinations_with_clen(self, config: GridSearchConfig, clen_values: List[float] | None):
        """Iterate over all parameter combinations including clen."""
        grid_params_no_clen = {k: v for k, v in config.grid_params.items() if k != "clen"}

        if not grid_params_no_clen and not clen_values:
            yield {}
            return

        if grid_params_no_clen:
            keys = list(grid_params_no_clen.keys())
            for values in itertools.product(*grid_params_no_clen.values()):
                base_combo = dict(zip(keys, values))
                if clen_values:
                    for clen in clen_values:
                        yield {**base_combo, "clen": clen}
                else:
                    yield base_combo
        else:
            for clen in clen_values:
                yield {"clen": clen}

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
            print(f"Submitted {len(all_jobs)} jobs across {self.n_runs} parameter combinations")

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
            if hasattr(run, "_grid_params"):
                run_result["grid_params"] = run._grid_params
            results.append(run_result)

        results.sort(key=lambda x: x["indexing_rate"], reverse=True)
        self._results = results
        self._save_csv(results)

        return results

    @property
    def best_params(self) -> Dict[str, Any]:
        """Parameters from the best-performing run (grid params only, including clen if scanned)."""
        if not self.results:
            return {}
        result = self.results[0]
        return result.get("grid_params", result.get("params", {}))

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

        lines.extend([
            "",
            "Best parameters:",
        ])
        for k, v in self.best_params.items():
            lines.append(f"  {k}: {v}")

        return "\n".join(lines)

    def _save_manifest(self) -> None:
        manifest = []
        for i, run in enumerate(self.runs):
            entry = {
                "run_id": f"run_{i:03d}",
                "directory": str(run.directory),
                "params": run.params,
            }
            if hasattr(run, "_grid_params"):
                entry["grid_params"] = run._grid_params
            manifest.append(entry)

        with open(self.directory / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def _save_csv(self, results: List[Dict[str, Any]]) -> None:
        if not results:
            return

        flat = []
        for r in results:
            row = {k: v for k, v in r.items() if k not in ("params", "grid_params")}
            if "grid_params" in r:
                row.update(r["grid_params"])
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
        instance = object.__new__(cls)
        instance.directory = directory
        instance.verbose = False
        instance._results = None

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