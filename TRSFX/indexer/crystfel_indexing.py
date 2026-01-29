import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import submitit

from ._configs import GridSearchConfig, IndexamajigConfig, SlurmConfig
from ._utils import split_list


class Indexamajig:
    """
    Manages parallel indexamajig execution over chunked input lists.

    Splits input into chunks and prepares jobs. Call `submit()` to launch.
    Use `GridSearch` class for parameter optimization.

    Example
    -------
    >>> idx = Indexamajig(
    ...     directory="indexing_output",
    ...     list_file="events.lst",
    ...     geometry="detector.geom",
    ...     params={"indexing": "xgandalf", "peaks": "peakfinder8"},
    ...     cell_file="cell.pdb",
    ...     n_jobs=100,
    ... )
    >>> idx.submit()
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
        slurm: SlurmConfig | None = None,
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
        self.slurm = slurm or SlurmConfig()
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
            directives = self.slurm.to_dict()
            if "slurm_job_name" not in directives:
                directives["slurm_job_name"] = f"idx_{i:04d}"
            executor.update_parameters(**directives)

            func = submitit.helpers.CommandFunction(["bash", "-c", config._cmd_str])
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
        slurm: SlurmConfig | None = None,
        verbose: bool = False,
    ) -> "GridSearch":
        """
        Create a grid search over indexamajig parameters.

        This is a convenience method. For more control, use GridSearch directly.

        Parameters
        ----------
        directory : path
            Output directory for grid search results
        list_file : path
            Input event list file
        geometry : path
            Detector geometry file
        base_params : dict
            Base indexamajig parameters (applied to all runs)
        grid_params : dict
            Parameters to search over. Values must be lists.
        cell_file : path, optional
            Unit cell file
        modules : list, optional
            Environment modules to load
        n_subsample : int
            Number of events to subsample for testing
        n_jobs_per_run : int
            Number of parallel jobs per parameter combination
        slurm : SlurmConfig, optional
            SLURM configuration
        verbose : bool
            Print progress information

        Returns
        -------
        GridSearch
            Configured grid search ready for submission
        """
        from .crystfel_grid_search import GridSearch

        config = GridSearchConfig(
            base_params=base_params,
            grid_params=grid_params,
            n_subsample=n_subsample,
            n_jobs_per_run=n_jobs_per_run,
        )

        return GridSearch(
            directory=directory,
            list_file=list_file,
            geometry=geometry,
            config=config,
            cell_file=cell_file,
            modules=modules,
            slurm=slurm,
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
        slurm: SlurmConfig | None = None,
        align_flags: Dict[str, bool] | None = None,
        verbose: bool = False,
    ) -> "DetectorRefinement":
        """
        Create a detector refinement workflow using Millepede.

        Parameters
        ----------
        directory : path
            Output directory
        list_file : path
            Input event list
        geometry : path
            Initial detector geometry
        cell_file : path
            Unit cell file (required for refinement)
        params : dict
            Indexamajig parameters
        modules : list, optional
            Environment modules to load
        n_jobs : int
            Number of parallel mille jobs
        mille_level : int
            Millepede hierarchy level (1-3)
        slurm : SlurmConfig, optional
            SLURM configuration
        align_flags : dict, optional
            Flags for align_detector (camera_length, out_of_plane, etc.)
        verbose : bool
            Print progress information

        Returns
        -------
        DetectorRefinement
            Configured refinement ready for submission
        """
        from .crystfel_detector import DetectorRefinement

        return DetectorRefinement(
            directory=directory,
            list_file=list_file,
            geometry=geometry,
            cell_file=cell_file,
            params=params,
            modules=modules,
            n_jobs=n_jobs,
            mille_level=mille_level,
            slurm=slurm,
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
