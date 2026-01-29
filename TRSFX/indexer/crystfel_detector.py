from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import submitit

from ._configs import AlignDetectorConfig, IndexamajigConfig, SlurmConfig
from ._utils import split_list


class DetectorRefinement:
    """
    Two-stage detector geometry refinement using Millepede.

    Generates calibration data then aligns detector panels.

    Example
    -------
    >>> ref = DetectorRefinement(
    ...     directory="refinement_output",
    ...     list_file="events.lst",
    ...     geometry="detector.geom",
    ...     cell_file="cell.pdb",
    ...     params={"indexing": "xgandalf"},
    ... )
    >>> ref.submit()  # Submit mille generation jobs
    >>> ref.align()   # Wait for mille, then submit alignment
    >>> print(ref.refined_geometry)  # Path to refined geometry
    """

    def __init__(
        self,
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
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        self.geometry = Path(geometry).resolve()
        self.cell_file = Path(cell_file).resolve()
        self.modules = modules
        self.slurm = slurm or SlurmConfig()
        self.mille_level = mille_level

        self.mille_dir = self.directory / "mille_bins"
        self.lists_dir = self.directory / "lists"
        self.logs_dir = self.directory / "logs"

        for d in [self.mille_dir, self.lists_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.mille_jobs: List[submitit.Job] = []
        self.align_job: Optional[submitit.Job] = None
        self.configs: List[IndexamajigConfig] = []

        chunks = split_list(list_file, self.lists_dir, n_jobs)

        for i, chunk in enumerate(chunks):
            mille_params = dict(params)
            mille_params["mille"] = True
            mille_params["mille_file"] = str(self.mille_dir / f"mille_{i:03d}.bin")
            mille_params["max_mille_level"] = mille_level
            mille_params["check_peaks"] = True

            config = IndexamajigConfig(
                geometry=self.geometry,
                input_list=chunk,
                output_stream=self.directory / f"trash_{i:03d}.stream",
                cell_file=self.cell_file,
                params=mille_params,
            )
            config.to_cli(modules)
            self.configs.append(config)

        flags = align_flags or {}
        self.align_config = AlignDetectorConfig(
            geometry_in=self.geometry,
            geometry_out=self.directory / "refined.geom",
            mille_dir=self.mille_dir,
            level=mille_level,
            camera_length=flags.get("camera_length", True),
            out_of_plane=flags.get("out_of_plane", False),
            out_of_plane_tilts=flags.get("out_of_plane_tilts", False),
            panel_totals=flags.get("panel_totals", False),
        )
        self.align_config.to_cli(modules)

    def submit(self) -> List[submitit.Job]:
        """Submit mille data generation jobs."""
        executor = submitit.AutoExecutor(folder=self.logs_dir)

        for i, config in enumerate(self.configs):
            directives = self.slurm.to_dict()
            directives["slurm_job_name"] = f"mille_{i:03d}"
            executor.update_parameters(**directives)

            func = submitit.helpers.CommandFunction(["bash", "-c", config._cmd_str])
            job = executor.submit(func)
            self.mille_jobs.append(job)

        return self.mille_jobs

    def align(self, slurm: SlurmConfig | None = None) -> submitit.Job:
        """Wait for mille jobs and submit the alignment job."""
        for job in self.mille_jobs:
            job.wait()

        bin_files = list(self.mille_dir.glob("*.bin"))
        if not bin_files:
            raise RuntimeError(f"No .bin files found in {self.mille_dir}")

        align_logs = self.directory / "align_logs"
        align_logs.mkdir(exist_ok=True)

        align_slurm = slurm or SlurmConfig(
            time=self.slurm.time,
            mem_gb=max(self.slurm.mem_gb, 150),
            partition=self.slurm.partition,
            job_name="align_detector",
        )

        directives = align_slurm.to_dict()
        executor = submitit.AutoExecutor(folder=align_logs)
        executor.update_parameters(**directives)

        func = submitit.helpers.CommandFunction(
            ["bash", "-c", self.align_config._cmd_str]
        )
        self.align_job = executor.submit(func)

        return self.align_job

    @property
    def refined_geometry(self) -> Path:
        """Wait for alignment and return path to refined geometry."""
        if self.align_job is None:
            raise RuntimeError("Call align() first")

        self.align_job.wait()

        geom_path = self.align_config.geometry_out
        if not geom_path.exists():
            raise RuntimeError(f"Alignment completed but {geom_path} not found")

        return geom_path

    @property
    def job_ids(self) -> Dict[str, Any]:
        return {
            "mille": [j.job_id for j in self.mille_jobs],
            "align": self.align_job.job_id if self.align_job else None,
        }
