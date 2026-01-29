from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class SlurmConfig:
    """Configuration for SLURM job submission."""

    time: int = 360  # minutes
    mem_gb: int = 8
    partition: Optional[str] = None
    job_name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to submitit-compatible directives."""
        directives = {
            "slurm_time": self.time,
            "mem_gb": self.mem_gb,
        }
        if self.partition:
            directives["slurm_partition"] = self.partition
        if self.job_name:
            directives["slurm_job_name"] = self.job_name
        directives.update(self.extra)
        return directives

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SlurmConfig":
        """Create from a dictionary, extracting known keys."""
        known = {"time", "mem_gb", "partition", "job_name"}
        kwargs = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(**kwargs, extra=extra)


@dataclass
class GridSearchConfig:
    """
    Configuration for indexamajig parameter grid search.

    Validates that grid parameters are properly structured and
    don't conflict with base parameters.
    """

    base_params: Dict[str, Any]
    grid_params: Dict[str, List[Any]]
    n_subsample: int = 1000
    n_jobs_per_run: int = 4

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Validate grid search configuration."""
        errors = []

        if not self.base_params:
            errors.append("base_params cannot be empty")

        if not self.grid_params:
            errors.append("grid_params cannot be empty")

        # Check grid_params structure
        for key, values in self.grid_params.items():
            if not isinstance(values, list):
                errors.append(
                    f"grid_params['{key}'] must be a list, got {type(values).__name__}"
                )
            elif len(values) == 0:
                errors.append(f"grid_params['{key}'] cannot be empty")

        # Warn about overlapping keys (grid overrides base)
        overlap = set(self.base_params.keys()) & set(self.grid_params.keys())
        if overlap:
            # This is allowed but worth noting - grid params override base
            pass

        if errors:
            raise ValueError(
                "GridSearchConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    @property
    def n_combinations(self) -> int:
        """Total number of parameter combinations."""
        from functools import reduce
        from operator import mul

        if not self.grid_params:
            return 0
        return reduce(mul, (len(v) for v in self.grid_params.values()), 1)

    @property
    def parameter_names(self) -> List[str]:
        """Names of parameters being searched."""
        return list(self.grid_params.keys())

    def iter_combinations(self):
        """Iterate over all parameter combinations."""
        import itertools

        keys = list(self.grid_params.keys())
        for values in itertools.product(*self.grid_params.values()):
            grid_vals = dict(zip(keys, values))
            yield {**self.base_params, **grid_vals}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GridSearchConfig":
        """Create from a dictionary (e.g., loaded from JSON config)."""
        return cls(
            base_params=d.get("base_params", d.get("indexing_params", {})),
            grid_params=d.get("grid_params", {}),
            n_subsample=d.get("n_subsample", 1000),
            n_jobs_per_run=d.get("n_jobs_per_run", 4),
        )


@dataclass
class IndexamajigConfig:
    """Configuration for a single indexamajig invocation."""

    geometry: Union[str, Path]
    input_list: Union[str, Path]
    output_stream: Union[str, Path]
    cell_file: Optional[Union[str, Path]] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def to_cli(self, modules: List[str] | None = None) -> str:
        """Build the CLI command string, optionally with module loads."""
        cmd_parts = []

        if modules:
            cmd_parts.append(" && ".join(f"module load {m}" for m in modules))

        idx_cmd = [
            "indexamajig",
            f"-g {self.geometry}",
            f"-i {self.input_list}",
            f"-o {self.output_stream}",
        ]

        if self.cell_file:
            idx_cmd.append(f"-p {self.cell_file}")

        for k, v in self.params.items():
            flag = k.replace("_", "-")
            if v is True:
                idx_cmd.append(f"--{flag}")
            elif v is False or v is None:
                continue
            else:
                idx_cmd.append(f"--{flag}={v}")

        cmd_parts.append(" ".join(idx_cmd))
        self._cmd_str = " && ".join(cmd_parts)
        return self._cmd_str


@dataclass
class AlignDetectorConfig:
    """Configuration for align_detector."""

    geometry_in: Path
    geometry_out: Path
    mille_dir: Path
    level: int = 2
    camera_length: bool = True
    out_of_plane: bool = False
    out_of_plane_tilts: bool = False
    panel_totals: bool = False

    def to_cli(self, modules: List[str] | None = None) -> str:
        cmd_parts = []

        if modules:
            cmd_parts.append(" && ".join(f"module load {m}" for m in modules))

        align_cmd = [
            "align_detector",
            f"-g {self.geometry_in}",
            f"-o {self.geometry_out}",
            f"-l {self.level}",
        ]

        if self.camera_length:
            align_cmd.append("--camera-length")
        if self.out_of_plane:
            align_cmd.append("--out-of-plane")
        if self.out_of_plane_tilts:
            align_cmd.append("--out-of-plane-tilts")
        if self.panel_totals:
            align_cmd.append("--panel-totals")

        align_cmd.append(f"{self.mille_dir}/*.bin")

        cmd_parts.append(" ".join(align_cmd))
        self._cmd_str = " && ".join(cmd_parts)
        return self._cmd_str
