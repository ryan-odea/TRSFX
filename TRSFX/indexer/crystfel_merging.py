from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import submitit

from ._configs import SlurmConfig


def _build_flags(params: Dict[str, Any]) -> List[str]:
    parts = []
    for k, v in params.items():
        if k in ("n_threads", "j"):
            parts.append(f"-j {v}")
            continue

        flag = k.replace("_", "-")
        if v is True:
            parts.append(f"--{flag}")
        elif v is False or v is None:
            continue
        else:
            parts.append(f"--{flag}={v}")

    return parts


@dataclass
class AmbigatorConfig:
    input_stream: Path
    output_stream: Path
    true_symmetry: str  # -y: true point group symmetry
    apparent_symmetry: str  # -w: apparent lattice symmetry (determines operator)
    params: Dict[str, Any] = field(default_factory=dict)

    def to_cli(self, modules: List[str] | None = None) -> str:
        cmd_parts = []

        if modules:
            cmd_parts.append(" && ".join(f"module load {m}" for m in modules))

        cmd = [
            "ambigator",
            f"-o {self.output_stream}",
            f"-y {self.true_symmetry}",
            f"-w {self.apparent_symmetry}",
        ]
        cmd.extend(_build_flags(self.params))
        cmd.append(str(self.input_stream))

        cmd_parts.append(" ".join(cmd))
        self._cmd_str = " && ".join(cmd_parts)
        return self._cmd_str


class Ambigator:
    """
    Resolves indexing ambiguities when point group symmetry is lower than lattice symmetry.

    Parameters
    ----------
    true_symmetry : str
        The true point group symmetry of the structure (-y)
    apparent_symmetry : str
        The apparent lattice symmetry (-w), used to determine the ambiguity operator
    """

    def __init__(
        self,
        directory: Union[str, Path],
        input_stream: Union[str, Path],
        true_symmetry: str,
        apparent_symmetry: str,
        params: Dict[str, Any] | None = None,
        modules: List[str] | None = None,
        slurm: SlurmConfig | None = None,
        verbose: bool = False,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        self.input_stream = Path(input_stream).resolve()
        self.output_stream = self.directory / "ambigator.stream"
        self.logs_dir = self.directory / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.modules = modules
        self.slurm = slurm or SlurmConfig()
        self.job: Optional[submitit.Job] = None

        self.config = AmbigatorConfig(
            input_stream=self.input_stream,
            output_stream=self.output_stream,
            true_symmetry=true_symmetry,
            apparent_symmetry=apparent_symmetry,
            params=params or {},
        )
        self.config.to_cli(modules)

    def submit(self) -> submitit.Job:
        executor = submitit.AutoExecutor(folder=self.logs_dir)

        directives = self.slurm.to_dict()
        if "slurm_job_name" not in directives:
            directives["slurm_job_name"] = "ambigator"
        executor.update_parameters(**directives)

        func = submitit.helpers.CommandFunction(["bash", "-c", self.config._cmd_str])
        self.job = executor.submit(func)

        return self.job

    @property
    def result(self) -> Path:
        if self.job is None:
            raise RuntimeError("Call submit() first")

        self.job.wait()

        if not self.output_stream.exists():
            raise RuntimeError(
                f"Ambigator completed but {self.output_stream} not found"
            )

        return self.output_stream


@dataclass
class PartialatorConfig:
    input_stream: Path
    output_hkl: Path
    symmetry: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_cli(self, modules: List[str] | None = None) -> str:
        cmd_parts = []

        if modules:
            cmd_parts.append(" && ".join(f"module load {m}" for m in modules))

        cmd = [
            "partialator",
            f"-i {self.input_stream}",
            f"-o {self.output_hkl}",
            f"-y {self.symmetry}",
        ]

        cmd.extend(_build_flags(self.params))

        cmd_parts.append(" ".join(cmd))
        self._cmd_str = " && ".join(cmd_parts)
        return self._cmd_str


class Partialator:
    """Merges and scales crystallographic reflections with partiality correction."""

    def __init__(
        self,
        directory: Union[str, Path],
        input_stream: Union[str, Path],
        symmetry: str,
        output_name: str = "merged",
        params: Dict[str, Any] | None = None,
        modules: List[str] | None = None,
        slurm: SlurmConfig | None = None,
        verbose: bool = False,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        self.input_stream = Path(input_stream).resolve()
        self.output_hkl = self.directory / f"{output_name}.hkl"
        self.logs_dir = self.directory / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.modules = modules
        self.slurm = slurm or SlurmConfig()
        self.job: Optional[submitit.Job] = None

        self.config = PartialatorConfig(
            input_stream=self.input_stream,
            output_hkl=self.output_hkl,
            symmetry=symmetry,
            params=params or {},
        )
        self.config.to_cli(modules)

    def submit(self) -> submitit.Job:
        executor = submitit.AutoExecutor(folder=self.logs_dir)

        directives = self.slurm.to_dict()
        if "slurm_job_name" not in directives:
            directives["slurm_job_name"] = "partialator"
        executor.update_parameters(**directives)

        func = submitit.helpers.CommandFunction(["bash", "-c", self.config._cmd_str])
        self.job = executor.submit(func)

        return self.job

    @property
    def result(self) -> Path:
        if self.job is None:
            raise RuntimeError("Call submit() first")

        self.job.wait()

        if not self.output_hkl.exists():
            raise RuntimeError(f"Partialator completed but {self.output_hkl} not found")

        return self.output_hkl

    @property
    def output_files(self) -> Dict[str, Path]:
        stem = self.output_hkl.stem
        files = {
            "hkl": self.output_hkl,
            "hkl1": self.directory / f"{stem}.hkl1",
            "hkl2": self.directory / f"{stem}.hkl2",
        }

        if self.config.params.get("unmerged-output") or self.config.params.get(
            "unmerged_output"
        ):
            files["unmerged"] = self.directory / f"{stem}.unmerged"

        return files
