from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


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
