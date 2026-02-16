import subprocess
from pathlib import Path
from typing import Union, List


def _run(cmd: List[str], cwd: Path, log_path: Path) -> None:
    """Run a command and redirect stdout/stderr to a log file."""
    with open(log_path, "w") as f:
        subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, check=True)


def statistics(
    stats_dir: Union[str, Path],
    cell_file: Union[str, Path],
    symmetry: str,
    highres: float,
) -> List[Path]:
    """
    Wrap CrystFEL statistics generation for all HKL files in a directory.

    Replicates the behaviour of the provided shell script.

    Returns
    -------
    list of Path
        Paths to generated statistics files.
    """
    stats_dir = Path(stats_dir)
    cell_file = Path(cell_file)

    if not stats_dir.exists():
        raise FileNotFoundError(stats_dir)
    if not cell_file.exists():
        raise FileNotFoundError(cell_file)

    hkls = sorted(stats_dir.glob("*.hkl"))
    if not hkls:
        raise FileNotFoundError(f"No HKL files found in {stats_dir}")

    outputs = []

    for hkl in hkls:
        stem = hkl.name

        check_log = stats_dir / "check.log"
        rsplit_log = stats_dir / "rsplit.log"
        ccstar_log = stats_dir / "ccstar.log"
        cc_log = stats_dir / "cc.log"

        check_dat = stats_dir / "check.dat"
        rsplit_dat = stats_dir / "rsplit.dat"
        ccstar_dat = stats_dir / "ccstar.dat"
        cc_dat = stats_dir / "cc.dat"

        _run(
            [
                "check_hkl",
                stem,
                "-y",
                symmetry,
                "-p",
                str(cell_file),
                f"--highres={highres}",
            ],
            cwd=stats_dir,
            log_path=check_log,
        )
        (stats_dir / "shells.dat").rename(check_dat)

        _run(
            [
                "compare_hkl",
                f"{stem}1",
                f"{stem}2",
                "-p",
                str(cell_file),
                f"--highres={highres}",
                "--fom=rsplit",
                "-y",
                symmetry,
            ],
            cwd=stats_dir,
            log_path=rsplit_log,
        )
        (stats_dir / "shells.dat").rename(rsplit_dat)

        _run(
            [
                "compare_hkl",
                f"{stem}1",
                f"{stem}2",
                "-p",
                str(cell_file),
                f"--highres={highres}",
                "--fom=ccstar",
                "-y",
                symmetry,
            ],
            cwd=stats_dir,
            log_path=ccstar_log,
        )
        (stats_dir / "shells.dat").rename(ccstar_dat)

        _run(
            [
                "compare_hkl",
                f"{stem}1",
                f"{stem}2",
                "-p",
                str(cell_file),
                f"--highres={highres}",
                "--fom=cc",
                "-y",
                symmetry,
            ],
            cwd=stats_dir,
            log_path=cc_log,
        )
        (stats_dir / "shells.dat").rename(cc_dat)

        out = stats_dir / f"statistics_{stem}.dat"
        with open(out, "wb") as w:
            for p in [
                rsplit_log,
                rsplit_dat,
                ccstar_log,
                ccstar_dat,
                cc_log,
                cc_dat,
                check_log,
                check_dat,
            ]:
                with open(p, "rb") as r:
                    w.write(r.read())

        for p in [
            rsplit_log,
            rsplit_dat,
            ccstar_log,
            ccstar_dat,
            cc_log,
            cc_dat,
            check_log,
            check_dat,
        ]:
            p.unlink(missing_ok=True)

        outputs.append(out)

    return outputs
