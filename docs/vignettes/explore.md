# Dataset exploration (`sfx.explore`)
This submodule provides functions to explore and analyze crystallographic data during different steps of the process. To gather a full list of functions underneath this hood via the CLI, you can always use `sfx.explore --help` to print a list of accessible functions and help documents.

## peak-dist
This provides information about the number of peaks detected in a stream file and provides a distribution of the amount of peaks detected per frame. Output is a png file and should be saved as such.

```bash
sfx.explore peak-dist data.stream --output peak_distribution.png --bins 50
```

## consistent-crystals
Finds consistently indexed crystals within a stream file, plots distribution of these consecutive crystals.

```bash
sfx.explore consistent-crystals input.stream output.lst --plot dist.png
```