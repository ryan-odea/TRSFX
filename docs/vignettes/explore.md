# Dataset exploration (`sfx.explore`)
This submodule provides functions to explore and analyze crystallographic data during different steps of the process. To gather a full list of functions underneath this hood via the CLI, you can always use `sfx.explore --help` to print a list of accessible functions and help documents.

## peak-dist
This provides information about the number of peaks detected in a stream file and provides a distribution of the amount of peaks detected per frame. Output is a png file and should be saved as such.

```bash
sfx.explore peak-dist data.stream --output peak_distribution.png --bins 50
```

## peak-time-series
Sorts by file and frame number to form a time series of detected peaks. Output is a png file and should be saved as such.

```bash
sfx.explore peak-time-series data.stream --output peak_ts.png 
```