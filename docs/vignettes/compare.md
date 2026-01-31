# Comparison of files (`sfx.compare`)
This submodule holds functions to compare different files against each other - usually this would mean 'like' filetypes, e.g. hkl to hkl or mtz to mtz and not between filetypes. To gather a full list of functions underneath this hood via the CLI, you can always use `sfx.compare --help` to print a list of accessible functions and help documents.

## map-cc
This function builds a difference map - difference map correlation matrix and will plot the information for you. Additionally, you can glob files to build a larger correlation matrix without the need to provide it with individual files over and over.
Similarly in 'like' filetypes, you should expect that the file structure is also the same - this function will try to translate the files for you (crystfel or meteor mtz files) but between structure is not officially supported.

```bash
sfx.compare map-cc -g "data/*.mtz" -p heatmap.png
```

## hd5_trace
This function calculates frame to frame correlation coefficients for h5 files. You can provide a single h5 file or glob pattern. Results are written to csv. Plot option will provide a trace plot saved to the same name as the input file (with a .png extension)

```bash
sfx.compare hd5-trace file.h5 --start 0 --stop 45 -i corr.csv --plot
```