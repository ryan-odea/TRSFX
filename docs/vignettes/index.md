# Indexing with Crystfel (`sfx.index`)
This package also has a pipeline to interact with crystfel (defaulted to 0.12.0) through a thin wrapper; hopefully limiting how much input a user needs to specify. This will automatically submit these through a slurm backend. SGE interaction may happen in the future, but is not implemented in it's current state.

## Grid Search for optimal indexing hyperparameters
If you are unsure of your threshold, signal to noise ratio, minimum numer of peaks or pixels, etc. You can create a grid search for the best parameters. While you can give json into the command line, you may also use `sfx.index init` to create a template for you:

### Base-params (non-changing)
```json
{
  "indexing": "xgandalf,asdf,mosflm",
  "peaks": "peakfinder8",
  "int_radius": "3,4,7",
  "multi": true,
  "no_check_peaks": true
}
```

### Grid-Search (n-way change)
```json
{
  "threshold": [
    6,
    8,
    10,
    12
  ],
  "min_snr": [
    4.0,
    4.5,
    5.0
  ],
  "min_peaks": [
    8,
    10,
    12
  ]
}
```

This will create quite a wide variety of different results (care to keep to less than what your cluster allows for maximum number of jobs) and can be now run with-

```bash
sfx.index grid-search -d indexing/ -i h5_files.lst -g geom.geom -p cell.cell --n-subsample 100 --n-jobs 20 --partition day --base-params crystflow_config_base.json --grid-params crystflow_config_grid.json --mem 150
```

This can be analyzed, with best parameters saved with
```bash
sfx.index grid-analyze -d indexing/ -o best.json
```

## Full Indexing Pipeline
Now that we have our optimal settings, you can run the full indexing job (split in parallel) with:

```bash
sfx.index index -d indexing/ -i files.lst -g geom.geom -p cell.cell --params best.json --time 360 --partition day
```

After this finishes you must concatenate the streams together
```bash
sfx.index merge-streams -d indexing/ -o data.stream
```

## Resolving Ambiguities 
Now if you would like to resolve ambiguities, you can follow the above syntax - we are constanly pointing at the same directory and manipulating it's contents

```bash
sfx.index ambigator -d indexing/ -i data.stream -w "6/mmm" -j 1 --time 360 --mem 250 --partition day -y "6/m"
```

## Merging and Partialating
Now that the ambiguities are resolved, we should make our custom split and merge/scale our intensities via partialator.

Make the custom split list-
```bash
sfx.index expand -i sample.lst -o events.lst 
```

```bash
sfx.index partialator -d indexing/ -i indexing/ambigator.stream -y "6/mmm" --unmerged-output unmerged.ifc --custom-split events.lst --time 240 --mem 250 --partition day -o frame
```
