# File Manipulation (`sfx.manip`)

This submodule provides tools to manipulate and change files, rather than either exploring them for data analysis or comparing them against each other. To gather a full list of functions underneath this hood via the CLI, you can always use `sfx.manip --help` to print a list of accessible functions and help documents.

## crystfel-to-meteor

Crystfel and meteor use different data in the creation of their difference maps. Crystfel uses intensities while meteor used amplitudes. This function provides a wrapped around a french-wilson to change these crystfel intensities to amplitudes useable by meteor. As mentioned previously in the meteor section of the documentation, this is done with

```bash
for i in *_crystfel.mtz; do sfx.manip crystfel-to-meteor $i; done
```

## sample-crystals

Sample-crystals intakes a data.stream file and outputs a file with randomly downsampled crystals. 
You can either provide a number of crystals you would like out or a percentage of total crystals indexed.

```bash
sfx.manip sample-crystals data.stream output.stream --count 40000 --seed 2026
```

Which will write out a 40,000 crystal file to `output.stream` (assuming you have as many crystals)

