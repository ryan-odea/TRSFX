# Included (but not exported) packages

With the installation of this package, you'll find both [`hatTrick`](https://github.com/ryan-odea/hatTrick), [`storageCat`](https://github.com/ryan-odea/storageCat), and [`meteor`](https://github.com/rs-station/meteor) included (but not used elsewhere in the package). All these serve to be utility functions related to the standfuss group at PSI. For these packages, you can consider this a 'metapackage', or a collection of commonly used packages following from a single import call. You can find documentation on their use in their respective repositories; however, I will also echo examples here:

## storageCat

The more useful of the included, not exported, pacakges is `storageCat`. This package is a lightweight tool to make the archival process to [sciCat](https://www.scicatproject.org/) a breeze. Instead of building your metadata file by hand - this package asks you a series of questions through the cli and then builds it for you.

The entire experience around storagecat is meant to be streamlined through three command line calls. First you create, then you check, then you submit (and don't forget to login to scicat and accept the upload!)

```bash
storageCat create

#    |\---/|
#    | ,_, |
#     \_`_/-..----.
#  ___/ `   ' ,""+ \  storageCat
# (__...'   __\    |`.___.';
#  (_,...'(_,.`__)/'.....+
#
#
#=== SciCat Dataset Creation ===
#
#Scientific Metadata (Required):
#Sample name:

storageCat check
storageCat submit
```

## hatTrick (HATRX)

hatTrick is more of a data science package for encoding and decoding of crystallographic files following the hadamard transform. The steps are relatively simple - first you provide raw crystallographic files to be encoded follinwg the hadamard transform, then you proceed as normal through your indexing pipeline. The decoding step occurs when you have hkl files. For this example, we will use `crystfel`.

### Example Usage (With Crystfel)

While this package does support access through the python api, it is more simple to build a pipeline along the command line. This makes interacting with crystfel simpler as you can immediately place everything into a job script without the need for an intermediate python script.

For our example, let's assume we have 45 frames, each of 5ms, and want to do a rank 3 encoding of our data.

#### (1) Encoding Data

This step intakes a list file containing information of where to find your .h5 data as well as information about your data (how many frames you would like to encode, how many frames exist in your data).

```bash
LIST_FILE={DATA.LST}
N_MERGED_FRAMES=3           # Must be prime and ≡ 3 (mod 4): 3, 7, 11, 19, 23, 31...
N_FRAMES=45
DATA_LOCATION="entry/data"
DATA_NAME="data"
OUTDIR="hadamard_outputs"
N_FILES=8                   # Process 8 files in parallel
WORKERS_PER_MERGE=4         # Workers per single merge

export LIST_FILE N_MERGED_FRAMES DATA_LOCATION DATA_NAME OUTDIR WORKERS_PER_MERGE

mkdir -p "$OUTDIR" logs

run_hadamard_merge() {
    local INPUT_FILE="$1"
    local BASENAME=$(basename "$INPUT_FILE" .h5)
    local OUTPUT_FILE="${OUTDIR}/${BASENAME}_hadamard.h5"
    
    hatrx encode \
        -f "$INPUT_FILE" \
        -o "$OUTPUT_FILE" \
        --n-frames "$N_FRAMES" \
        --n-merged-frames "$N_MERGED_FRAMES" \
        --type hadamard \
        --data-location "$DATA_LOCATION" \
        --data-name "$DATA_NAME" \
        --n-workers "$WORKERS_PER_MERGE"
}

export -f run_hadamard_merge

cat "$LIST_FILE" | parallel -j "$N_FILES" run_hadamard_merge {}
```

#### (2) Indexing, Resolving Ambiguities, Scaling with Crystfel

As normal, you will index, resolve any ambiguities, and scale. Here, we provide this process with Crystfel, noting that you would need to change indexing parameters to those which best fits your data. Anecdotally, hadamard encoded files preform best with a higher SNR compared to your standard data processing hyperparameters.

For our example, let's assume a threshold of 8, SNR of 4.0, min-peaks of 8, and min-pix of 2 were the optimal hyperparameters found. In some of our data, we've found that each of the encoded files arrives at the same optimal hyperparameters; however, it's probably wisest to test each of the sets.

```shell
# 110 Encoded Files
indexamajig -g {.GEOM} -i {LIST_110} -o {110.STREAM} --indexing=xgandalf,asdf,mosflm,taketwo --peaks=peakfinder8 --int-radius=3,4,7 --multi --no-check-peaks --threshold=8 --min-snr=4.0 --min-peaks=8 --min-pix-count=2 -p {.CELL}

# 101 Encoded Files
indexamajig -g {.GEOM} -i {LIST_101} -o {101.STREAM} --indexing=xgandalf,asdf,mosflm,taketwo --peaks=peakfinder8 --int-radius=3,4,7 --multi --no-check-peaks --threshold=8 --min-snr=4.0 --min-peaks=8 --min-pix-count=2 -p {.CELL}

# 011 Encoded Files
indexamajig -g {.GEOM} -i {LIST_011} -o {011.STREAM} --indexing=xgandalf,asdf,mosflm,taketwo --peaks=peakfinder8 --int-radius=3,4,7 --multi --no-check-peaks --threshold=8 --min-snr=4.0 --min-peaks=8 --min-pix-count=2 -p {.CELL}

# Combine Streams
cat 110.STREAM 101.STREAM 011.STREAM > data.stream

# Resolve ambiguities together
ambigator -o {AMBI.STREAM} -w 6/mmm --lowres=10.0 --highres=3.0 --ncorr=1000 -j 32 --symmetry=6/m data.stream

# Scale (Custom output split with lists created before x n_frames)
partialator -o data_110 -y 6/m -i {AMBI.STREAM} --model=xsphere --iterations=1 --push-res=1.5 -j 32 --custom-split={LIST_110}

partialator -o data_101 -y 6/m -i {AMBI.STREAM} --model=xsphere --iterations=1 --push-res=1.5 -j 32 --custom-split={LIST_101}

partialator -o data_011 -y 6/m -i {AMBI.STREAM} --model=xsphere --iterations=1 --push-res=1.5 -j 32 --custom-split={LIST_011}
```

#### (3) Decoding

Now that we have our encoded data in hkl format, with ambiguities resolved and appropriately scaled together, we can decode the data to retrieve our normal hkls.

```shell
hatrx decode -n 3 -p "data_110-*.hkl" -p "data_101-*.hkl" -p "data_011-*.hkl" -o .
```

## Meteor

`Meteor` is a tool for computing difference maps, specializing in the identification of weak signals. In addition to being very powerful at creating nice looking difference maps, meteor is also very simple to use. Phaseboost being particularly interesting at developing good looking difference maps.

The primary challenge is, if you come from phenix, you must translate the mtz files to files expected by meteor. This package provides such cli tools.

```bash
for i in *_crystfel.mtz; do sfx.manip crystfel-to-meteor $i; done
```

After this, you can use meteor as you would like, I find phaseboost particularly good at creating nice difference maps:
```bash
for i in *.mtz; 
do meteor.phaseboost $i /path/to/dark -o meteor_map_${i} -s /path/to/pdb -m metadata_${i}.json; 
done
```