import numpy as np
import reciprocalspaceship as rs
from reciprocalspaceship.algorithms import scale_merged_intensities


def crystfel_to_meteor(input: str = None, output: str = None) -> None:
    """
    Converts Phenix-processed MTZ files to Meteor-ready MTZ files

    :param input: Input file path
    :type input: str
    :param output: Output file path, if None, appends '_meteor.mtz' to input filename
    :type output: str
    """

    if output is None:
        output = input.replace(".mtz", "_meteor.mtz")

    mtz = rs.read_mtz(input)
    valid = (mtz["IMEAN"] > 0) & (mtz["SIGIMEAN"] > 0)
    valid = valid & np.isfinite(mtz["IMEAN"]) & np.isfinite(mtz["SIGIMEAN"])
    mtz = mtz[valid]

    # French-Wilson
    mtz = scale_merged_intensities(mtz, "IMEAN", "SIGIMEAN")
    mtz.write_mtz(output)
    return output
