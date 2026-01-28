from pathlib import Path
from typing import List, Union

import h5py
import hdf5plugin
import numpy as np


def read_h5(filename: Union[str, Path]) -> List[np.ndarray]:
    """
    Reads all datasets from an HDF5 file into a flat list of numpy arrays.
    """
    frames = []

    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            if node.shape and node.ndim >= 2:
                try:
                    data = node[()]
                    if data.ndim == 3:
                        frames.extend([d for d in data])
                    else:
                        frames.append(data)
                except OSError as e:
                    print(f"Could not read {name}: {e}")

    with h5py.File(Path(filename), "r") as f:
        f.visititems(visitor_func)

    return frames
