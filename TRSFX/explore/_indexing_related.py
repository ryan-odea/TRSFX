from .._utils.stream_io import Stream, Chunk

def get_consistent_crystals(stream: Stream) -> list[str]:
    """
    Find files where every frame was successfully indexed (has crystals).

    :param stream: Parsed Stream object
    :type stream: Stream
    :return: List of filenames where all frames have crystals
    :rtype: list[str]
    """
    from collections import defaultdict

    chunks_by_file: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in stream.chunks:
        chunks_by_file[chunk.filename].append(chunk)

    consistent_files = []
    for filename, chunks in chunks_by_file.items():
        if all(len(c.crystals) > 0 for c in chunks):
            consistent_files.append(filename)

    return sorted(consistent_files)