import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    """
    Represents a data chunk from within a stream file
    """

    filename: str
    event: str
    serial_number: int
    hit: bool
    indexed_by: str
    num_peaks: int
    raw: str
    crystals: list = field(default_factory=list)

    @property
    def event_number(self) -> int:
        """
        Extract event number from event string
        """
        match = re.search(r"//(\d+)", self.event)
        return int(match.group(1)) if match else 0

    @property
    def sort_key(self) -> str:
        """
        Key for natural sorting of chunks
        """
        return (self.filename, self.event_number)


@dataclass
class Stream:
    """
    Represents a full stream file
    """

    preamble: str
    chunks: list[Chunk]

    @property
    def hits(self) -> list[Chunk]:
        """
        List of chunks that are hits
        """
        return [chunk for chunk in self.chunks if chunk.hit]

    @property
    def non_hits(self) -> list[Chunk]:
        """
        List of chunks that are non-hits
        """
        return [chunk for chunk in self.chunks if not chunk.hit]

    @property
    def n_crystals(self) -> int:
        """
        Total number of crystals across all chunks
        """
        return sum(len(chunk.crystals) for chunk in self.chunks)


def read_stream(filepath: str | Path) -> Stream:
    """
    Parse stream file into structured data

    :param filepath: Filepath for stream file
    :type filepath: str | Path
    :return: Stream object containing parsed data
    :rtype: Stream
    """

    content = Path(filepath).read_text()

    chunk_pattern = r"----- Begin chunk -----"
    parts = re.split(chunk_pattern, content)
    preamble = parts[0].rstrip()
    chunks = []

    for t in parts[1:]:
        end_match = re.search(r"----- End chunk -----", t)
        if end_match:
            chunk_content = t[: end_match.end()]
        else:
            chunk_content = t

        filename = re.search(r"Image filename: (.+)", chunk_content)
        event = re.search(r"Event: (.+)", chunk_content)
        serial = re.search(r"Image serial number: (\d+)", chunk_content)
        hit = re.search(r"hit = (\d+)", chunk_content)
        indexed = re.search(r"indexed_by = (.+)", chunk_content)
        peaks = re.search(r"num_peaks = (\d+)", chunk_content)

        crystals = []
        crystal_blocks = re.findall(
            r"--- Begin crystal.*?--- End crystal", chunk_content, re.DOTALL
        )

        for c in crystal_blocks:
            crystals.append(c)

        chunk = Chunk(
            filename=filename.group(1) if filename else "",
            event=event.group(1) if event else "",
            serial_number=int(serial.group(1)) if serial else 0,
            hit=hit.group(1) == "1" if hit else False,
            indexed_by=indexed.group(1) if indexed else "none",
            num_peaks=int(peaks.group(1)) if peaks else 0,
            raw=chunk_content,
            crystals=crystals,
        )

        chunks.append(chunk)

    return Stream(preamble=preamble, chunks=chunks)


def write_stream(preamble: str, chunks: list[Chunk], output: str | Path) -> None:
    """
    Write structured stream data back to file

    :param preamble: Preamble text for stream file
    :type preamble: str
    :param chunks: List of Chunk objects to write
    :type chunks: list[Chunk]
    :param output: Output file path
    :type output: str | Path
    """
    output = Path(output)
    with output.open("w") as f:
        f.write(preamble + "\n")
        for c in chunks:
            f.write(c.raw)
            if not c.raw.endswith("\n"):
                f.write("\n")
