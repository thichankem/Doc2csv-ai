"""Append-mode CSV writer in Alpaca format (instruction/input/output)."""
import csv
from pathlib import Path

COLUMNS = ["instruction", "input", "output", "source", "chunk_id"]


class AlpacaCSVWriter:
    """Write training samples to CSV, header written only once.

    UTF-8 BOM is used so Excel opens Vietnamese text correctly.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = (not self.path.exists()) or self.path.stat().st_size == 0
        self._f = open(self.path, "a", newline="", encoding="utf-8-sig")
        self._w = csv.DictWriter(
            self._f,
            fieldnames=COLUMNS,
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        if is_new:
            self._w.writeheader()
            self._f.flush()
        self.count = 0

    def write_samples(self, samples: list[dict], source: str, chunk_id: int) -> int:
        """Write a batch of samples; rows missing instruction or output are skipped."""
        written = 0
        for s in samples:
            instruction = str(s.get("instruction", "")).strip()
            output = str(s.get("output", "")).strip()
            if not instruction or not output:
                continue
            self._w.writerow({
                "instruction": instruction,
                "input": str(s.get("input", "")).strip(),
                "output": output,
                "source": source,
                "chunk_id": chunk_id,
            })
            written += 1
        if written:
            self._f.flush()
        self.count += written
        return written

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
