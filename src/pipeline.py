"""End-to-end extraction pipeline: file -> chunks -> Ollama -> CSV.

Two operating modes:
  * "Custom instruction" — user supplies a fixed instruction. Each chunk is
    emitted as one CSV row: instruction = user text, input = chunk content,
    output = LLM response. Use this to fine-tune for a specific task.
  * "Auto Q&A" — when the instruction is empty, the LLM is asked to invent
    N samples per chunk in Alpaca format.
"""
import json
import re
import time
from pathlib import Path
from typing import Callable, Optional

from .csv_writer import AlpacaCSVWriter
from .extractors.docx_extractor import extract_doc_legacy, extract_docx
from .extractors.pdf_extractor import extract_pdf
from .ollama_client import generate
from .text_chunker import chunk_text, count_words

# Prompt used in auto Q&A mode (no user instruction provided).
AUTO_PROMPT_TEMPLATE = """You are an expert AI training-data generator. From the source text below, create {n_samples} high-quality instruction-following training samples for fine-tuning a language model.

STRICT RULES:
1. Write the samples in the SAME language as the source text. Vietnamese source -> Vietnamese samples. English source -> English samples. Mixed -> follow the dominant language.
2. "instruction" = a clear, self-contained task, question, or directive about the content.
3. "input" = optional supporting context (use empty string "" when not needed).
4. "output" = the correct, complete answer derived strictly from the source. Do NOT invent facts not present in the source.
5. Cover different aspects: factual recall, summarization, explanation, comparison, analysis.
6. Output ONLY a valid JSON array. No markdown fences. No commentary. No prose before or after.

SOURCE TEXT:
\"\"\"
{chunk}
\"\"\"

Respond with exactly this format:
[
  {{"instruction": "...", "input": "", "output": "..."}},
  {{"instruction": "...", "input": "...", "output": "..."}}
]
"""

# Prompt used in custom-instruction mode.
CUSTOM_PROMPT_TEMPLATE = """{instruction}

---
INPUT:
{chunk}
---

Respond with ONLY the result of applying the instruction above to the INPUT.
Do not repeat the instruction. Do not add preamble, explanation, or markdown fences."""

_FENCE_OPEN = re.compile(r"^\s*```(?:\w+)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```\s*$")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ITEM_RE = re.compile(
    r'\{\s*"instruction"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,'
    r'\s*"input"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,'
    r'\s*"output"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
    re.DOTALL,
)


def extract_text(path: str) -> str:
    """Auto-route to the right extractor based on file extension."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return extract_pdf(path)
    if ext == ".docx":
        return extract_docx(path)
    if ext == ".doc":
        return extract_doc_legacy(path)
    if ext in (".txt", ".md"):
        return p.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Định dạng không hỗ trợ: {ext} ({p.name})")


def _strip_artifacts(text: str) -> str:
    text = _THINK_BLOCK.sub("", text).strip()
    text = _FENCE_OPEN.sub("", text)
    text = _FENCE_CLOSE.sub("", text)
    return text.strip()


def parse_json_response(text: str) -> list[dict]:
    """Robustly extract a JSON array of samples from the model response."""
    if not text:
        return []
    text = _strip_artifacts(text)

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass

    items: list[dict] = []
    for m in _ITEM_RE.finditer(text):
        try:
            items.append({
                "instruction": json.loads(f'"{m.group(1)}"'),
                "input": json.loads(f'"{m.group(2)}"'),
                "output": json.loads(f'"{m.group(3)}"'),
            })
        except json.JSONDecodeError:
            continue
    return items


class Pipeline:
    """Run the extract->chunk->LLM->CSV pipeline for one or more files."""

    def __init__(
        self,
        files: list[str],
        model: str,
        output_csv: str,
        instruction: str = "",
        chunk_words: int = 1500,
        samples_per_chunk: int = 3,
        temperature: float = 0.3,
        num_ctx: int = 8192,
        ollama_url: str = "http://localhost:11434",
        on_log: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[int, int, Optional[float]], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ):
        self.files = files
        self.model = model
        self.output_csv = output_csv
        self.instruction = instruction.strip()
        self.chunk_words = chunk_words
        self.samples_per_chunk = samples_per_chunk
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.ollama_url = ollama_url
        self.on_log = on_log or (lambda m: None)
        self.on_progress = on_progress or (lambda c, t, eta=None: None)
        self.on_status = on_status or (lambda s: None)
        self.should_stop = should_stop or (lambda: False)
        self._chunk_durations: list[float] = []

    def _log(self, msg: str) -> None:
        self.on_log(msg)

    def _status(self, msg: str) -> None:
        self.on_status(msg)

    def _record_and_report(self, cur: int, total: int, elapsed: float) -> None:
        """Track chunk duration, compute rolling-average ETA, report progress."""
        self._chunk_durations.append(elapsed)
        # Use last 5 chunks for stable ETA
        window = self._chunk_durations[-5:]
        avg = sum(window) / len(window) if window else 0.0
        remaining = max(total - cur, 0)
        eta = avg * remaining if remaining else 0.0
        self.on_progress(cur, total, eta)

    def _build_chunks(self) -> list[tuple[str, int, str]]:
        """Read all files and return a flat list of (source_name, global_idx, chunk_text)."""
        all_chunks: list[tuple[str, int, str]] = []
        gid = 0
        for f in self.files:
            if self.should_stop():
                return all_chunks
            try:
                self._status(f"Đang đọc file: {Path(f).name}")
                self._log(f"📄 Đọc: {f}")
                text = extract_text(f)
                wc = count_words(text)
                self._log(f"   → {wc:,} từ")
                if wc == 0:
                    self._log("   ⚠ Không có text. Bỏ qua.")
                    continue
                self._status(f"Đang chia chunks: {Path(f).name}")
                chunks = chunk_text(text, target_words=self.chunk_words)
                self._log(f"   → Chia thành {len(chunks)} chunks")
                source = Path(f).name
                for c in chunks:
                    all_chunks.append((source, gid, c))
                    gid += 1
            except Exception as e:
                self._log(f"   ❌ Lỗi đọc file: {e}")
        return all_chunks

    def _call_llm(self, prompt: str, chunk_idx: int, total: int) -> str:
        """Call Ollama with streaming; update status as tokens arrive."""
        t0 = time.time()
        last_update = [0.0]

        def on_token(piece: str, total_chars: int) -> None:
            now = time.time()
            if now - last_update[0] >= 0.15:  # throttle UI updates ~6 Hz
                last_update[0] = now
                elapsed = now - t0
                self._status(
                    f"Chunk {chunk_idx}/{total} · {total_chars:,} ký tự sinh ra · {elapsed:.1f}s"
                )

        return generate(
            model=self.model,
            prompt=prompt,
            base_url=self.ollama_url,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
            on_token=on_token,
            should_stop=self.should_stop,
        )

    def _process_custom(
        self,
        writer: AlpacaCSVWriter,
        chunks: list[tuple[str, int, str]],
        stats: dict,
    ) -> None:
        total = len(chunks)
        for i, (source, chunk_id, chunk) in enumerate(chunks, start=1):
            if self.should_stop():
                self._log(f"⏹ Đã dừng ở chunk {i}/{total}.")
                break
            t0 = time.time()
            try:
                prompt = CUSTOM_PROMPT_TEMPLATE.format(
                    instruction=self.instruction, chunk=chunk
                )
                response = self._call_llm(prompt, i, total)
                output = _strip_artifacts(response)
                if not output:
                    stats["errors"] += 1
                    self._log(f"   ⚠ Chunk {i}/{total}: model trả về rỗng")
                else:
                    n = writer.write_samples(
                        [{"instruction": self.instruction, "input": chunk, "output": output}],
                        source,
                        chunk_id,
                    )
                    stats["samples"] += n
                    elapsed = time.time() - t0
                    self._log(f"   ✓ Chunk {i}/{total}: +{n} sample ({elapsed:.1f}s, {len(output):,} ký tự)")
                stats["chunks"] += 1
            except Exception as e:
                stats["errors"] += 1
                self._log(f"   ❌ Chunk {i}/{total}: {e}")
            self._record_and_report(i, total, time.time() - t0)

    def _process_auto(
        self,
        writer: AlpacaCSVWriter,
        chunks: list[tuple[str, int, str]],
        stats: dict,
    ) -> None:
        total = len(chunks)
        for i, (source, chunk_id, chunk) in enumerate(chunks, start=1):
            if self.should_stop():
                self._log(f"⏹ Đã dừng ở chunk {i}/{total}.")
                break
            t0 = time.time()
            try:
                prompt = AUTO_PROMPT_TEMPLATE.format(
                    n_samples=self.samples_per_chunk, chunk=chunk
                )
                response = self._call_llm(prompt, i, total)
                samples = parse_json_response(response)
                if not samples:
                    stats["errors"] += 1
                    self._log(f"   ⚠ Chunk {i}/{total}: không parse được JSON")
                else:
                    n = writer.write_samples(samples, source, chunk_id)
                    stats["samples"] += n
                    elapsed = time.time() - t0
                    self._log(f"   ✓ Chunk {i}/{total}: +{n} samples ({elapsed:.1f}s)")
                stats["chunks"] += 1
            except Exception as e:
                stats["errors"] += 1
                self._log(f"   ❌ Chunk {i}/{total}: {e}")
            self._record_and_report(i, total, time.time() - t0)

    def run(self) -> dict:
        stats = {"files": len(self.files), "chunks": 0, "samples": 0, "errors": 0}

        all_chunks = self._build_chunks()
        total = len(all_chunks)
        if total == 0:
            self._log("Không có chunk nào để xử lý.")
            self._status("Không có chunk.")
            return stats

        mode_name = "Custom instruction" if self.instruction else "Auto Q&A"
        self._log(f"🚀 Bắt đầu trích xuất | Mode: {mode_name} | Model: {self.model}")
        if self.instruction:
            preview = self.instruction[:120] + ("..." if len(self.instruction) > 120 else "")
            self._log(f"   Instruction: {preview}")
            self._log(f"   Tổng chunks: {total} (1 sample/chunk)")
        else:
            self._log(f"   Tổng chunks: {total} | samples/chunk: {self.samples_per_chunk}")
        self.on_progress(0, total, None)
        self._status(f"Đang xử lý chunk 0/{total}...")

        with AlpacaCSVWriter(self.output_csv) as writer:
            if self.instruction:
                self._process_custom(writer, all_chunks, stats)
            else:
                self._process_auto(writer, all_chunks, stats)

        self._log(
            f"✅ Xong! Files: {stats['files']} | Chunks: {stats['chunks']} | "
            f"Samples: {stats['samples']} | Lỗi: {stats['errors']}"
        )
        self._log(f"📁 Output: {self.output_csv}")
        self._status(
            f"Hoàn tất: {stats['chunks']} chunks, {stats['samples']} samples, {stats['errors']} lỗi."
        )
        return stats
