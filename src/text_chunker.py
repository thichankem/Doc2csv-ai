"""Smart paragraph/sentence-aware chunking for large documents."""
import re

_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")
_PARA_SPLIT = re.compile(r"\n\s*\n")
_MULTI_NL = re.compile(r"\n{3,}")
_MULTI_SP = re.compile(r"[ \t]+")
_PAGE_NUM_LINE = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
_HYPHEN_LB = re.compile(r"(\w+)-\n(\w+)")


def clean_text(text: str) -> str:
    """Normalize whitespace and fix common PDF artifacts."""
    if not text:
        return ""
    text = _HYPHEN_LB.sub(r"\1\2", text)           # join hyphen line-breaks
    text = _PAGE_NUM_LINE.sub("", text)            # drop bare page numbers
    text = _MULTI_NL.sub("\n\n", text)
    text = _MULTI_SP.sub(" ", text)
    return text.strip()


def count_words(text: str) -> int:
    return len(text.split())


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]


def _split_long_paragraph(para: str, target_words: int) -> list[str]:
    """Break an oversized paragraph at sentence boundaries."""
    sentences = _SENT_SPLIT.split(para)
    chunks: list[str] = []
    buf: list[str] = []
    count = 0
    for s in sentences:
        wc = count_words(s)
        if count + wc > target_words and buf:
            chunks.append(" ".join(buf))
            buf, count = [s], wc
        else:
            buf.append(s)
            count += wc
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def chunk_text(text: str, target_words: int = 1500, min_words: int = 100) -> list[str]:
    """Chunk text into ~target_words segments, prefer paragraph boundaries.

    For huge documents (~1M words) this produces hundreds-to-thousands of chunks
    that can be processed independently by the LLM.
    """
    text = clean_text(text)
    if not text:
        return []

    paragraphs = _split_paragraphs(text)
    chunks: list[str] = []
    buf: list[str] = []
    count = 0

    for para in paragraphs:
        wc = count_words(para)

        if wc > target_words * 2:
            if buf:
                chunks.append("\n\n".join(buf))
                buf, count = [], 0
            chunks.extend(_split_long_paragraph(para, target_words))
            continue

        if count + wc > target_words and count >= min_words:
            chunks.append("\n\n".join(buf))
            buf, count = [para], wc
        else:
            buf.append(para)
            count += wc

    if buf:
        chunks.append("\n\n".join(buf))

    return chunks
