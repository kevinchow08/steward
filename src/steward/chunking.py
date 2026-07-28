"""把提取后的长文本切成适合 embedding 的片段。"""

from dataclasses import asdict, dataclass


@dataclass
class TextChunk:
    """一段可独立生成 embedding 的文本。"""

    index: int
    text: str
    start_offset: int
    end_offset: int

    def to_dict(self):
        return asdict(self)


def chunk_text(text, max_chars=800, overlap_chars=100):
    """按字符数切分文本，优先在换行处断开，并保留少量重叠内容。

    overlap 的作用是避免一个语义句子刚好被切在两个片段之间，
    让相邻片段共享一小段上下文。
    """

    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars 必须大于等于 0 且小于 max_chars")

    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        proposed_end = min(start + max_chars, len(text))
        end = proposed_end

        # 不在片段前半段断开，避免因为很早出现换行而产生过短片段。
        if proposed_end < len(text):
            newline = text.rfind("\n", start + max_chars // 2, proposed_end)
            if newline > start:
                end = newline

        raw_chunk = text[start:end]
        leading_spaces = len(raw_chunk) - len(raw_chunk.lstrip())
        trailing_spaces = len(raw_chunk) - len(raw_chunk.rstrip())
        actual_start = start + leading_spaces
        actual_end = end - trailing_spaces
        chunk_value = text[actual_start:actual_end]

        if chunk_value:
            chunks.append(
                TextChunk(
                    index=chunk_index,
                    text=chunk_value,
                    start_offset=actual_start,
                    end_offset=actual_end,
                )
            )
            chunk_index += 1

        if end >= len(text):
            break

        # 下一段回退一点，保留和上一段的上下文重叠。
        next_start = end - overlap_chars
        start = max(next_start, start + 1)

    return chunks
